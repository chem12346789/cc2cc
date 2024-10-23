from itertools import product

import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad

import opt_einsum as oe

from cc2cc.utils import gen_basis, process_input, Grid
from cc2cc.utils import (
    DATA_PATH,
    AU2KCALMOL,
    CUBE_SIZE,
    CUBE_LEN,
    CUBE_MIDDLE,
)


def cc(molecular, name, args):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """
    mol = pyscf.M(
        atom=molecular,
        basis=gen_basis(
            molecular,
            args.basis,
            args.if_basis_str,
        ),
        spin=0,
    )

    print(mol.atom)
    print(f"Generate data for {name}")

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    e_cc = mycc.e_tot

    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    grids = Grid(mol, level=1)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
    rho_cc_all = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="mGGA")
    exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]

    dm2_cc = mycc.make_rdm2(ao_repr=True)
    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * (dm2_cc - oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc))
        + 0.05 * oe.contract("pr,qs->pqrs", dm1_cc, dm1_cc),
        (mol.nao,),
        (mol.nao,),
        (mol.nao, mol.nao),
        constants=[0],
        optimize="optimal",
    )

    for i, coord in enumerate(grids.coords):
        if i * 10 % len(grids.coords) == 0:
            print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)

        ao_0_i = ao_value[0][i]
        if abs(rho_cc[0][i]) < 1e-14:
            continue
        with mol.with_rinv_origin(coord):
            rinv = mol.intor("int1e_rinv")
            exc_over_dm_cc_grids[i] += expr_rinv_dm2_r(
                ao_0_i,
                ao_0_i,
                rinv,
                backend="torch",
            ) / (rho_cc[0][i] + 1e-14)

    rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
    for p, p_coords in enumerate(grids.coords):
        if p * 10 % len(grids.coords) == 0:
            print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

        coords_cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        for i, j, k in product(range(CUBE_SIZE), repeat=3):
            coords_cube[i, j, k] = p_coords + [
                (i - CUBE_MIDDLE) * CUBE_LEN,
                (j - CUBE_MIDDLE) * CUBE_LEN,
                (k - CUBE_MIDDLE) * CUBE_LEN,
            ]
        coor_cube[p] = coords_cube.copy()
        coords_cube = coords_cube.reshape(-1, 3)

        ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=1)
        rho_cube_p = pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_cc, xctype="GGA")
        rho_cube[p] = rho_cube_p.reshape(4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

    error_energy = e_cc - mdft.energy_tot(dm1_cc)
    error = np.sum(exc_over_dm_cc_grids * grids.weights * rho_cc[0]) - error_energy
    print(f"error_energy: {AU2KCALMOL * error_energy}, Error: {AU2KCALMOL * error}")

    np.savez_compressed(
        DATA_PATH / f"data_{name}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_inv_4_norm=rho_cc,
        rho_inv_4_norm_matrix=process_input(rho_cc, grids),
        weights=grids.weights,
        weights_matrix=grids.vector_to_matrix(grids.weights),
        exc_over_dm_cc_grids=exc_over_dm_cc_grids,
        exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
        exc_over_dm_b3lyp_grids=-pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0],
        exc_over_dm_b3lyp_grids_matrix=grids.vector_to_matrix(
            -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]
        ),
        rho_cc_all=rho_cc_all,
        rho_cube=rho_cube,
        coor_cube=coor_cube,
        error_energy=error_energy,
    )


def cc_change_cube(molecular, name, args):
    """
    Modify cube data for the CCSD method.
    """
    file_path = DATA_PATH / f"data_{name}.npz"
    if file_path.exists():
        print(f"Data {name} already exists.")

        data = np.load(file_path)
        e_cc = data["e_cc"]
        dm1_cc = data["dm_cc"]
        rho_cc = data["rho_inv_4_norm"]
        exc_over_dm_cc_grids = data["exc_over_dm_cc_grids"]
        error_energy = data["error_energy"]
        rho_cc_all = data["rho_cc_all"]

        mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                args.basis,
                args.if_basis_str,
            ),
            spin=0,
        )
        grids = Grid(mol, level=1)

        rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        for p, p_coords in enumerate(grids.coords):
            if p * 10 % len(grids.coords) == 0:
                print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

            coords_cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
            for i, j, k in product(range(CUBE_SIZE), repeat=3):
                coords_cube[i, j, k] = p_coords + [
                    (i - CUBE_MIDDLE) * CUBE_LEN,
                    (j - CUBE_MIDDLE) * CUBE_LEN,
                    (k - CUBE_MIDDLE) * CUBE_LEN,
                ]
            coor_cube[p] = coords_cube.copy()
            coords_cube = coords_cube.reshape(-1, 3)

            ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=1)
            rho_cube_p = pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_cc, xctype="GGA")
            rho_cube[p] = rho_cube_p.reshape(4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

        np.savez_compressed(
            DATA_PATH / f"data_{name}.npz",
            e_cc=e_cc,
            dm_cc=dm1_cc,
            rho_inv_4_norm=rho_cc,
            rho_inv_4_norm_matrix=process_input(rho_cc, grids),
            weights=grids.weights,
            weights_matrix=grids.vector_to_matrix(grids.weights),
            exc_over_dm_cc_grids=exc_over_dm_cc_grids,
            exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
            rho_cc_all=rho_cc_all,
            rho_cube=rho_cube,
            coor_cube=coor_cube,
            error_energy=error_energy,
        )


def cc_add_data(molecular, name, args):
    """
    Append data for the CCSD method.
    """
    file_path = DATA_PATH / f"data_{name}.npz"
    if file_path.exists():
        print(f"Data {name} already exists.")

        data = np.load(file_path)
        e_cc = data["e_cc"]
        dm1_cc = data["dm_cc"]
        rho_cc = data["rho_inv_4_norm"]
        exc_over_dm_cc_grids = data["exc_over_dm_cc_grids"]
        error_energy = data["error_energy"]
        rho_cube = data["rho_cube"]
        coor_cube = data["coor_cube"]
        rho_cc_all = data["rho_cc_all"]

        mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                args.basis,
                args.if_basis_str,
            ),
            spin=0,
        )
        grids = Grid(mol, level=1)

        np.savez_compressed(
            DATA_PATH / f"data_{name}.npz",
            e_cc=e_cc,
            dm_cc=dm1_cc,
            rho_inv_4_norm=rho_cc,
            rho_inv_4_norm_matrix=process_input(rho_cc, grids),
            weights=grids.weights,
            weights_matrix=grids.vector_to_matrix(grids.weights),
            exc_over_dm_cc_grids=exc_over_dm_cc_grids,
            exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
            rho_cc_all=rho_cc_all,
            rho_cube=rho_cube,
            coor_cube=coor_cube,
            error_energy=error_energy,
        )
