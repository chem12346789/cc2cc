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
    LEVEL,
    PERIOD,
)


def mrks(molecular, name, args):
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
    print(f"Generate data for data_{name}_{LEVEL}_{PERIOD}")

    data = np.load(DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz")
    dm1_inv = data["dm_inv"]

    grids = Grid(mol, level=LEVEL, period=PERIOD)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
    rho_inv = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_inv, xctype="GGA")
    exc_over_dm_b3lyp_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_inv)[0]
    exc_over_dm_cc_1_k_grids = np.zeros_like(exc_over_dm_b3lyp_grids)
    exc_over_dm_lda_grids = -pyscf.dft.libxc.eval_xc("lda", rho_inv[0])[0]

    expr_rinv_dm1_k_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.05 * oe.contract("pr,qs->pqrs", dm1_inv, dm1_inv),
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
        with mol.with_rinv_origin(coord):
            rinv = mol.intor("int1e_rinv")
            exc_over_dm_cc_1_k_grids[i] += expr_rinv_dm1_k_r(
                ao_0_i,
                ao_0_i,
                rinv,
                backend="torch",
            ) / (rho_inv[0][i] + 1e-14)

    print(
        f"0: {AU2KCALMOL * np.sum(exc_over_dm_b3lyp_grids * grids.weights * rho_inv[0])}"
    )
    print(
        f"1: {AU2KCALMOL * np.sum(exc_over_dm_cc_1_k_grids * grids.weights * rho_inv[0])}"
    )

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
        rho_cube_p = pyscf.dft.numint.eval_rho(mol, ao_cube, dm1_inv, xctype="GGA")
        rho_cube[p] = rho_cube_p.reshape(4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

    vxc = grids.matrix_to_vector(data["vxc"])
    exc1_tr = grids.matrix_to_vector(data["exc1_tr"])
    vxc1_lda = grids.matrix_to_vector(data["vxc1_lda"])
    exc1_tr_lda = grids.matrix_to_vector(data["exc1_tr_lda"])
    weights = grids.matrix_to_vector(data["weights"])
    print(np.sum(np.abs(weights - grids.weights)))

    np.savez_compressed(
        DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
        dm_cc=data["dm_cc"],
        dm_inv=data["dm_inv"],
        weights=weights,
        vxc=vxc,
        exc1_tr=exc1_tr,
        vxc_over_dm_mrks_grids=vxc1_lda,
        exc_over_dm_mrks_grids=exc1_tr_lda,
        exc_over_dm_lda_grids=exc_over_dm_lda_grids,
        exc_over_dm_b3lyp_grids=exc_over_dm_b3lyp_grids,
        exc_over_dm_cc_1_k_grids=exc_over_dm_cc_1_k_grids,
        rho_cube=rho_cube,
        coor_cube=coor_cube,
    )


def mrks_append(molecular, name, args):
    mol = pyscf.M(
        atom=molecular,
        basis=gen_basis(
            molecular,
            args.basis,
            args.if_basis_str,
        ),
        spin=0,
    )

    data = np.load(DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz")

    grids = Grid(mol, level=LEVEL, period=PERIOD)
    weights = grids.matrix_to_vector(data["weights"])

    np.savez_compressed(
        DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
        dm_cc=data["dm_cc"],
        dm_inv=data["dm_inv"],
        weights=weights,
        vxc=data["vxc"],
        exc1_tr=data["exc1_tr"],
        vxc_over_dm_mrks_grids=data["vxc_over_dm_mrks_grids"],
        exc_over_dm_mrks_grids=data["exc_over_dm_mrks_grids"],
        exc_over_dm_lda_grids=data["exc_over_dm_lda_grids"],
        exc_over_dm_b3lyp_grids=data["exc_over_dm_b3lyp_grids"],
        exc_over_dm_cc_1_k_grids=data["exc_over_dm_cc_1_k_grids"],
        rho_cube=data["rho_cube"],
        coor_cube=data["coor_cube"],
    )
