from itertools import product

import numpy as np
import pyscf

# from pyscf.grad import ccsd as ccsd_grad
from pyscf.dft.numint import _dot_ao_dm, _contract_rho
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

    print(f"Generate data for {name}")

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()
    dm1_cc = mycc.make_rdm1(ao_repr=True)
    dm2_cc = mycc.make_rdm2(ao_repr=True)
    e_cc = mycc.e_tot

    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()
    dm1_dft = mdft.make_rdm1(ao_repr=True)

    grids = Grid(mol, level=LEVEL, period=PERIOD)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
    exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_dft)[0]

    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * dm2_cc
        - 0.5 * oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
        + 0.05 * oe.contract("pr,qs->pqrs", dm1_dft, dm1_dft),
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
            exc_over_dm_cc_grids[i] += expr_rinv_dm2_r(
                ao_0_i,
                ao_0_i,
                rinv,
                backend="torch",
            ) / (rho_dft[0][i] + 1e-14)

    kin = mol.intor("int1e_kin")
    error_kin = np.einsum("pq,pq", kin, dm1_cc) - np.einsum("pq,pq", kin, dm1_dft)
    nuc = mol.intor("int1e_nuc")
    error_nuc = np.einsum("pq,pq", nuc, dm1_cc) - np.einsum("pq,pq", nuc, dm1_dft)
    eri = mol.intor("int2e")
    error_eris = 0.5 * np.einsum("pqrs,pq,rs", eri, dm1_cc, dm1_cc) - 0.5 * np.einsum(
        "pqrs,pq,rs", eri, dm1_dft, dm1_dft
    )
    error_energy = e_cc - error_kin - error_nuc - error_eris - mdft.energy_tot(dm1_dft)
    error = np.sum(exc_over_dm_cc_grids * grids.weights * rho_dft[0]) - error_energy
    print(
        f"error_energy: {AU2KCALMOL * error_energy},",
        f"Error: {AU2KCALMOL * error},",
        f"Exact Error kin: {AU2KCALMOL * error_kin}",
        f"Exact Error nuc: {AU2KCALMOL * error_nuc}",
        f"Exact Error eri: {AU2KCALMOL * error_eris}",
    )

    dm1_input = dm1_dft.copy()
    # Hessian matrix
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    rho_input = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
    rho_input_1 = np.zeros((3, len(grids.coords)))
    rho_input_2 = np.zeros((3, 3, len(grids.coords)))
    assert (
        np.linalg.norm(dm1_input.conj().T - dm1_input) < 1e-10
    ), "Density matrix is not symmetric."
    c0 = _dot_ao_dm(mol, ao_value[0], dm1_input, None, shls_slice, ao_loc)
    rho_input_1[0, :] = _contract_rho(ao_value[1], c0)
    rho_input_1[1, :] = _contract_rho(ao_value[2], c0)
    rho_input_1[2, :] = _contract_rho(ao_value[3], c0)
    rho_input_2[0, 0, :] = _contract_rho(ao_value[4], c0)
    rho_input_2[0, 1, :] = _contract_rho(ao_value[5], c0)
    rho_input_2[0, 2, :] = _contract_rho(ao_value[6], c0)
    rho_input_2[1, 1, :] = _contract_rho(ao_value[7], c0)
    rho_input_2[1, 2, :] = _contract_rho(ao_value[8], c0)
    rho_input_2[2, 2, :] = _contract_rho(ao_value[9], c0)
    rho_input_2[1, 0, :] = rho_input_2[0, 1, :]
    rho_input_2[2, 0, :] = rho_input_2[0, 2, :]
    rho_input_2[2, 1, :] = rho_input_2[1, 2, :]

    rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
    for p, p_coords in enumerate(grids.coords):
        if p * 10 % len(grids.coords) == 0:
            print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

        norm_2d = rho_input_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_input_1[:, p]
        for i in range(3):
            if eig_vec[:, i] @ norm_1d < 0:
                eig_vec[:, i] *= -1

        coords_cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        for i, j, k in product(range(CUBE_SIZE), repeat=3):
            coords_cube[i, j, k, :] = (
                p_coords
                + (i - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 0]
                + (j - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 1]
                + (k - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 2]
            )
        coor_cube[p] = coords_cube.copy()
        coords_cube = coords_cube.reshape(-1, 3)

        ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=2)
        rho_cube_p = pyscf.dft.numint.eval_rho(
            mol, ao_cube, dm1_input, xctype="mGGA", with_lapl=False
        )

        # rho_cube_p_norm = np.zeros((3, CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
        # rho_cube_p_norm[0, :] = rho_cube_p[0, :]
        # rho_cube_p_norm[1, :] = (
        #     rho_cube_p[1, :] ** 2 + rho_cube_p[2, :] ** 2 + rho_cube_p[3, :] ** 2
        # )
        # rho_cube_p_norm[2, :] = rho_cube_p[4, :]
        # rho_cube[p] = np.reshape(
        #     rho_cube_p_norm, (3, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        # )

        exc_slater = pyscf.dft.libxc.eval_xc("SLATER,", rho_cube_p[0])[0]
        exc_b88 = pyscf.dft.libxc.eval_xc("B88,", rho_cube_p[:4])[0]
        exc_lyp = pyscf.dft.libxc.eval_xc(",LYP", rho_cube_p[:4])[0]
        exc_vwn = pyscf.dft.libxc.eval_xc(",VWN3", rho_cube_p[0])[0]
        rho_cube_p_norm = np.zeros((4, CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
        rho_cube_p_norm[0, :] = exc_slater
        rho_cube_p_norm[1, :] = exc_b88
        rho_cube_p_norm[2, :] = exc_lyp
        rho_cube_p_norm[3, :] = exc_vwn
        rho_cube[p] = np.reshape(rho_cube_p_norm, (4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

    np.savez_compressed(
        DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
        e_cc=e_cc,
        dm_cc=dm1_cc,
        rho_inv_4_norm=rho_input,
        rho_inv_4_norm_matrix=process_input(rho_input, grids),
        weights=grids.weights,
        weights_matrix=grids.vector_to_matrix(grids.weights),
        exc_over_dm_cc_grids=exc_over_dm_cc_grids,
        exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
        rho_cube=rho_cube,
        coor_cube=coor_cube,
        coor=grids.coords,
        error_energy=error_energy,
    )
