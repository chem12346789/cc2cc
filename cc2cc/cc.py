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
    e_cc = mycc.e_tot

    mdft = pyscf.scf.RKS(mol)
    mdft.xc = "b3lyp"
    mdft.kernel()

    grids = Grid(mol, level=LEVEL, period=PERIOD)
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
    exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]
    exc_over_dm_b3lyp_grids = exc_over_dm_cc_grids.copy()
    rho_cc_1 = np.zeros((3, len(grids.coords)))
    rho_cc_2 = np.zeros((3, 3, len(grids.coords)))
    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()

    # Hessian matrix
    assert (
        np.linalg.norm(dm1_cc.conj().T - dm1_cc) < 1e-10
    ), "Density matrix is not symmetric."
    c0 = _dot_ao_dm(mol, ao_value[0], dm1_cc, None, shls_slice, ao_loc)
    rho_cc_1[0, :] = _contract_rho(ao_value[1], c0)
    rho_cc_1[1, :] = _contract_rho(ao_value[2], c0)
    rho_cc_1[2, :] = _contract_rho(ao_value[3], c0)
    rho_cc_2[0, 0, :] = _contract_rho(ao_value[4], c0)
    rho_cc_2[0, 1, :] = _contract_rho(ao_value[5], c0)
    rho_cc_2[0, 2, :] = _contract_rho(ao_value[6], c0)
    rho_cc_2[1, 1, :] = _contract_rho(ao_value[7], c0)
    rho_cc_2[1, 2, :] = _contract_rho(ao_value[8], c0)
    rho_cc_2[2, 2, :] = _contract_rho(ao_value[9], c0)
    rho_cc_2[1, 0, :] = rho_cc_2[0, 1, :]
    rho_cc_2[2, 0, :] = rho_cc_2[0, 2, :]
    rho_cc_2[2, 1, :] = rho_cc_2[1, 2, :]

    dm2_cc = mycc.make_rdm2(ao_repr=True)
    expr_rinv_dm2_r = oe.contract_expression(
        "ijkl,i,j,kl->",
        0.5 * dm2_cc
        - 0.5 * oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
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

    error_energy = e_cc - mdft.energy_tot(dm1_cc)
    error = np.sum(exc_over_dm_cc_grids * grids.weights * rho_cc[0]) - error_energy
    print(f"error_energy: {AU2KCALMOL * error_energy}, Error: {AU2KCALMOL * error}")

    rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
    coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
    for p, p_coords in enumerate(grids.coords):
        if p * 10 % len(grids.coords) == 0:
            print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

        norm_2d = rho_cc_2[:, :, p]
        eig_val, eig_vec = np.linalg.eigh(norm_2d)
        eig_val_sort = np.argsort(eig_val)
        eig_vec = eig_vec[:, eig_val_sort]
        norm_1d = rho_cc_1[:, p]
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
            mol, ao_cube, dm1_cc, xctype="mGGA", with_lapl=False
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
        rho_inv_4_norm=rho_cc,
        rho_inv_4_norm_matrix=process_input(rho_cc, grids),
        weights=grids.weights,
        weights_matrix=grids.vector_to_matrix(grids.weights),
        exc_over_dm_cc_grids=exc_over_dm_cc_grids,
        exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
        exc_over_dm_b3lyp_grids=exc_over_dm_b3lyp_grids,
        exc_over_dm_b3lyp_grids_matrix=grids.vector_to_matrix(exc_over_dm_b3lyp_grids),
        rho_cube=rho_cube,
        coor_cube=coor_cube,
        coor=grids.coords,
        error_energy=error_energy,
    )


def cc_change_cube(molecular, name, args):
    """
    Modify cube data for the CCSD method.
    """
    file_path = DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz"
    if file_path.exists():
        print(f"Data {name}_{LEVEL}_{PERIOD} already exists.")
        data = np.load(file_path)
        e_cc = data["e_cc"]
        dm1_cc = data["dm_cc"]
        rho_cc = data["rho_inv_4_norm"]
        exc_over_dm_cc_grids = data["exc_over_dm_cc_grids"]
        exc_over_dm_b3lyp_grids = data["exc_over_dm_b3lyp_grids"]
        error_energy = data["error_energy"]
        weights = data["weights"]
        weights_matrix = data["weights_matrix"]

        mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                args.basis,
                args.if_basis_str,
            ),
            spin=0,
        )

        grids = Grid(mol, level=LEVEL, period=PERIOD)
        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        rho_cc_1 = np.zeros((3, len(grids.coords)))
        rho_cc_2 = np.zeros((3, 3, len(grids.coords)))
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()

        # Hessian matrix
        assert (
            np.linalg.norm(dm1_cc.conj().T - dm1_cc) < 1e-10
        ), "Density matrix is not symmetric."
        c0 = _dot_ao_dm(mol, ao_value[0], dm1_cc, None, shls_slice, ao_loc)
        rho_cc_1[0, :] = _contract_rho(ao_value[1], c0)
        rho_cc_1[1, :] = _contract_rho(ao_value[2], c0)
        rho_cc_1[2, :] = _contract_rho(ao_value[3], c0)
        rho_cc_2[0, 0, :] = _contract_rho(ao_value[4], c0)
        rho_cc_2[0, 1, :] = _contract_rho(ao_value[5], c0)
        rho_cc_2[0, 2, :] = _contract_rho(ao_value[6], c0)
        rho_cc_2[1, 1, :] = _contract_rho(ao_value[7], c0)
        rho_cc_2[1, 2, :] = _contract_rho(ao_value[8], c0)
        rho_cc_2[2, 2, :] = _contract_rho(ao_value[9], c0)
        rho_cc_2[1, 0, :] = rho_cc_2[0, 1, :]
        rho_cc_2[2, 0, :] = rho_cc_2[0, 2, :]
        rho_cc_2[2, 1, :] = rho_cc_2[1, 2, :]

        rho_cube = np.zeros((len(grids.coords), 3, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        # rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        for p, p_coords in enumerate(grids.coords):
            if p * 10 % len(grids.coords) == 0:
                print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

            norm_2d = rho_cc_2[:, :, p]
            eig_val, eig_vec = np.linalg.eigh(norm_2d)
            eig_val_sort = np.argsort(eig_val)
            eig_vec = eig_vec[:, eig_val_sort]
            norm_1d = rho_cc_1[:, p]
            for i in range(3):
                if eig_vec[:, i] @ norm_1d < 0:
                    eig_vec[:, i] *= -1

            assert (
                np.linalg.norm(eig_vec @ eig_vec.T - np.eye(3)) < 1e-8
            ), f"Eigenvectors are not orthogonal.{eig_vec}"
            assert np.linalg.norm(
                eig_vec @ np.diag(eig_val) @ eig_vec.T - norm_2d
            ) < 1e-8 * np.linalg.norm(
                norm_2d
            ), f"Eigenvalues are not correct.{eig_vec @ np.diag(eig_val) @ eig_vec.T, norm_2d}"

            coords_cube = np.zeros((CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
            for i, j, k in product(range(CUBE_SIZE), repeat=3):
                coords_cube[i, j, k, :] = (
                    p_coords
                    + (i - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 0]
                    + (j - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 1]
                    + (k - CUBE_MIDDLE) * CUBE_LEN * eig_vec[:, 2]
                )
            coor_cube[p] = coords_cube.copy()
            # print(f"coords_cube: {coords_cube}")
            coords_cube = coords_cube.reshape(-1, 3)

            ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=1)
            rho_cube_p = pyscf.dft.numint.eval_rho(
                mol, ao_cube, dm1_cc, xctype="mGGA", with_lapl=False
            )

            rho_cube_p_norm = np.zeros((3, CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
            rho_cube_p_norm[0, :] = rho_cube_p[0, :]
            rho_cube_p_norm[1, :] = (
                rho_cube_p[1, :] ** 2 + rho_cube_p[2, :] ** 2 + rho_cube_p[3, :] ** 2
            )
            rho_cube_p_norm[2, :] = rho_cube_p[4, :]
            rho_cube[p] = np.reshape(
                rho_cube_p_norm, (3, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            )

            # exc_slater = pyscf.dft.libxc.eval_xc("SLATER,", rho_cube_p[0])[0]
            # exc_b88 = pyscf.dft.libxc.eval_xc("B88,", rho_cube_p[:4])[0]
            # exc_lyp = pyscf.dft.libxc.eval_xc(",LYP", rho_cube_p[:4])[0]
            # exc_vwn = pyscf.dft.libxc.eval_xc(",VWN3", rho_cube_p[0])[0]
            # rho_cube_p_norm = np.zeros((4, CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
            # rho_cube_p_norm[0, :] = exc_slater
            # rho_cube_p_norm[1, :] = exc_b88
            # rho_cube_p_norm[2, :] = exc_lyp
            # rho_cube_p_norm[3, :] = exc_vwn
            # rho_cube[p] = np.reshape(
            #     rho_cube_p_norm, (4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            # )

        np.savez_compressed(
            DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
            e_cc=e_cc,
            dm_cc=dm1_cc,
            rho_inv_4_norm=rho_cc,
            rho_inv_4_norm_matrix=process_input(rho_cc, grids),
            weights=weights,
            weights_matrix=weights_matrix,
            exc_over_dm_cc_grids=exc_over_dm_cc_grids,
            exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
            exc_over_dm_b3lyp_grids=exc_over_dm_b3lyp_grids,
            exc_over_dm_b3lyp_grids_matrix=grids.vector_to_matrix(
                exc_over_dm_b3lyp_grids
            ),
            rho_cube=rho_cube,
            coor_cube=coor_cube,
            coor=grids.coords,
            error_energy=error_energy,
        )
    else:
        print(f"Data {name}_{LEVEL}_{PERIOD} does not exist.")


def cc_add_data(molecular, name, args):
    """
    Append data for the CCSD method.
    """
    file_path = DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz"
    if file_path.exists():
        print(f"Data {name}_{LEVEL}_{PERIOD} already exists.")

        data = np.load(file_path)
        error_energy = data["error_energy"]
        rho_cube = data["rho_cube"]
        coor_cube = data["coor_cube"]

        mol = pyscf.M(
            atom=molecular,
            basis=gen_basis(
                molecular,
                args.basis,
                args.if_basis_str,
            ),
            spin=0,
        )
        grids = Grid(mol, level=LEVEL, period=PERIOD)

        mf = pyscf.scf.RHF(mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        e_cc = mycc.e_tot

        mdft = pyscf.scf.RKS(mol)
        mdft.xc = "b3lyp"
        mdft.kernel()

        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        print(np.linalg.norm(rho_cc - data["rho_inv_4_norm"]))

        exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]
        exc_over_dm_b3lyp_grids = exc_over_dm_cc_grids.copy()

        dm2_cc = mycc.make_rdm2(ao_repr=True)
        expr_rinv_dm2_r = oe.contract_expression(
            "ijkl,i,j,kl->",
            0.5 * dm2_cc
            - 0.5 * oe.contract("pq,rs->pqrs", dm1_cc, dm1_cc)
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
            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                exc_over_dm_cc_grids[i] = (
                    expr_rinv_dm2_r(
                        ao_0_i,
                        ao_0_i,
                        rinv,
                        backend="torch",
                    )
                    + rho_cc[0][i] * exc_over_dm_cc_grids[i]
                    + 1e-14
                ) / (rho_cc[0][i] + 1e-12)

        error_energy = e_cc - mdft.energy_tot(dm1_cc)
        error = np.sum(exc_over_dm_cc_grids * grids.weights * rho_cc[0]) - error_energy
        print(f"error_energy: {AU2KCALMOL * error_energy}, Error: {AU2KCALMOL * error}")

        np.savez_compressed(
            DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
            e_cc=e_cc,
            dm_cc=dm1_cc,
            rho_inv_4_norm=rho_cc,
            rho_inv_4_norm_matrix=process_input(rho_cc, grids),
            weights=grids.weights,
            weights_matrix=grids.vector_to_matrix(grids.weights),
            exc_over_dm_cc_grids=exc_over_dm_cc_grids,
            exc_over_dm_cc_grids_matrix=grids.vector_to_matrix(exc_over_dm_cc_grids),
            exc_over_dm_b3lyp_grids=exc_over_dm_b3lyp_grids,
            exc_over_dm_b3lyp_grids_matrix=grids.vector_to_matrix(
                exc_over_dm_b3lyp_grids
            ),
            rho_cube=rho_cube,
            coor_cube=coor_cube,
            coor=grids.coords,
            error_energy=error_energy,
        )
