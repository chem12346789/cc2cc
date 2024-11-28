from itertools import product

import numpy as np
import pyscf
from pyscf.dft.numint import _dot_ao_dm, _contract_rho

from cc2cc.utils import gen_basis, process_input, Grid
from cc2cc.utils import (
    DATA_PATH,
    CUBE_SIZE,
    CUBE_LEN,
    CUBE_MIDDLE,
    CUBE_USE_MIDDLE,
    AU2KCALMOL,
    LEVEL,
    PERIOD,
)


def mrks(molecular, name, args):
    """
    Generate data for the CCSD method. (Restrict scenario to spin 0).
    """
    file_path = DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz"
    if file_path.exists():
        print(f"Data {name}_{LEVEL}_{PERIOD} already exists.")
        data = np.load(file_path)
        dm1_cc = data["dm_cc"]
        e_cc = data["e_cc"]
        dm1_inv = data["dm_inv"]
        weights = data["weights"]
        print(data["mol_atom"])
        print(molecular)

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
        vxc1_lda = grids.matrix_to_vector(data["vxc1_lda"])
        exc1_tr_lda = grids.matrix_to_vector(data["exc1_tr_lda"])
        exc1_tr = grids.matrix_to_vector(data["exc1_tr"])
        weights = grids.matrix_to_vector(data["weights"])
        weights_matrix = grids.vector_to_matrix(weights)

        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
        rho_inv = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_inv, xctype="GGA")
        rho_inv_4_norm_matrix = process_input(rho_inv, grids)
        print(
            np.linalg.norm(
                rho_inv_4_norm_matrix[0, :, :, :] - data["rho_inv_4_norm"][0, :, :, :]
            )
        )
        rho_inv_1 = np.zeros((3, len(grids.coords)))
        rho_inv_2 = np.zeros((3, 3, len(grids.coords)))
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()

        error_energy = np.sum(exc1_tr_lda * rho_inv[0] * weights)
        mf = pyscf.scf.RHF(mol)
        mf.kernel()
        h1e = mf.get_hcore()
        hcore_vj_energy = (
            np.sum(h1e * dm1_inv)
            + 0.5 * np.sum(mf.get_jk(mol, dm1_inv, 1)[0] * dm1_inv)
            + mol.energy_nuc()
        )
        print(
            2625.5 * ((hcore_vj_energy + np.sum(exc1_tr * rho_inv[0] * weights)) - e_cc)
        )

        # Hessian matrix
        assert (
            np.linalg.norm(dm1_inv.conj().T - dm1_inv) < 1e-10
        ), "Density matrix is not symmetric."
        c0 = _dot_ao_dm(mol, ao_value[0], dm1_inv, None, shls_slice, ao_loc)
        rho_inv_1[0, :] = _contract_rho(ao_value[1], c0)
        rho_inv_1[1, :] = _contract_rho(ao_value[2], c0)
        rho_inv_1[2, :] = _contract_rho(ao_value[3], c0)
        rho_inv_2[0, 0, :] = _contract_rho(ao_value[4], c0)
        rho_inv_2[0, 1, :] = _contract_rho(ao_value[5], c0)
        rho_inv_2[0, 2, :] = _contract_rho(ao_value[6], c0)
        rho_inv_2[1, 1, :] = _contract_rho(ao_value[7], c0)
        rho_inv_2[1, 2, :] = _contract_rho(ao_value[8], c0)
        rho_inv_2[2, 2, :] = _contract_rho(ao_value[9], c0)
        rho_inv_2[1, 0, :] = rho_inv_2[0, 1, :]
        rho_inv_2[2, 0, :] = rho_inv_2[0, 2, :]
        rho_inv_2[2, 1, :] = rho_inv_2[1, 2, :]

        # rho_cube = np.zeros((len(grids.coords), 3, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
        coor_cube = np.zeros((len(grids.coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
        for p, p_coords in enumerate(grids.coords):
            if p * 10 % len(grids.coords) == 0:
                print(f"Progress: {(p*100)/len(grids.coords):.1f}%", flush=True)

            norm_2d = rho_inv_2[:, :, p]
            eig_val, eig_vec = np.linalg.eigh(norm_2d)
            eig_val_sort = np.argsort(eig_val)
            eig_vec = eig_vec[:, eig_val_sort]
            norm_1d = rho_inv_1[:, p]
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
                mol, ao_cube, dm1_inv, xctype="mGGA", with_lapl=False
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
            rho_cube[p] = np.reshape(
                rho_cube_p_norm, (4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            )
        print(
            AU2KCALMOL
            * np.sum(
                exc1_tr_lda
                * (
                    rho_cube[:, 0, CUBE_USE_MIDDLE, CUBE_USE_MIDDLE, CUBE_USE_MIDDLE]
                    / (-3 / 4 * (3 / np.pi) ** (1 / 3))
                )
                ** 3
                * weights
            )
        )
        print(AU2KCALMOL * error_energy)

        np.savez_compressed(
            DATA_PATH / f"data_{name}_{LEVEL}_{PERIOD}.npz",
            dm_cc=dm1_cc,
            dm_inv=dm1_inv,
            error_energy=error_energy,
            rho_inv=rho_inv,
            weights=weights,
            weights_matrix=weights_matrix,
            rho_inv_4_norm_matrix=rho_inv_4_norm_matrix,
            vxc_over_dm_mrks_grids=vxc1_lda,
            exc_over_dm_mrks_grids=exc1_tr_lda,
            rho_cube=rho_cube,
            coor_cube=coor_cube,
            coor=grids.coords,
        )
    else:
        print(f"Data {name}_{LEVEL}_{PERIOD} does not exist.")
