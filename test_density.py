"""
Test the model.
Other parameter are from the argparse.
"""

import argparse
from itertools import product
from pathlib import Path
import datetime
import os

import torch
import numpy as np
import pyscf
import opt_einsum as oe

from pyscf.dft.numint import _dot_ao_dm, _contract_rho

from cc2cc import add_args, extend
from cc2cc.utils import ModelDict, MAIN_PATH
from cc2cc.utils import rotate

from cc2cc.utils import gen_basis, process_input, Grid
from cc2cc.utils import (
    DATA_PATH,
    AU2KCALMOL,
    CUBE_SIZE,
    CUBE_LEN,
    CUBE_MIDDLE,
    STRUCTURE,
    LEVEL,
    PERIOD,
)
from cc2cc.test_rks import TEST_DATA

# from cadft.utils.ModelDict_xy import ModelDict
# from cadft.utils import ModelDict_xy1 as ModelDict
# from cadft.utils.ModelDict_xy2 import ModelDict

# class ModelDict_data()
if __name__ == "__main__":
    # 0. Prepare the args
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Init the model
    modeldict = ModelDict(
        load=args.load,
        device=device,
        precision=args.precision,
        if_mkdir=False,
        load_epoch=args.load_epoch,
    )
    modeldict.load_model()
    modeldict.eval()

    # 2. Test loop
    df_dict = {
        "name": [],
        "error_scf_ene": [],
        "error_dft_ene": [],
        "abs_cc_ene": [],
        "density_diff_scf": [],
        "density_diff_dft": [],
        "dipole_diff_scf": [],
        "dipole_diff_dft": [],
        "force_diff_scf": [],
        "force_diff_dft": [],
    }

    name_mol_now = args.name_mol[0]

    df_dict_path = Path(
        f"{MAIN_PATH}/validate/ccdft_{args.load}_{datetime.datetime.today():%Y-%m-%d-%H-%M-%S}.csv"
    )

    for (
        name_mol,
        extend_atom,
        extend_xyz,
        distance,
    ) in product(
        args.name_mol,
        args.extend_atom,
        args.extend_xyz,
        args.distance_list,
    ):
        molecular, name = extend(
            name_mol, extend_atom, extend_xyz, distance, args.basis
        )
        if molecular is None:
            print(f"Skip: {name:>40}")
            continue
        # rotate(molecular)
        rotation = rotate(molecular, rotation="r")

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

        mdft = pyscf.scf.RKS(mol)
        mdft.xc = "b3lyp"
        mdft.kernel()

        name = f"{name_mol}_{args.basis}_{extend_atom}_{extend_xyz}_{distance:.4f}"
        data_path = Path(f"{DATA_PATH}") / f"data_{name}_{LEVEL}_{PERIOD}.npz"
        if not (data_path).exists():
            print(f"No file: {data_path}")
            continue
        else:
            print(f"Load the data: {data_path}")
        data = np.load(data_path)

        mf = pyscf.scf.RHF(mol)
        mf.kernel()
        mycc = pyscf.cc.CCSD(mf)
        mycc.kernel()
        dm1_cc = mycc.make_rdm1(ao_repr=True)
        e_cc = mycc.e_tot

        grids = Grid(mol, level=LEVEL, period=PERIOD)
        # coords = grids.coords
        # weights = grids.weights
        coords = data["coor"]
        coords = (rotation @ coords.T).T
        print(f"Number of grids: {len(coords)}")
        weights = data["weights"]

        weights_save = data["weights"]
        ao_value = pyscf.dft.numint.eval_ao(mol, coords, deriv=2)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value[:4], dm1_cc, xctype="GGA")
        rho_cc_save = data["rho_inv_4_norm"]

        if "exc_over_dm_mrks_grids" in data.files:
            output_ = data["exc_over_dm_mrks_grids"]
        else:
            output_ = data["exc_over_dm_cc_grids"]
            print(
                f"{AU2KCALMOL * np.sum(output_ * rho_cc[0] * weights):.2f} kcal/mol\n"
            )

        if STRUCTURE == "cnn3d":
            rho_cc_1 = np.zeros((3, len(coords)))
            rho_cc_2 = np.zeros((3, 3, len(coords)))
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
            rho_cube = np.zeros((len(coords), 2, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
            coor_cube = np.zeros((len(coords), CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 3))
            rho_cube_save = data["rho_cube"]
            coor_cube_save = data["coor_cube"]
            for p, p_coords in enumerate(coords):
                if p * 10 % len(coords) == 0:
                    print(f"Progress: {(p*100)/len(coords):.1f}%", flush=True)

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

                ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=1)
                rho_cube_p = pyscf.dft.numint.eval_rho(
                    mol, ao_cube, dm1_cc, xctype="mGGA"
                )
                rho_cube_p_norm = np.zeros((3, CUBE_SIZE * CUBE_SIZE * CUBE_SIZE))
                rho_cube_p_norm[0, :] = rho_cube_p[0, :]
                rho_cube_p_norm[1, :] = (
                    rho_cube_p[1, :] ** 2
                    + rho_cube_p[2, :] ** 2
                    + rho_cube_p[3, :] ** 2
                ) ** (1 / 2)
                rho_cube_p_norm[2, :] = rho_cube_p[4, :]
                rho_cube[p] = rho_cube_p_norm.reshape(
                    3, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE
                )

                # if np.linalg.norm(rho_cube[p] - rho_cube_save[p]) > 1e-10:
                #     print(
                #         f"Error: {np.linalg.norm(coor_cube[p] - coor_cube_save[p]):.2e}"
                #     )
                #     print(
                #         f"Error: {coor_cube[p][0, 0, 0, :]}"
                #         f"{rotation @ coor_cube_save[p][0, 0, 0, :]}"
                #     )
                #     print(
                #         f"Error: {coor_cube[p][1, 1, 1, :]}"
                #         f"{rotation @ coor_cube_save[p][1, 1, 1, :]}"
                #     )
                #     print(
                #         f"Error: {rho_cube[p][:, 0, 0, 0]}"
                #         f"{rho_cube_save[p][:, 0, 0, 0]}"
                #     )
                #     print(
                #         f"Error: {rho_cube[p][:, 1, 1, 1]}"
                #         f"{rho_cube_save[p][:, 1, 1, 1]}"
                #     )
            input_mat = torch.tensor(rho_cube, dtype=modeldict.dtype).to("cuda")
        elif STRUCTURE == "unet":
            input_mat = process_input(rho_cc, grids)
            input_mat = np.transpose(input_mat, (1, 0, 2, 3))
            input_mat = input_mat[:, [0], :, :]
            input_mat = torch.tensor(input_mat, dtype=modeldict.dtype).to("cuda")

        with torch.no_grad():
            output_mat = modeldict.model(input_mat)

        if STRUCTURE == "cnn3d":
            correct_ene = output_mat.cpu().detach().numpy()[:, 0]
        elif STRUCTURE == "unet":
            correct_ene = grids.matrix_to_vector(
                (output_mat.cpu().detach().numpy())[:, 0, :, :]
            )
        else:
            raise ValueError("Unknown structure.")

        exc_over_dm_cc_predict = (
            correct_ene * rho_cc[0] * weights - output_ * rho_cc_save[0] * weights_save
        )
        print(
            f"ERROR: {AU2KCALMOL * np.sum(exc_over_dm_cc_predict):.2f} kcal/mol\n",
            f"ABS ERROR: {AU2KCALMOL * np.sum(np.abs(exc_over_dm_cc_predict)):.2f} kcal/mol\n",
            f"GRIDS ERROR AI: {AU2KCALMOL * (e_cc - mdft.energy_tot(mycc.make_rdm1(ao_repr=True)) - np.sum(correct_ene * rho_cc[0] * weights)):.2f} kcal/mol\n",
            f"GRIDS ERROR CC: {AU2KCALMOL * (e_cc - mdft.energy_tot(dm1_cc) - np.sum(output_ * rho_cc_save[0] * weights_save)):.2f} kcal/mol\n",
        )
        print(AU2KCALMOL * (mycc.e_tot - data["e_cc"]))
        print(AU2KCALMOL * (mdft.e_tot - data["e_cc"]))
        print(np.linalg.norm(mycc.make_rdm1(ao_repr=True) - data["dm_cc"]))
