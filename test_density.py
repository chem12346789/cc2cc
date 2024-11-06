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
        rotate(molecular)

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

        grids = Grid(mol, level=1, period=1)
        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=1)
        rho_cc = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc, xctype="GGA")
        exc_over_dm_cc_grids = np.zeros_like(rho_cc[0])

        dm2_cc = mycc.make_rdm2(ao_repr=True)
        exc_over_dm_cc_grids = -pyscf.dft.libxc.eval_xc("b3lyp", rho_cc)[0]
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

        error_energy = e_cc - mdft.energy_tot(dm1_cc)
        error = np.sum(exc_over_dm_cc_grids * grids.weights * rho_cc[0]) - error_energy
        print(f"error_energy: {AU2KCALMOL * error_energy}, Error: {AU2KCALMOL * error}")

        if STRUCTURE == "cnn3d":
            rho_cube = np.zeros((len(grids.coords), 4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
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

                coords_cube = coords_cube.reshape(-1, 3)
                ao_cube = pyscf.dft.numint.eval_ao(mol, coords_cube, deriv=1)
                rho_cube_p = pyscf.dft.numint.eval_rho(
                    mol, ao_cube, dm1_cc, xctype="GGA"
                )
                rho_cube[p] = rho_cube_p.reshape(4, CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

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

        exc_over_dm_cc_predict = (
            (correct_ene - exc_over_dm_cc_grids) * rho_cc[0] * grids.weights
        )
        print(
            AU2KCALMOL * np.sum(exc_over_dm_cc_predict),
            AU2KCALMOL * np.sum(np.abs(exc_over_dm_cc_predict)),
            AU2KCALMOL
            * (
                e_cc
                - mdft.energy_tot(dm1_cc)
                - np.sum(correct_ene * rho_cc[0] * grids.weights)
            ),
            AU2KCALMOL
            * (
                e_cc
                - mdft.energy_tot(dm1_cc)
                - np.sum(exc_over_dm_cc_grids * rho_cc[0] * grids.weights)
            ),
        )
