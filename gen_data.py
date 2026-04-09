""" """

import argparse
import copy

import numpy as np

import pyscf
import pyscf.md

from cc2cc.utils import (
    gen_mole,
    print_computer_info,
    add_args,
)
from cc2cc.utils.rotate import rotate
from cc2cc.utils import Grid, DATA_PATH, AU2KCALMOL
from cc2cc.utils.parser import gen_name_args
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc

from cc2cc.utils.get_zmp import get_zmp_rks, get_zmp_uks

from cc2cc.utils.get_dft_energy_rks import get_dft_energy as get_dft_energy_rks
from cc2cc.utils.get_dft_energy_uks import get_dft_energy as get_dft_energy_uks
from cc2cc.utils.get_dft_grad_rks import get_dft_grad as get_dft_grad_rks
from cc2cc.utils.get_dft_grad_uks import get_dft_grad as get_dft_grad_uks

train_str_list = [
    # # #####################
    # # ########  0  ########
    # # #####################
    "molecule0-W4_11",
    "molecule0-ADDON",
    "AHB21-1A",
    "AHB21-4A",
    "ALK8-li+",
    "ALK8-na+",
    "ALKBDE10-ca",
    "ALKBDE10-k",
    "ALKBDE10-li",
    "ALKBDE10-mg",
    "ALKBDE10-na",
    "CHB6-24A",
    "DIPCS10-be_2+",
    "DIPCS10-mg_2+",
    "G21EA-EA_c-",
    "G21EA-EA_o-",
    "G21EA-EA_p-",
    "G21EA-EA_s-",
    "G21EA-EA_si-",
    "G21IP-al+",
    "G21IP-b+",
    "G21IP-be+",
    "G21IP-c+",
    "G21IP-cl+",
    "G21IP-f+",
    "G21IP-mg+",
    "G21IP-n+",
    "G21IP-o+",
    "G21IP-p+",
    "G21IP-s+",
    "G21IP-si+",
    "HEAVYSB11-br",
    "SIE4x4-he",
    "SIE4x4-he+",
    "RG18-ne",
    "RG18-ar",
    "RG18-kr",
    # # #####################
    # # ########  1  ########
    # # #####################
    "molecule1-W4_11",
    # # #####################
    # # ########  2  ########
    # # #####################
    # "molecule2-W4_11",
]

eval_str_list = [
    # # #####################
    # # ######  2(<9)  ######
    # # #####################
    # # "molecule2-W4_11",
    # # #####################
    # # ######  3(<9)  ######
    # # #####################
    # "molecule3-W4_11",
    # # #####################
    # # ######  4(<9)  ######
    # # #####################
    # "molecule4-W4_11",
    # # #####################
    # # ######  5(<9)  ######
    # # #####################
    # "molecule5-W4_11",
    # # ######## 0 H ########
    # "ALK8-li4_c",
    # "G2RC-62",
    # "G2RC-67",
    # "HAL59-29_CF3Br-benB",
    "HAL59-30_CF3I-benB",
    # "IL16-152B",
    # "IL16-214B",
    # "IL16-229B",
    # # ######## 1 H ########
    # "HAL59-BrBr_FCCH",
    "HAL59-FI_FCCH",
    # "PNICO23-22b",
    # # ######## 2 H ########
    # "DC13-o3_c2h2_add",
    # "RSE43-P5",
    # "RSE43-P7",
    # "TAUT15-7a",
    # "TAUT15-7b",
    # "YBDE18-nf3-ch2",
    # "YBDE18-pf3-ch2",
    # # ######## 3 H ########
    # "IL16-230B",
    # "RSE43-E5",
    # "RSE43-E7",
    # # #####################
    # # ######  6(<9)  ######
    # # #####################
    # # ######## 0 H ########
    # "RG18-ne6",
    # # ######## 1 H ########
    # "ALK8-li5_ch",
    # "IL16-144B",
    # "PArel-c2cl41",
    # "PArel-c2cl42",
    # "PArel-c2cl43",
    # # ######## 2 H ########
    # "RSE43-P28",
    # # ######## 3 H ########
    "HAL59-NH3_F3CI",
    # "PArel-c2h2f41",
    # "PArel-c2h2f42",
    # "RSE43-E28",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    args = add_args(parser)

    print_computer_info(args.device)

    train_str_list = gen_name_args(train_str_list, args.dataset, args.name_mol_reverse)
    eval_str_list = gen_name_args(eval_str_list, args.dataset, args.name_mol_reverse)

    if args.if_eval:
        if args.mp_total != 0:
            name_mol_list = eval_str_list[args.mp_number :: args.mp_total]
        else:
            name_mol_list = eval_str_list
        evaluate = True
    else:
        if args.mp_total != 0:
            name_mol_list = train_str_list[args.mp_number :: args.mp_total]
        else:
            name_mol_list = train_str_list
        evaluate = False

    error_molecule = []
    print(f"Name Molecule List: {name_mol_list}")

    for name_mol in name_mol_list:
        name = f"{name_mol}_{args.basis}"

        try:
            mol = gen_mole(
                name_mol,
                args.basis,
                dataset_name=args.dataset,
            )

            if mol is None:
                print(f"SKIP: {name_mol} due to missing molecule file.")
                continue

            if args.md_number != 0 and mol.natm != 1:
                if not (DATA_PATH / f"{name}.traj.npz").exists():
                    # sample the molecular CONFORMATIONS through MD
                    mol_md = mol.copy()
                    mol_md.basis = "def2-svp"
                    if mol_md.spin != 0:
                        myks = mol_md.UKS()
                        myks.xc = "b3lyp"
                    else:
                        myks = mol_md.RKS()
                        myks.xc = "b3lyp"

                    # initial velocities from a Maxwell-Boltzmann distribution [T in K and velocities are returned in (Bohr/ time a.u.)]
                    init_veloc = pyscf.md.distributions.MaxwellBoltzmannVelocity(
                        mol_md, T=2500
                    )

                    # We set the initial velocity by passing to "veloc",
                    # T is the ensemble temperature in K and taut is the Berendsen Thermostat time constant given in time a.u.
                    myintegrator = pyscf.md.integrators.NVTBerendson(
                        myks,
                        T=2500,
                        taut=50,
                        dt=0.5 / 0.024188843265857,
                        steps=1000,
                        veloc=init_veloc,
                        incore_anyway=True,
                        frames=[],
                    ).run()
                    save_frames_list = myintegrator.frames[::20]
                    save_coords = np.array([frame.coord for frame in save_frames_list])
                    np.savez_compressed(DATA_PATH / f"{name}.traj", coords=save_coords)

                load_coords = np.load(DATA_PATH / f"{name}.traj.npz")["coords"]
                traj_mole_pool = []
                for frame_coords in load_coords:
                    molecule = np.array(copy.deepcopy(mol.atom), dtype=object)
                    for i, pos in enumerate(frame_coords):
                        molecule[i][1:] = pos.tolist()
                    molecule, _ = rotate(
                        molecule,
                        solve_symmetry=True,
                        verbose=0,
                    )
                    traj_mole_pool.append(molecule.copy())
                    if len(traj_mole_pool) > args.md_number:
                        print(molecule)
                        break

                mol = pyscf.M(
                    atom=traj_mole_pool[args.md_number],
                    basis=mol.basis,
                    ecp=mol.ecp,
                    spin=mol.spin,
                    charge=mol.charge,
                    unit="B",
                )
                print(f"MD frame number: {args.md_number}", flush=True)
                print(f"Molecule atoms:\n{mol.atom}", flush=True)

            if args.md_number != 0:
                if mol.natm != 1:
                    name = f"{name}_{args.md_number}"
                else:
                    continue  # for single atom, no need to do md
            print(f"Processing: {name}", flush=True)

            grids = Grid(mol, args.grid_level, 7)

            if args.if_continue and (DATA_PATH / f"data_{name}.npz").exists():
                data_dict = dict(np.load(DATA_PATH / f"data_{name}.npz"))
                dm_tar = data_dict["dm1_cc"]
                dm_dft = data_dict["dm1_dft"]
                e_cc = data_dict["e_cc"]

                data_dict["tol_cc_grids"] = (
                    data_dict["exc_cc_grids"]
                    + data_dict["hatree_cc_grids"]
                    + data_dict["kin_cc_grids"]
                    + data_dict["nuc_cc_grids"]
                )

                data_dict["tol_dft_grids"] = (
                    data_dict["exc_dft_grids"]
                    + data_dict["exc_k_dft_grids"]
                    + data_dict["hatree_dft_grids"]
                    + data_dict["kin_dft_grids"]
                    + data_dict["nuc_dft_grids"]
                )

                print(f"mol.spin: {mol.spin}")
                max_l = 20

                if mol.spin == 0:
                    mzmp, dm1_zmp = get_zmp_rks(mol, dm_tar, dm_dft, grids, max_l)
                    data_dict["dm1_zmp"] = dm1_zmp
                    data_append_dict = get_dft_energy_rks(
                        mol,
                        grids,
                        mzmp,
                        dm1_zmp,
                        evaluate=evaluate,
                    )
                    for key in data_append_dict:
                        key_zmp = key.replace("dft", "zmp")
                        data_dict[key_zmp] = data_append_dict[key]
                    data_dict["tol_delta_zmp_grids"] = (
                        data_dict["tol_cc_grids"] - data_dict["tol_zmp_grids"]
                    )

                    data_append_dict = get_dft_grad_rks(mol, grids, dm1_zmp, data_dict)
                    for key in data_append_dict:
                        if "dft" in key:
                            key_zmp = key.replace("dft", "zmp")
                            data_dict[key_zmp] = data_append_dict[key]
                        else:
                            key_zmp = key + "_zmp"
                            data_dict[key_zmp] = data_append_dict[key]

                    grad_zmp = mzmp.Gradients()
                    grad_zmp = grad_zmp.kernel()
                    data_dict["grad_zmp"] = grad_zmp

                    e_zmp = mzmp.energy_tot(dm1_zmp)
                    energy_train = e_cc - e_zmp
                    error_zmp = (
                        np.sum(data_dict["tol_delta_zmp_grids"] * grids.weights)
                        - energy_train
                    )
                    print(f"Error ZMP: {AU2KCALMOL * error_zmp}")
                else:
                    mzmp, dm1_zmp = get_zmp_uks(mol, dm_tar, dm_dft, grids, max_l)
                    data_dict["dm1_zmp"] = dm1_zmp
                    data_append_dict = get_dft_energy_uks(
                        mol,
                        grids,
                        mzmp,
                        dm1_zmp,
                        evaluate=evaluate,
                    )
                    for key in data_append_dict:
                        key_zmp = key.replace("dft", "zmp")
                        data_dict[key_zmp] = data_append_dict[key]
                    data_dict["tol_delta_zmp_grids"] = (
                        data_dict["tol_cc_grids"] - data_dict["tol_zmp_grids"]
                    )

                    data_append_dict = get_dft_grad_uks(mol, grids, dm1_zmp, data_dict)
                    for key in data_append_dict:
                        if "dft" in key:
                            key_zmp = key.replace("dft", "zmp")
                            data_dict[key_zmp] = data_append_dict[key]
                        else:
                            key_zmp = key + "_zmp"
                            data_dict[key_zmp] = data_append_dict[key]

                    grad_zmp = mzmp.Gradients()
                    grad_zmp = grad_zmp.kernel()
                    data_dict["grad_zmp"] = grad_zmp

                    e_zmp = mzmp.energy_tot(dm1_zmp)
                    energy_train = e_cc - e_zmp
                    error_zmp = (
                        np.sum(data_dict["tol_delta_zmp_grids"] * grids.weights)
                        - energy_train
                    )
                    print(f"Error ZMP: {AU2KCALMOL * error_zmp}")
                np.savez_compressed(DATA_PATH / f"data_{name}", **data_dict)
            else:
                if mol.spin == 0:
                    cc(mol, grids, name, args, evaluate=evaluate)
                else:
                    ucc(mol, grids, name, args, evaluate=evaluate)
        except (ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {args.md_number}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {args.md_number}")
        print()

    print(f"Error molecule: {error_molecule}")
