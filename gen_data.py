""" """

import argparse
import copy

import numpy as np

import pyscf
import pyscf.md

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils.rotate import rotate
from cc2cc.utils import Grid, DATA_PATH
from cc2cc.utils.parser import gen_name_args
from cc2cc.gen_cc import cc, get_dft_grad as get_rks_grad
from cc2cc.gen_ucc import ucc, get_dft_grad as get_uks_grad
from cc2cc.utils.env_var import EDGE_SIZE


train_str_list = [
    # # #####################
    # # ########  0  ########
    # # #####################
    # "molecule0-W4_11",
    # "AHB21-1A",
    # "AHB21-4A",
    # "ALK8-li+",
    # "W4_11-ch4",
    # "ALK8-na+",
    # "ALKBDE10-ca",
    # "ALKBDE10-k",
    # "ALKBDE10-li",
    # "ALKBDE10-mg",
    # "ALKBDE10-na",
    # "CHB6-24A",
    # "DIPCS10-be_2+",
    # "DIPCS10-mg_2+",
    # "G21EA-EA_c-",
    # "G21EA-EA_o-",
    # "G21EA-EA_p-",
    # "G21EA-EA_s-",
    # "G21EA-EA_si-",
    # "G21IP-al+",
    # "G21IP-b+",
    # "G21IP-be+",
    # "G21IP-c+",
    # "G21IP-cl+",
    # "G21IP-f+",
    # "G21IP-mg+",
    # "G21IP-n+",
    # "G21IP-o+",
    # "G21IP-p+",
    # "G21IP-s+",
    # "G21IP-si+",
    # "HEAVYSB11-br",
    # "RG18-ar",
    # "RG18-kr",
    # "RG18-ne",
    # "SIE4x4-he",
    # "SIE4x4-he+",
    # # #####################
    # # ########  1  ########
    # # #####################
    # "molecule1-W4_11",
    # # #####################
    # # ########  2  ########
    # # #####################
    # "molecule2-W4_11",
    "W4_11-o2",
    # "W4_11-ch2c",
    # "W4_11-ch2ch",
    # "W4_11-ch2nh2",
    # "W4_11-ch3nh",
    # "W4_11-clo",
    # "W4_11-hcnh",
    # "W4_11-hoo",
    # "W4_11-s2",
    # "W4_11-so",
    # "W4_11-ssh",
    # "ADDON_Se",
    # "ADDON_Ge",
    # "ADDON_As",
    # "ADDON_Te",
    # "ADDON_I",
    # "ADDON_Bi",
    # "ADDON_Pb",
    # "ADDON_Sb",
]

eval_str_list = [
    # #####################
    # ######  2(<9)  ######
    # #####################
    # "molecule2-W4_11",
    # #####################
    # ######  3(<9)  ######
    # #####################
    "molecule3-W4_11",
    # #####################
    # ######  4(<9)  ######
    # #####################
    "molecule4-W4_11",
    # #####################
    # ######  5(<9)  ######
    # #####################
    "molecule5-W4_11",
    # ######## 0 H ########
    "ALK8-li4_c",
    "G2RC-62",
    "G2RC-67",
    "HAL59-29_CF3Br-benB",
    # "HAL59-30_CF3I-benB",
    "IL16-152B",
    "IL16-214B",
    "IL16-229B",
    # ######## 1 H ########
    "HAL59-BrBr_FCCH",
    # "HAL59-FI_FCCH",
    "PNICO23-22b",
    # ######## 2 H ########
    "DC13-o3_c2h2_add",
    "RSE43-P5",
    "RSE43-P7",
    "TAUT15-7a",
    "TAUT15-7b",
    "YBDE18-nf3-ch2",
    "YBDE18-pf3-ch2",
    # ######## 3 H ########
    "IL16-230B",
    "RSE43-E5",
    "RSE43-E7",
    # #####################
    # ######  6(<9)  ######
    # #####################
    # ######## 0 H ########
    "RG18-ne6",
    # ######## 1 H ########
    "ALK8-li5_ch",
    "IL16-144B",
    "PArel-c2cl41",
    "PArel-c2cl42",
    "PArel-c2cl43",
    # ######## 2 H ########
    "RSE43-P28",
    # ######## 3 H ########
    # "HAL59-NH3_F3CI",
    "PArel-c2h2f41",
    "PArel-c2h2f42",
    "RSE43-E28",
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

        # try:
        if 1:
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
            print(f"Processing: {name}")

            if args.if_continue:
                if (DATA_PATH / f"data_{name}.npz").exists():
                    continue

            grids = Grid(mol, args.grid_level, 7)
            if mol.spin == 0:
                cc(mol, grids, name, args, evaluate=evaluate)
            else:
                ucc(mol, grids, name, args, evaluate=evaluate)
        # except (ValueError, RuntimeError) as e:
        #     print(f"ERROR: {name_mol} {args.md_number}")
        #     print(e)
        #     error_molecule.append(name)
        #     print(f"Error molecule: {error_molecule}")
        # finally:
        #     print(f"Processed: {name_mol} {args.md_number}")
        # print()

    print(f"Error molecule: {error_molecule}")
