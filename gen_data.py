""" """

import argparse
import copy
from itertools import product

import numpy as np

import pyscf
import pyscf.md

from cc2cc.utils import gen_mole, print_computer_info, add_args
from cc2cc.utils.rotate import rotate
from cc2cc.utils import Grid, DATA_PATH
from cc2cc.utils.parser import gen_name_args
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc


train_str_list = [
    # #####################
    # ########  0  ########
    # #####################
    "molecule0-W4_11",
    # #####################
    # ########  1  ########
    # #####################
    # ######## 0 H ########
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
    "RG18-ar",
    "RG18-kr",
    "RG18-ne",
    "SIE4x4-he",
    "SIE4x4-he+",
    "molecule1-W4_11",
    # ######## 1 H ########
    "AHB21-1B",
    "AHB21-10A",
    "AHB21-15A",
    "ALK8-li_h",
    "G21EA-EA_11",
    "G21EA-EA_14",
    "G21EA-EA_17",
    "G21EA-EA_17n",
    "G21EA-EA_8",
    "G21IP-IP_61",
    "G21IP-IP_63",
    "G21IP-IP_65",
    "G21IP-IP_68",
    "G21IP-IP_70",
    "HEAVY28-hbr",
    "HEAVY28-hi",
    "HEAVYSB11-seh",
    "MB16_43-NaH",
    "SIE4x4-h2+_1.0",
    "SIE4x4-h2+_1.25",
    "SIE4x4-h2+_1.5",
    "SIE4x4-h2+_1.75",
    # ######## 2 H ########
    "BH76-PH2",
    "BH76-RKT01",
    "BH76-RKT13",
    "BH76-RKT06",
    "BH76-RKT10",
    "BH76-hfhts",
    "BH76-RKT14",
    "CARBHB12-3CL_B",
    "DIPCS10-h2s_2+",
    "G21EA-EA_12",
    "G21EA-EA_15",
    "G21EA-EA_18",
    "G21EA-EA_9",
    "G21IP-IP_62",
    "G21IP-IP_66",
    "HEAVY28-teh2",
    "MB16_43-BeH2",
    "MB16_43-MgH2",
    "PA26-h2p",
    "PA26-hclp",
    # #####################
    # ########  2  ########
    # #####################
    "W4_11-alcl",
    "W4_11-alf",
    "W4_11-b2",
    "W4_11-be2",
    "W4_11-bf",
    "W4_11-bn",
    "W4_11-bn3pi",
    "W4_11-c2",
    "W4_11-cf",
    "W4_11-cl2",
    "W4_11-clf",
    "W4_11-clo",
    "W4_11-cn",
    "W4_11-co",
    "W4_11-cs",
    "W4_11-f2",
    "W4_11-n2",
    "W4_11-no",
    "W4_11-o2",
    "W4_11-of",
    "W4_11-p2",
    "W4_11-s2",
    "W4_11-sif",
    "W4_11-sio",
    "W4_11-so",
    "W4_11-cch",
    "W4_11-hcn",
    "W4_11-hnc",
    "W4_11-hco",
    "W4_11-hno",
    "W4_11-hocl",
    "W4_11-hof",
    "W4_11-hoo",
    "W4_11-n2h",
    "W4_11-ssh",
    "W4_11-c-hcoh",
    "W4_11-h2co",
    "W4_11-t-hcoh",
    "W4_11-c-n2h2",
    "W4_11-t-n2h2",
    "W4_11-c2h2",
    "W4_11-ch2c",
    "W4_11-h2cn",
    "W4_11-hcnh",
    "W4_11-hooh",
    "W4_11-nh2cl",
    "W4_11-ch2ch",
    "W4_11-ch2nh",
    "W4_11-ch3f",
    "W4_11-sih3f",
    "W4_11-c2h4",
    "W4_11-ch2nh2",
    "W4_11-ch3nh",
    "W4_11-methanol",
    "W4_11-n2h4",
    "W4_11-ch3nh2",
    "W4_11-b2h6",
    "W4_11-c2h6",
    "W4_11-si2h6",
    # # ######## 0 H ########
    "ALK8-li2",
    "ALK8-na2",
    "ALKBDE10-bef",
    "ALKBDE10-beo",
    "ALKBDE10-cao",
    "ALKBDE10-kf",
    "ALKBDE10-lif",
    "ALKBDE10-lio",
    "ALKBDE10-mgo",
    "ALKBDE10-mgs",
    "ALKBDE10-nao",
    "G21EA-EA_20",
    "G21EA-EA_21",
    "G21EA-EA_22",
    "G21EA-EA_23",
    "G21EA-EA_23n",
    "G21EA-EA_24",
    "G21EA-EA_25",
    "G21IP-IP_73",
    "G21IP-IP_74",
    "G21IP-IP_75",
    "G21IP-IP_76",
    "G21IP-IP_77",
    "G21IP-IP_78",
    "G21IP-IP_79",
    # "G21IP-IP_80",
    "HAL59-BrBr",
    "HAL59-FBr",
    "HAL59-FI",
    "RG18-ar2",
    "RG18-kr2",
    "RG18-ne2",
    "SIE4x4-he2+_1.0",
    "SIE4x4-he2+_1.25",
    "SIE4x4-he2+_1.5",
    "SIE4x4-he2+_1.75",
]

eval_str_list = [
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
        name_mol_list, evaluate = eval_str_list[args.training_cycle :: 3], True
    else:
        name_mol_list, evaluate = train_str_list[args.training_cycle :: 3], False

    error_molecule = []
    print(f"Name Molecule List: {name_mol_list}")

    for name_mol in name_mol_list:
        name = f"{name_mol}_{args.basis}"

        try:
            mol = gen_mole(
                name_mol,
                args.basis,
                ma_basis=False,
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
                    # if mol.charge >= 0:
                    #     print(f"SKIP: {name_mol} {extend_atom} {extend_xyz} {distance}")
                    #     continue

            grids = Grid(mol, args.grid_level)
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
