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
    config_list,
    process_config,
)
from cc2cc.utils.rotate import rotate
from cc2cc.utils import (
    Grid,
    DATA_PATH,
)
from cc2cc.utils.b3lyp_reference import get_vv10_data
from cc2cc.utils.parser import gen_name_args, str2bool
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc
from cc2cc.utils.zmp_constraints import get_zmp_constraint_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the inversed potential and energy."
    )
    parser.add_argument(
        "--gen_config",
        type=str,
        default="gen_test.json",
        help="Path to JSON file defining train/eval splits.",
    )
    parser.add_argument(
        "--if_eval",
        type=str2bool,
        default=False,
        help="Whether to use the evaluation mode in generating the data. Default is False.",
    )
    parser.add_argument(
        "--mp_number",
        type=int,
        default=0,
        help="Number of the current training cycle. Default is 0.",
    )
    parser.add_argument(
        "--mp_total",
        type=int,
        default=3,
        help="Total number of training cycles. Default is 3.",
    )
    parser.add_argument(
        "--check_convergence",
        type=str2bool,
        default=True,
        help="Whether to check the convergence of the wave function. Default is True.",
    )
    parser.add_argument(
        "--zmp_feature_ov", type=str2bool, default=True,
        help="Stream feature/occupied-virtual rotation Jacobians to float64 sidecars.",
    )
    parser.add_argument(
        "--zmp_observable_response", type=str2bool, default=True,
        help="Generate compact dipole, second-moment and atom-projection response data.",
    )
    parser.add_argument(
        "--zmp_response_block_size", type=int, default=32,
        help="Number of grid centers per ZMP constraint generation block.",
    )
    parser.add_argument(
        "--zmp_response_gap_floor", type=float, default=1e-3,
        help="Positive response denominator floor in Hartree; nonpositive gaps are errors.",
    )
    args = add_args(parser)

    print_computer_info(args.device)

    gen_config = process_config(args.gen_config)
    train_str_list = config_list(gen_config, "train")
    eval_str_list = config_list(gen_config, "eval")
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
                    np.savez(DATA_PATH / f"{name}.traj", coords=save_coords)

                load_coords = np.load(DATA_PATH / f"{name}.traj.npz")["coords"]
                traj_mole_pool = []
                for frame_coords in load_coords:
                    molecule = np.array(copy.deepcopy(mol.atom), dtype=object)
                    for i, pos in enumerate(frame_coords):
                        molecule[i][1:] = pos.tolist()
                    molecule, _ = rotate(molecule, solve_symmetry=True, verbose=0)
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
                # print(f"SKIP: {name} already exists.")

                if args.zmp_feature_ov or args.zmp_observable_response:
                    cache_path = DATA_PATH / f"data_{name}.npz"
                    with np.load(cache_path, allow_pickle=True) as cached:
                        grids.coords = cached["coords"]
                        grids.weights = cached["weights"]
                        if len(grids.coords) != len(grids.weights):
                            raise ValueError("Cached coordinates/weights have different lengths.")
                        feature_shape = cached["rho_cube_zmp"].shape
                        grids.input_level = feature_shape[1]
                        if int(np.prod(feature_shape[2:])) != grids.cube_size:
                            raise ValueError("Cached cube layout differs from the current layout.")
                        baseline = pyscf.dft.RKS(mol) if mol.spin == 0 else pyscf.dft.UKS(mol)
                        baseline = baseline.density_fit()
                        baseline.xc = "b3lyp"
                        baseline.nlc = ""
                        constraint_data = get_zmp_constraint_data(
                            baseline, grids, cached["dm1_zmp"], cached["dm1_cc"],
                            DATA_PATH / f"data_{name}_zmp", args,
                        )
                    addon_path = DATA_PATH / f"data_{name}_addon.npz"
                    if addon_path.exists():
                        with np.load(addon_path, allow_pickle=True) as addon:
                            constraint_addon = dict(addon.items())
                    else:
                        constraint_addon = {}
                    constraint_addon.update(constraint_data)
                    np.savez(addon_path, **constraint_addon)
                    print(f"Updated ZMP constraints for {name}")
                    continue

                print(f"Add zmp VV10 data for {name}")
                cache_path = DATA_PATH / f"data_{name}.npz"
                with np.load(cache_path, allow_pickle=True) as cached:
                    required = ("dm1_dft", "e_dft", "dm1_cc", "weights")
                    missing = [key for key in required if key not in cached]
                    if missing:
                        print(
                            f"SKIP VV10 update for {name}: missing cached fields "
                            f"{', '.join(missing)}."
                        )
                    else:
                        cached_weights = cached["weights"]
                        if cached_weights.shape != grids.weights.shape:
                            raise RuntimeError(
                                f"Cached grid weights for {name} have shape "
                                f"{cached_weights.shape}, expected {grids.weights.shape}."
                            )
                        grids.weights = cached_weights
                        vv10_data = get_vv10_data(
                            mol,
                            grids,
                            cached["dm1_dft"],
                            cached["e_dft"],
                            cached["dm1_cc"],
                            args,
                        )
                        if "dm1_zmp" in cached and "e_zmp" in cached:
                            vv10_data.update(
                                get_vv10_data(
                                    mol,
                                    grids,
                                    cached["dm1_zmp"],
                                    cached["e_zmp"],
                                    cached["dm1_cc"],
                                    args,
                                    suffix="_zmp",
                                )
                            )
                        else:
                            print(
                                f"SKIP ZMP-based VV10 update for {name}: missing "
                                "cached dm1_zmp or e_zmp."
                            )

                        addon_path = DATA_PATH / f"data_{name}_addon.npz"
                        if addon_path.exists():
                            with np.load(addon_path, allow_pickle=True) as addon:
                                data_dict_addon = dict(addon.items())
                        else:
                            data_dict_addon = {}
                        data_dict_addon.pop("exc_post_grid", None)
                        data_dict_addon.pop("enlc_post", None)
                        data_dict_addon.update(vv10_data)
                        np.savez(addon_path, **data_dict_addon)
            else:
                if mol.spin == 0:
                    cc(mol, grids, name, args, evaluate=evaluate)
                else:
                    ucc(mol, grids, name, args, evaluate=evaluate)

        except (KeyError, ValueError, RuntimeError) as e:
            print(f"ERROR: {name_mol} {args.md_number}")
            print(e)
            error_molecule.append(name)
            print(f"Error molecule: {error_molecule}")
        finally:
            print(f"Processed: {name_mol} {args.md_number}")
            print("========================================\n\n")
        print()

    print(f"Error molecule: {error_molecule}")
