""" """

import argparse
import copy

import numpy as np

import pyscf
import pyscf.dft
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
from cc2cc.utils.parser import gen_name_args, str2bool
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc
from cc2cc.utils.b3lyp_reference import get_vv10_data


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
        "--check_post_vv10_gradient",
        action="store_true",
        help="Check analytical Post-DFT VV10 gradients against finite differences "
        "(six additional B3LYP calculations per atom).",
    )
    parser.add_argument(
        "--check_convergence",
        type=str2bool,
        default=True,
        help="Whether to check the convergence of the wave function. Default is True.",
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
                print(f"SKIP: {name} already exists.")
                # 1. Standard DFT (B3LYP)
                with np.load(
                    DATA_PATH / f"data_{name}.npz", allow_pickle=True
                ) as data_npz:
                    missing_keys = [
                        key
                        for key in ("dm1_dft", "e_dft", "dm1_cc", "weights")
                        if key not in data_npz
                    ]
                    if missing_keys:
                        print(
                            f"SKIP comparison: {name} missing cached fields: "
                            f"{', '.join(missing_keys)}",
                            flush=True,
                        )
                        continue
                    dm_dft = data_npz["dm1_dft"]
                    e_dft = float(data_npz["e_dft"])
                    dm_cc = data_npz["dm1_cc"]
                    weights = data_npz["weights"]
                    if weights.shape != grids.weights.shape:
                        raise ValueError(
                            f"Cached weights shape {weights.shape} does not match "
                            f"grid weights shape {grids.weights.shape} for {name}"
                        )
                    assert np.allclose(
                        grids.weights, weights, rtol=1e-10, atol=1e-12
                    ), f"Cached weights do not match grid weights for {name}"
                    grids.weights = weights

                vv10_data = get_vv10_data(
                    mol, grids, dm_dft, e_dft, dm_cc, args, evaluate
                )
                addon_path = DATA_PATH / f"data_{name}_addon.npz"
                if addon_path.exists():
                    with np.load(addon_path, allow_pickle=True) as data_addon:
                        data_dict_addon = dict(data_addon)
                else:
                    data_dict_addon = {}
                data_dict_addon.pop("exc_post_grid", None)
                data_dict_addon.pop("enlc_post", None)
                data_dict_addon.update(vv10_data)
                data_dict_addon["coords"] = grids.coords
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
