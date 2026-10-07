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
    diff_rho,
    AU2KCALMOL,
    eval_nlc_exc_density,
)
from cc2cc.utils.parser import gen_name_args, str2bool
from cc2cc.gen_cc import cc
from cc2cc.gen_ucc import ucc

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
                        for key in ("dm1_dft", "e_dft", "dm1_cc")
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

                # 2. Post-DFT VV10 (using standard DFT 1-RDM)
                nlcgrids = grids
                dm_dft_tot = dm_dft if mol.spin == 0 else (dm_dft[0] + dm_dft[1])
                exc_post_grid = eval_nlc_exc_density(
                    mol, nlcgrids, dm_dft_tot, "vv10"
                )
                enlc_post = np.dot(exc_post_grid, nlcgrids.weights)
                _, enlc_post_check, _ = pyscf.dft.numint.NumInt().nr_nlc_vxc(
                    mol, nlcgrids, "vv10", dm_dft_tot
                )
                print(
                    "NLC energy check (kcal/mol): difference = "
                    f"{(enlc_post - enlc_post_check) * AU2KCALMOL:.12e}"
                )
                if not np.isclose(enlc_post, enlc_post_check, rtol=1e-10, atol=1e-12):
                    raise RuntimeError(
                        "NLC grid integral disagrees with nr_nlc_vxc "
                        f"for {name}: {enlc_post} vs {enlc_post_check} Hartree"
                    )
                e_post_vv10 = e_dft + enlc_post

                # 3. SCF-VV10 (VV10 in the SCF loop)
                mf_scf = (
                    pyscf.dft.RKS(mol, xc="b3lyp")
                    if mol.spin == 0
                    else pyscf.dft.UKS(mol, xc="b3lyp")
                )
                mf_scf.nlc = "vv10"
                mf_scf.verbose = 4
                mf_scf.kernel()
                dm_scf_vv10 = mf_scf.make_rdm1()
                e_scf_vv10 = float(mf_scf.e_tot)

                # 4. Electronic density difference
                drho = diff_rho(mol, dm_dft, dm_scf_vv10, grids)
                drho_dft_cc = diff_rho(mol, dm_dft, dm_cc, grids)
                drho_scf_vv10_cc = diff_rho(mol, dm_scf_vv10, dm_cc, grids)

                print(f"\n=== Energy and density comparison: {name} ===")
                print("Energy relative to B3LYP total (kcal/mol):")
                print("  B3LYP baseline:                      0.00000000")
                print(
                    "  Post-DFT VV10:                      "
                    f"{(e_post_vv10 - e_dft) * AU2KCALMOL:>16.8f}"
                )
                print(
                    "  Self-consistent SCF-VV10:            "
                    f"{(e_scf_vv10 - e_dft) * AU2KCALMOL:>16.8f}"
                )
                print(
                    "  SCF-VV10 minus post-DFT:             "
                    f"{(e_scf_vv10 - e_post_vv10) * AU2KCALMOL:>16.8f}"
                )
                print(
                    f"Integrated absolute density difference (electrons): {drho:.10e}"
                )
                print(
                    "Integrated absolute density difference "
                    f"(B3LYP/post-DFT VV10 vs CCSD, electrons): {drho_dft_cc:.10e}"
                )
                print(
                    "Integrated absolute density difference "
                    f"(SCF-VV10 vs CCSD, electrons): {drho_scf_vv10_cc:.10e}\n"
                )

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
