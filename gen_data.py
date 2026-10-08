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
from cc2cc.utils.nlc import post_dft_vv10_gradient


def _converged_b3lyp(mol, dm0=None):
    mf = (
        pyscf.dft.RKS(mol, xc="b3lyp")
        if mol.spin == 0
        else pyscf.dft.UKS(mol, xc="b3lyp")
    )
    mf.verbose = 0
    mf.max_cycle = 200
    mf.nlc = ""
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.conv_tol_cpscf = 1e-10
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError("B3LYP did not converge for Post-DFT VV10.")
    return mf


def _post_vv10_total_energy(mol, grid_level):
    mf = _converged_b3lyp(mol)
    dm = mf.make_rdm1()
    dm_tot = dm if mol.spin == 0 else dm[0] + dm[1]
    grids = Grid(mol, grid_level, 7)
    enlc = np.dot(
        eval_nlc_exc_density(mol, grids, dm_tot, "vv10"),
        np.asarray(grids.weights),
    )
    return float(mf.e_tot) + enlc


def _post_vv10_finite_difference_gradient(mol, grid_level, step=1e-3):
    """Central-difference reference gradient in Hartree/Bohr; step in Bohr."""
    coords = mol.atom_coords()
    gradient = np.empty_like(coords)
    for atom in range(mol.natm):
        for axis in range(3):
            displaced_energies = []
            for sign in (1, -1):
                displaced_mol = mol.copy()
                displaced_coords = coords.copy()
                displaced_coords[atom, axis] += sign * step
                displaced_mol.set_geom_(displaced_coords, unit="Bohr")
                displaced_energies.append(
                    _post_vv10_total_energy(displaced_mol, grid_level)
                )
            gradient[atom, axis] = (displaced_energies[0] - displaced_energies[1]) / (
                2 * step
            )
    return gradient


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
                    grids.weights = weights

                # 2. Post-DFT VV10 (using standard DFT 1-RDM)
                nlcgrids = grids
                dm_dft_tot = dm_dft if mol.spin == 0 else (dm_dft[0] + dm_dft[1])
                exc_post_vv10_grid = eval_nlc_exc_density(
                    mol, nlcgrids, dm_dft_tot, "vv10"
                )
                e_post_vv10 = np.dot(exc_post_vv10_grid, nlcgrids.weights)
                _, e_post_vv10_check, _ = pyscf.dft.numint.NumInt().nr_nlc_vxc(
                    mol, nlcgrids, "vv10", dm_dft_tot
                )
                print(
                    "NLC energy check (kcal/mol): difference = "
                    f"{(e_post_vv10 - e_post_vv10_check) * AU2KCALMOL:.12e}"
                )
                if not np.isclose(
                    e_post_vv10, e_post_vv10_check, rtol=1e-10, atol=1e-12
                ):
                    raise RuntimeError(
                        "NLC grid integral disagrees with nr_nlc_vxc "
                        f"for {name}: {e_post_vv10} vs {e_post_vv10_check} Hartree"
                    )
                e_post_vv10 = e_dft + e_post_vv10
                if mol.natm == 1:
                    grad_post_vv10 = np.zeros((mol.natm, 3))
                else:
                    mf_dft = _converged_b3lyp(mol, dm0=dm_dft)
                    grad_post_vv10 = post_dft_vv10_gradient(mf_dft, nlcgrids)
                    grad_post_vv10_fd = _post_vv10_finite_difference_gradient(
                        mol, args.grid_level
                    )
                    grad_error = np.max(np.abs(grad_post_vv10 - grad_post_vv10_fd))
                    print(
                        "Post-DFT VV10 analytical/FD gradient max difference "
                        f"(Hartree/Bohr): {grad_error:.12e}"
                    )

                addon_path = DATA_PATH / f"data_{name}_addon.npz"
                if addon_path.exists():
                    with np.load(addon_path, allow_pickle=True) as data_addon:
                        data_dict_addon = dict(data_addon)
                else:
                    data_dict_addon = {}
                data_dict_addon.pop("exc_post_grid", None)
                data_dict_addon.pop("enlc_post", None)
                data_dict_addon["exc_post_vv10_grid"] = exc_post_vv10_grid
                data_dict_addon["e_post_vv10"] = e_post_vv10
                data_dict_addon["grad_post_vv10"] = grad_post_vv10
                np.savez(addon_path, **data_dict_addon)
                print("Post-DFT VV10 gradient (Hartree/Bohr):\n" f"{grad_post_vv10}")

                # 3. SCF-VV10 (VV10 in the SCF loop)
                mf_scf = (
                    pyscf.dft.RKS(mol, xc="b3lyp")
                    if mol.spin == 0
                    else pyscf.dft.UKS(mol, xc="b3lyp")
                )
                mf_scf.nlc = "vv10"
                mf_scf.verbose = 4
                mf_scf.kernel()
                if args.check_convergence and not mf_scf.converged:
                    raise RuntimeError("SCF-VV10 did not converge.")
                dm_scf_vv10 = mf_scf.make_rdm1()
                e_scf_vv10 = float(mf_scf.e_tot)

                # 4. Electronic density difference
                drho = diff_rho(mol, dm_dft, dm_scf_vv10, grids)
                drho_dft_cc = diff_rho(mol, dm_dft, dm_cc, grids)
                drho_scf_vv10_cc = diff_rho(mol, dm_scf_vv10, dm_cc, grids)

                print(f"\n=== Energy and density comparison: {name} ===")
                print("Energy relative to B3LYP total (kcal/mol):")
                print("  B3LYP baseline:                      0.0")
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

            # else:
            #     if mol.spin == 0:
            #         cc(mol, grids, name, args, evaluate=evaluate)
            #     else:
            #         ucc(mol, grids, name, args, evaluate=evaluate)

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
