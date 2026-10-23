"""Full-pipeline nuclear finite differences of B3LYP energy on ZMP orbitals."""

import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf

from cc2cc.utils.Grids import Grid
from cc2cc.utils.b3lyp_reference import converged_b3lyp
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.TestDataDFT import diff_rho, diff_I_value
from cc2cc.utils.ccsd_t_target import make_ccsd, ccsd_t_target_density
from cc2cc.utils.zmp import RZMP, UZMP

LEVEL_SHIFT = 3
MAX_CYCLE = 20000


def get_zmp(
    mol, dm_tar, dm_dft, grids, max_l=4, verbose=0, start_l=0,
    max_cycle=MAX_CYCLE, *, strict=False,
):
    """Invert a restricted/unrestricted density and print per-stage diagnostics.

    Strict references require every stage to converge and use a fixed schedule.
    Otherwise preserve the data-generation divergence stop and previous density.
    """
    if dm_tar is None:
        return None, None
    zmp_class = RZMP if np.ndim(dm_tar) == 2 else UZMP
    mzmp = zmp_class(mol, dm_tar, grids, dftxc=1, xc="b3lyp")
    mzmp.diis_space = 64
    mzmp.verbose = verbose
    mzmp.max_cycle = max_cycle
    if strict:
        mzmp.conv_tol_dm = 1e-10
        mzmp.conv_tol_diis = 1e-7

    ao = mzmp.ao
    print("Start hatree part...")
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    nuc_grids = np.zeros_like(grids.weights)
    for charge, coord in zip(mol.atom_charges(), mol.atom_coords()):
        distance = np.linalg.norm(grids.coords - coord, axis=1)
        nonzero = distance > 1e-8
        nuc_grids[nonzero] -= charge / distance[nonzero]

    def energy_grids(dm):
        dm_tot = dm if np.ndim(dm) == 2 else np.sum(dm, axis=0)
        rho = mzmp._numint.eval_rho(mol, ao, dm_tot, xctype="LDA")
        hartree = 0.5 * np.einsum("pij,ij->p", int1e_grids, dm_tot) * rho
        return hartree, rho * nuc_grids

    hartree_tar, nuc_tar = energy_grids(dm_tar)
    hartree_dft, nuc_dft = energy_grids(dm_dft)
    tar_dipole = mzmp.dip_moment(mol=mol, dm=dm_tar, unit="A.U.")
    dft_dipole = mzmp.dip_moment(mol=mol, dm=dm_dft, unit="A.U.")
    dft_difference = diff_rho(mol, dm_dft, dm_tar, grids)
    dft_i_value = diff_I_value(mol, dm_dft, dm_tar, grids)
    best_difference = None
    previous_dm = None
    for exponent in range(start_l, max_l + 1):
        mzmp.level_shift = LEVEL_SHIFT**exponent
        print(f"Running ZMP with lambda_ZMP = {2**exponent}")
        mzmp.zscf(2**exponent)
        if strict and not mzmp.converged:
            raise RuntimeError(
                f"Reference ZMP did not converge at lambda={2**exponent}."
            )
        dm = mzmp.make_rdm1()
        difference = diff_rho(mol, dm, dm_tar, grids)
        dipole = mzmp.dip_moment(mol=mol, dm=dm, unit="A.U.")
        print(
            f"ZMP, rho diff = {difference:.3e} "
            f"I value = {diff_I_value(mol, dm, dm_tar, grids):.3e} "
            f"dipole diff = {np.linalg.norm(dipole - tar_dipole):.3e} ",
            flush=True,
        )
        print(
            f"DFT, rho diff = {dft_difference:.3e} "
            f"I value = {dft_i_value:.3e} "
            f"dipole diff = {np.linalg.norm(dft_dipole - tar_dipole):.3e} ",
            flush=True,
        )
        if best_difference is None or difference < best_difference:
            best_difference = difference
        elif not strict and difference > best_difference * 10:
            print("Warning: rho difference increased significantly, may be diverged.")
            print("Warning: ZMP did not converge within the specified lambda range.")
            return mzmp, previous_dm
        hartree, nuclear = energy_grids(dm)
        for label, target, dft_values, zmp_values in (
            ("Hartree", hartree_tar, hartree_dft, hartree),
            ("nuclear", nuc_tar, nuc_dft, nuclear),
        ):
            error_dft = np.dot(np.abs(target - dft_values), grids.weights)
            error_zmp = np.dot(np.abs(target - zmp_values), grids.weights)
            print(
                f"Difference in {label} grids between CCSD: "
                f"DFT {error_dft * AU2KCALMOL}, ZMP {error_zmp * AU2KCALMOL}"
            )
        previous_dm = np.array(dm, copy=True)
    return mzmp, mzmp.make_rdm1()


def zmp_reference_energy(mol, grid_level=3, max_l=20):
    """Return (ZMP-density B3LYP, self-consistent B3LYP) energies in Hartree.

    Reproduce the CCSD(T) target-density construction in gen_cc.cc/gen_ucc.ucc.
    Every inversion stage must converge; no previous-density fallback is used.
    """
    if mol.nelectron < 2:
        raise ValueError("The CCSD(T) ZMP reference requires at least two electrons.")
    if max_l < 0:
        raise ValueError("max_l must be nonnegative.")
    mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-8
    mf.max_cycle = 2500
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Reference HF did not converge.")
    mdft = converged_b3lyp(mol, mf.make_rdm1())
    if not mdft.converged:
        raise RuntimeError("Reference B3LYP did not converge.")
    mycc = make_ccsd(mf, unrestricted=mol.spin != 0, reference=True)
    mycc.kernel()
    eris = mycc.ao2mo()
    dm_tar = ccsd_t_target_density(
        mycc, eris, unrestricted=mol.spin != 0, reference=True
    )[-1]
    grids = Grid(mol, grid_level, 7)
    mzmp, dm_zmp = get_zmp(
        mol, dm_tar, mdft.make_rdm1(), grids, max_l=max_l, strict=True
    )
    energy = float(mzmp.energy_tot(dm_zmp))
    if not np.isfinite(energy) or not np.isfinite(mdft.e_tot):
        raise RuntimeError("Reference calculation produced a non-finite energy.")
    return energy, float(mdft.e_tot)


def zmp_finite_difference(mol, steps=(0.002, 0.001), grid_level=3, max_l=20):
    """Rebuild the complete pipeline at +/- each step (Bohr), for every atom.

    Gradients/forces have shape (nsteps, natm, 3), in Hartree/Bohr.
    Displaced energies have shape (nsteps, natm, 3, 2); sign order is (-, +).
    """
    steps = np.asarray(steps, dtype=np.float64)
    if (
        steps.ndim != 1
        or steps.size == 0
        or not np.all(np.isfinite(steps) & (steps > 0))
    ):
        raise ValueError(
            "steps must be a nonempty sequence of positive finite Bohr steps."
        )
    coords = mol.atom_coords(unit="Bohr")
    energies = np.empty((len(steps), mol.natm, 3, 2, 2), dtype=np.float64)
    for index, step in enumerate(steps):
        for atom in range(mol.natm):
            for axis in range(3):
                for side, sign in enumerate((-1, 1)):
                    displaced = coords.copy()
                    displaced[atom, axis] += sign * step
                    geometry = mol.set_geom_(displaced, unit="Bohr", inplace=False)
                    energies[index, atom, axis, side] = zmp_reference_energy(
                        geometry, grid_level=grid_level, max_l=max_l
                    )
    gradients = (energies[..., 1, :] - energies[..., 0, :]) / (
        2 * steps[:, None, None, None]
    )
    return {
        "steps_bohr": steps.tolist(),
        "coordinates_bohr": coords.tolist(),
        "elements": [mol.atom_symbol(atom) for atom in range(mol.natm)],
        "basis": mol.basis,
        "charge": mol.charge,
        "spin": mol.spin,
        "grid_level": grid_level,
        "max_l": max_l,
        "energy_units": "Hartree",
        "gradient_units": "Hartree/Bohr",
        "energy_sign_order": ["minus", "plus"],
        "zmp_energies": energies[..., 0].tolist(),
        "b3lyp_energies": energies[..., 1].tolist(),
        "grad_zmp": gradients[..., 0].tolist(),
        "force_zmp": (-gradients[..., 0]).tolist(),
        "grad_b3lyp": gradients[..., 1].tolist(),
        "force_b3lyp": (-gradients[..., 1]).tolist(),
        "grad_zmp_minus_b3lyp": (gradients[..., 0] - gradients[..., 1]).tolist(),
    }


def zmp_finite_difference_gradient(mol, grid_level, step=0.001, max_l=20):
    """Return the full-pipeline nuclear gradient (natm, 3), in Hartree/Bohr.

    Each sign/displacement rebuilds the CCSD(T) target and the ZMP inversion.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite, in Bohr.")
    if mol.natm == 1:
        return np.zeros((1, 3), dtype=np.float64)
    reference = zmp_finite_difference(
        mol, steps=(step,), grid_level=grid_level, max_l=max_l
    )
    return np.asarray(reference["grad_zmp"][0], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom", required=True, help="PySCF geometry string or XYZ path")
    parser.add_argument("--basis", required=True)
    parser.add_argument("--unit", choices=("Angstrom", "Bohr"), default="Angstrom")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument(
        "--spin", type=int, default=0, help="Number of alpha minus beta electrons"
    )
    parser.add_argument("--grid-level", type=int, default=3)
    parser.add_argument("--max-l", type=int, default=20)
    parser.add_argument("--steps", type=float, nargs="+", default=[0.002, 0.001])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib.num_threads(args.threads)
    mol = gto.M(
        atom=args.atom,
        basis=args.basis,
        unit=args.unit,
        charge=args.charge,
        spin=args.spin,
        symmetry=False,
        verbose=0,
    )
    result = zmp_finite_difference(mol, args.steps, args.grid_level, args.max_l)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved finite-difference reference to {args.output}")


if __name__ == "__main__":
    main()
