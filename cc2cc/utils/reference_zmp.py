"""ZMP inversion and projected analytic baseline gradients."""

import argparse
import json
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf

from cc2cc.utils.Grids import Grid
from cc2cc.utils.reference_b3lyp import converged_b3lyp
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.TestDataDFT import diff_rho, diff_I_value
from cc2cc.utils.reference_cc import make_ccsd, ccsd_t_target_density
from cc2cc.utils.zmp import RZMP, UZMP

LEVEL_SHIFT = 2
MAX_CYCLE = 200000
RHO_REL_TOL = 1e-1
I_VALUE_REL_TOL = 1e-3
DIIS_SPACE = 8


def get_zmp(
    mol,
    dm_tar,
    dm_dft,
    grids,
    max_l=4,
    verbose=0,
    start_l=0,
    max_cycle=MAX_CYCLE,
):
    """Invert a restricted/unrestricted density and print per-stage diagnostics.

    Stop a converged stage when both density errors meet the relative DFT
    tolerances, preserving the divergence stop and previous-density fallback.
    """
    if dm_tar is None:
        return None, None
    zmp_class = RZMP if np.ndim(dm_tar) == 2 else UZMP
    mzmp = zmp_class(mol, dm_tar, grids, dftxc=1, xc="b3lyp")
    mzmp.diis_space = DIIS_SPACE
    mzmp.verbose = verbose
    mzmp.max_cycle = max_cycle

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
    dft_difference_dipole = np.linalg.norm(dft_dipole - tar_dipole)
    best_difference = None
    previous_dm = None
    for exponent in range(start_l, max_l + 1):
        mzmp.level_shift = LEVEL_SHIFT**exponent
        print(f"Running ZMP with lambda_ZMP = {2**exponent}")
        mzmp.zscf(2**exponent)
        dm = mzmp.make_rdm1()
        zmp_difference = diff_rho(mol, dm, dm_tar, grids)
        zmp_i_value = diff_I_value(mol, dm, dm_tar, grids)
        zmp_dipole = mzmp.dip_moment(mol=mol, dm=dm, unit="A.U.")
        zmp_difference_dipole = np.linalg.norm(zmp_dipole - tar_dipole)
        print(
            f"ZMP, rho diff = {zmp_difference:.3e} electrons "
            f"I value = {zmp_i_value:.3e} (dimensionless) "
            f"dipole diff = {zmp_difference_dipole:.3e} A.U. ",
            flush=True,
        )
        print(
            f"DFT, rho diff = {dft_difference:.3e} electrons "
            f"I value = {dft_i_value:.3e} (dimensionless) "
            f"dipole diff = {dft_difference_dipole:.3e} A.U. ",
            flush=True,
        )
        if best_difference is None or zmp_difference < best_difference:
            best_difference = zmp_difference
        elif zmp_difference > best_difference * 10:
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
                f"Difference in {label} grids between CCSD (kcal/mol): "
                f"DFT {error_dft * AU2KCALMOL}, ZMP {error_zmp * AU2KCALMOL}"
            )
        if (
            mzmp.converged
            and zmp_difference <= RHO_REL_TOL * dft_difference
            and zmp_i_value <= I_VALUE_REL_TOL * dft_i_value
        ):
            print(
                f"ZMP density converged at lambda_ZMP = {2**exponent}: "
                f"rho diff <= {RHO_REL_TOL:g} * DFT rho diff and "
                f"I value <= {I_VALUE_REL_TOL:g} * DFT I value.",
                flush=True,
            )
            return mzmp, dm
        previous_dm = np.array(dm, copy=True)
    return mzmp, mzmp.make_rdm1()


def zmp_baseline_gradient(mf, dm_zmp):
    """Analytic baseline gradient at a supplied ZMP determinant, in Hartree/Bohr.

    Use the baseline occupied Fock block for the energy-weighted density,
    excluding response along rebuilt ZMP solutions. The learned correction
    must supply its own explicit, Pulay, and grid contributions.

    Pass the same SCF object used for the baseline energy so its integral
    approximation and native gradient agree, including DF auxiliary response.
    """
    overlap = mf.get_ovlp()
    density_energy, coeff = mf.eig(-overlap @ dm_zmp @ overlap, overlap)
    occ = mf.get_occ(density_energy, coeff)
    fock = mf.get_hcore() + mf.get_veff(mf.mol, dm_zmp)
    energy = np.zeros_like(occ)
    restricted = dm_zmp.ndim == 2
    spin_coeffs = (coeff,) if restricted else coeff
    spin_occs = (occ,) if restricted else occ
    spin_focks = (fock,) if restricted else fock
    spin_energies = (energy,) if restricted else energy
    for c, n, f, e in zip(spin_coeffs, spin_occs, spin_focks, spin_energies):
        occupied = n > 0
        c_occ = c[:, occupied]
        # Semicanonicalization gives W = P F P / occupation without inversion energies.
        e[occupied], rotation = np.linalg.eigh(c_occ.conj().T @ f @ c_occ)
        c[:, occupied] = c_occ @ rotation
    grad = mf.nuc_grad_method()
    grad.grid_response = True
    return grad.kernel(mo_energy=energy, mo_coeff=coeff, mo_occ=occ)


if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be run as a script.")
