"""Shared B3LYP calculations and finite-difference references."""

from __future__ import annotations

import numpy as np

import pyscf.dft


def converged_b3lyp(mol, dm0=None, *, nlc="", grids=None, verbose=0, use_gpu=False):
    mf = (
        pyscf.dft.RKS(mol, xc="b3lyp")
        if mol.spin == 0
        else pyscf.dft.UKS(mol, xc="b3lyp")
    )
    mf = mf.density_fit()
    if grids is not None:
        mf.grids = grids
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.nlc = nlc
    if use_gpu:
        mf = mf.to_gpu()
    mf.kernel(dm0=dm0)
    return mf


def b3lyp_finite_difference_gradient(mol, step=1e-3):
    """Central-difference B3LYP gradient in Hartree/Bohr; step in Bohr."""
    return finite_difference_gradient(
        mol, lambda displaced: converged_b3lyp(displaced).e_tot, step
    )


def finite_difference_gradient(mol, total_energy, step):
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
                displaced_energies.append(total_energy(displaced_mol))
            gradient[atom, axis] = (displaced_energies[0] - displaced_energies[1]) / (
                2 * step
            )
    return gradient
