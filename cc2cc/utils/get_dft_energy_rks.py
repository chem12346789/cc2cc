from itertools import product

import numpy as np

import pyscf
import pyscf.scf

from cc2cc.utils.tensor_helper import parallel_einsum


def _nuclear_energy_density(mol, grids, density):
    """Return the point-nucleus attraction energy-density grid."""
    nuclear = np.zeros_like(grids.weights)
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    for i, coord in enumerate(grids.coords):
        for atom_coord, charge in zip(atom_coords, atom_charges):
            distance = np.linalg.norm(atom_coord - coord)
            if distance > 1e-8:
                nuclear[i] -= density[i] * charge / distance
    return nuclear


def _ecp_energy_density(mol, ao, dm):
    """Return the pseudopotential correction in the AO-projected gauge."""
    if not mol.has_ecp():
        return np.zeros(ao.shape[0], dtype=ao.dtype)

    ecp = pyscf.scf.hf.get_hcore(mol)
    ecp -= mol.intor_symmetric("int1e_kin")
    ecp -= mol.intor_symmetric("int1e_nuc")
    overlap = mol.intor_symmetric("int1e_ovlp")
    projected_dm = np.linalg.solve(overlap, ecp @ dm)
    return parallel_einsum("pi,ij,pj->p", ao, projected_dm, ao)


def _orbital_gradient_squared(ao_gradient, orbital_coeff):
    orbital_gradient = parallel_einsum(
        "xpm,m->xp",
        ao_gradient,
        orbital_coeff,
    )
    return parallel_einsum(
        "xp,xp->p",
        orbital_gradient,
        orbital_gradient,
    )


def block_loop_rdm2(nao):
    """
    Generate slices for block processing of the 4-index 2-RDM.
    50 slices per dimension.
    """
    n_slices = 125
    n_batchs = nao // n_slices + 1
    total_batches = n_batchs**4
    print(
        f"Total batches for rdm2: {total_batches}. n_batchs: {n_batchs}, n_slices: {n_slices}, will consume about {n_slices**4 * 8 / 1024**3:.2f} GB memory."
    )
    for i_batch, j_batch, k_batch, l_batch in product(
        range(n_batchs),
        range(n_batchs),
        range(n_batchs),
        range(n_batchs),
    ):
        print(
            f"Processing batch {i_batch * n_batchs**3 + j_batch * n_batchs**2 + k_batch * n_batchs + l_batch + 1} / {total_batches}",
            flush=True,
        )
        nao_slice_i = n_slices if i_batch != n_batchs - 1 else nao - n_slices * i_batch
        nao_slice_j = n_slices if j_batch != n_batchs - 1 else nao - n_slices * j_batch
        nao_slice_k = n_slices if k_batch != n_batchs - 1 else nao - n_slices * k_batch
        nao_slice_l = n_slices if l_batch != n_batchs - 1 else nao - n_slices * l_batch

        i_slice = slice(n_slices * i_batch, n_slices * i_batch + nao_slice_i)
        j_slice = slice(n_slices * j_batch, n_slices * j_batch + nao_slice_j)
        k_slice = slice(n_slices * k_batch, n_slices * k_batch + nao_slice_k)
        l_slice = slice(n_slices * l_batch, n_slices * l_batch + nao_slice_l)
        yield i_slice, j_slice, k_slice, l_slice


def get_dft_energy(
    mol,
    grids,
    dft_mo_coeff,
    dm1_dft,
    evaluate=False,
):
    """Calculate restricted B3LYP energy-density components."""
    from cc2cc.utils.reference_b3lyp import get_dft_energy as shared_dft_energy

    return shared_dft_energy(
        mol, grids, dft_mo_coeff, dm1_dft,
        evaluate=evaluate, unrestricted=False,
    )
