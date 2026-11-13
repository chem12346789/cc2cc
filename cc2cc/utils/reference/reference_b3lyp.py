"""Shared B3LYP calculations and finite-difference references."""

from __future__ import annotations

from typing import cast

import numpy as np

import pyscf.dft


def converged_b3lyp(
    mol, dm0=None, *, nlc="", grids=None, verbose=0, use_gpu=False,
    unrestricted=None,
):
    if unrestricted is None:
        unrestricted = mol.spin != 0 or (dm0 is not None and np.ndim(dm0) == 3)
    mf_class = pyscf.dft.UKS if unrestricted else pyscf.dft.RKS
    mf = mf_class(mol, xc="b3lyp")
    mf = cast(
        pyscf.dft.rks.RKS | pyscf.dft.roks.ROKS | pyscf.dft.uks.UKS,
        mf.density_fit(),
    )
    if grids is not None:
        mf.grids = grids
    mf.verbose = verbose
    mf.max_cycle = 200
    mf.nlc = nlc
    if use_gpu:
        mf = mf.to_gpu()
    mf.kernel(dm0=dm0)
    return mf


def b3lyp_finite_difference_gradient(mol, step=1e-3, *, unrestricted=None, dm0=None):
    """Central-difference B3LYP gradient in Hartree/Bohr; step in Bohr."""
    return finite_difference_gradient(
        mol,
        lambda displaced: converged_b3lyp(
            displaced, dm0=dm0, unrestricted=unrestricted
        ).e_tot,
        step,
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


def get_dft_energy(
    mol, grids, dft_mo_coeff, dm1_dft, evaluate=False, *, unrestricted=False,
):
    """Calculate B3LYP energy-density components on the supplied grid."""
    if evaluate and not unrestricted:
        return {}

    import gc
    import torch

    from cc2cc.utils.helper.chemistry.grid_energy import (
        ecp_energy_density,
        nuclear_energy_density,
        orbital_gradient_squared,
        block_loop_rdm2,
    )
    from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum

    print("Using backend: numpy for get_dft_energy\n")
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    if unrestricted:
        rho_dft = [
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[0], xctype="GGA"),
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft[1], xctype="GGA"),
        ]
        dm_total = dm1_dft[0] + dm1_dft[1]
        rho_total = rho_dft[0][0] + rho_dft[1][0]
    else:
        rho_dft = pyscf.dft.numint.eval_rho(mol, ao_value, dm1_dft, xctype="GGA")
        dm_total = dm1_dft
        rho_total = rho_dft[0]

    if evaluate:
        return {}

    ecp_dft_grids = ecp_energy_density(mol, ao_value[0], dm_total)
    exc_dft_grids = (
        pyscf.dft.libxc.eval_xc("b3lyp", rho_dft, spin=int(unrestricted))[0]
        * rho_total
    )

    print("Start K part...")
    exc_k_dft_grids = np.zeros_like(grids.weights)
    for i_slice, j_slice, k_slice, l_slice in block_loop_rdm2(mol.nao):
        dm12 = None
        if unrestricted:
            dm12 = parallel_einsum(
                "pr,qs->pqrs",
                dm1_dft[0][i_slice, k_slice],
                dm1_dft[0][j_slice, l_slice],
            ) + parallel_einsum(
                "pr,qs->pqrs",
                dm1_dft[1][i_slice, k_slice],
                dm1_dft[1][j_slice, l_slice],
            )
        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
            ao_0_i = ao_value[0][i]
            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                if unrestricted:
                    exc_k_dft_grids[i] += -0.1 * parallel_einsum(
                        "ijkl,i,j,kl->",
                        dm12,
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                    )
                else:
                    exc_k_dft_grids[i] += -0.05 * parallel_einsum(
                        "ik,jl,i,j,kl->",
                        dm1_dft[i_slice, k_slice],
                        dm1_dft[j_slice, l_slice],
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                    )
        if unrestricted:
            del dm12
        gc.collect()
        torch.cuda.empty_cache()
    print("Exchange part done.\n")

    print("Start hatree part...")
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    vele = parallel_einsum("pij,ij->p", int1e_grids, dm_total)
    hatree_dft_grids = np.zeros_like(grids.weights)
    hatree_dft_grids += 0.5 * vele * rho_total
    print("Hatree part done.\n")

    print("Start kinetic part...")
    kin_dft_grids = np.zeros_like(grids.weights)
    kinl_dft_grids = np.zeros_like(grids.weights)
    coefficients = dft_mo_coeff if unrestricted else (dft_mo_coeff,)
    occupations = mol.nelec if unrestricted else (mol.nelec[0],)
    factor = 0.5 if unrestricted else 1.0
    for coeff, nocc in zip(coefficients, occupations):
        for i in range(nocc):
            part = parallel_einsum(
                "pm,m,n,pn->p",
                ao_value[0],
                coeff[:, i],
                coeff[:, i],
                ao_2_diag,
            )
            kin_dft_grids -= factor * part
            kinl_dft_grids += factor * orbital_gradient_squared(ao_1, coeff[:, i])
    print("Kinetic part done.\n")

    print("Start nuclear part...")
    nuc_dft_grids = nuclear_energy_density(mol, grids, rho_total)
    print("Nuclear part done.\n")

    tol_dft_grids = (
        exc_dft_grids
        + exc_k_dft_grids
        + hatree_dft_grids
        + kin_dft_grids
        + nuc_dft_grids
        + ecp_dft_grids
    )
    return {
        "exc_dft_grids": exc_dft_grids,
        "exc_k_dft_grids": exc_k_dft_grids,
        "hatree_dft_grids": hatree_dft_grids,
        "kin_dft_grids": kin_dft_grids,
        "kinl_dft_grids": kinl_dft_grids,
        "nuc_dft_grids": nuc_dft_grids,
        "ecp_dft_grids": ecp_dft_grids,
        "tol_dft_grids": tol_dft_grids,
    }


def get_dft_input(mol, grids, dm1_dft, max_memory=8000, *, unrestricted=False):
    """Calculate B3LYP grid inputs with first-order AO derivatives in both modes."""
    if dm1_dft is None:
        return {}

    from pyscf import lib
    from cc2cc.utils.helper.runtime.env_var import EDGE_SIZE

    mdft = converged_b3lyp(
        mol, dm0=dm1_dft, grids=grids if unrestricted else None,
        verbose=4, unrestricted=unrestricted,
    )
    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))
    ni = mdft._numint
    step = int(max_memory * 1024**2 / (dm1_dft.shape[-1] * EDGE_SIZE**3 * 32 * 8))
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        mask = None if grids.screen_index is None else grids.screen_index[p0:p1]
        coords = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords, mask)
        cube_rho = (
            gridcube.gen_cube_rho_uks if unrestricted else gridcube.gen_cube_rho_rks
        )
        rho_part, _, _ = cube_rho(ni, dm1_dft, ao_deriv=1)
        rho_cube_dft[p0:p1] = rho_part

    return {
        "rho_cube_dft": rho_cube_dft.reshape(
            len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
        )
    }


def get_dft_grad(mol, grids, dm1_dft, max_memory=8000, *, unrestricted=False):
    """Return feature and weight nuclear derivatives including moving grids."""
    if dm1_dft is None:
        return {}

    from pyscf import lib
    from pyscf.grad import rks as rks_grad, uks as uks_grad
    from cc2cc.utils.helper.runtime.env_var import CUBE_MIDDLE, EDGE_SIZE
    from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum

    atmlst = range(mol.natm)
    grad2force = np.zeros(
        (
            len(atmlst),
            grids.input_level,
            len(grids.coords),
            EDGE_SIZE,
            EDGE_SIZE,
            EDGE_SIZE,
            3,
        )
    )
    weight_response = np.zeros((mol.natm, 3, len(grids.coords)), dtype=np.float64)
    grid_owner = np.full(len(grids.coords), -1, dtype=np.int32)
    offset = 0
    for atom, (coords, weights, weight1) in enumerate(
        rks_grad.grids_response_cc(grids)
    ):
        stop = offset + len(weights)
        if (
            not np.allclose(grids.coords[offset:stop], coords, rtol=0, atol=1e-12)
            or not np.allclose(grids.weights[offset:stop], weights, rtol=1e-12, atol=1e-15)
        ):
            raise ValueError("Force mapping requires the original atom-ordered grid.")
        weight_response[:, :, offset:stop] = weight1
        grid_owner[offset:stop] = atom
        offset = stop
    if np.any(grids.weights[offset:] != 0):
        raise ValueError("Unexpected nonzero grid weights after atom-centered blocks.")
    rho_cube_dft = np.zeros((len(grids.coords), grids.input_level, EDGE_SIZE**3))
    spin_dm = dm1_dft if unrestricted else dm1_dft[None, :, :]
    ni = pyscf.dft.numint.NumInt()
    step = int(max_memory * 1024**2 / (dm1_dft.shape[-1] * EDGE_SIZE**3 * 32 * 8))
    print(f"Step size: {step}")
    for p0, p1 in lib.prange(0, len(grids.coords), step):
        mask = None if grids.screen_index is None else grids.screen_index[p0:p1]
        coords = grids.coords[p0:p1]
        gridcube = grids.gen_cube(mol, dm1_dft, coords, mask)
        cube_rho = (
            gridcube.gen_cube_rho_uks if unrestricted else gridcube.gen_cube_rho_rks
        )
        rho_part, wv, ao_value = cube_rho(ni, dm1_dft, ao_deriv=2)
        rho_cube_dft[p0:p1] = rho_part

        wv = wv.reshape(
            gridcube.input_level, 2 if unrestricted else 1, 4, len(gridcube.coords)
        )
        wv[:, :, 0, :] *= 0.5
        ao_array = np.array([ao_value[0], ao_value[1], ao_value[2], ao_value[3]])
        ao_mat = np.array(
            [
                [ao_value[1], ao_value[2], ao_value[3]],
                [ao_value[4], ao_value[5], ao_value[6]],
                [ao_value[5], ao_value[7], ao_value[8]],
                [ao_value[6], ao_value[8], ao_value[9]],
            ]
        )
        for k, ia in enumerate(atmlst):
            ao0, ao1 = mol.aoslice_by_atom()[ia, 2:]
            print(
                f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements, size of wv: {wv.shape} elements",
                flush=True,
            )
            grad2force_part = parallel_einsum(
                "isnp,xpu,npv,suv->ipx",
                wv,
                ao_value[1:4, :, ao0:ao1],
                ao_array,
                spin_dm[:, ao0:ao1],
            ) + parallel_einsum(
                "isnp,nxpu,pv,suv->ipx",
                wv,
                ao_mat[:, :, :, ao0:ao1],
                ao_value[0],
                spin_dm[:, ao0:ao1],
            )
            grad2force_part = -2 * grad2force_part
            grad2force[k, :, p0:p1] = np.reshape(
                grad2force_part,
                (grids.input_level, p1 - p0, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE, 3),
            )
        # Rigidly translating the sampling cube cancels the summed AO motion.
        grid_motion = -grad2force[:, :, p0:p1].sum(axis=0)
        for atom in atmlst:
            local = np.flatnonzero(grid_owner[p0:p1] == atom)
            grad2force[atom, :, p0 + local] += grid_motion[:, local].swapaxes(0, 1)
        print(
            f"current p0: {p0}, p1: {p1}, current size: {lib.current_memory()[0] / 1024:.2f} GB, max size: {max_memory / 1024:.2f} GB",
            flush=not unrestricted,
        )
        print(
            f"size of ao_array: {ao_array.shape} elements, size of ao_mat: {ao_mat.shape} elements, size of ao_value: {ao_value.shape} elements, size of grad2force: {grad2force.shape} elements",
            flush=True,
        )

    grad_mat = np.zeros(
        (grids.input_level, len(grids.coords), EDGE_SIZE, EDGE_SIZE, EDGE_SIZE)
    )
    grad_mat[0, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.08
    grad_mat[1, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.19
    grad_mat[2, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.72
    grad_mat[3, :, CUBE_MIDDLE, CUBE_MIDDLE, CUBE_MIDDLE] += 0.81
    force = parallel_einsum(
        "p,ipabc,tipabcx->tx",
        grids.weights,
        grad_mat,
        grad2force,
    )
    feature_energy = parallel_einsum(
        "ipabc,piabc->p",
        grad_mat,
        rho_cube_dft.reshape(
            len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
        ),
    )
    force += parallel_einsum("p,txp->tx", feature_energy, weight_response)
    get_vxc_full_response = (
        uks_grad.get_vxc_full_response if unrestricted
        else rks_grad.get_vxc_full_response
    )
    xc_grid_response, xc_matrix = get_vxc_full_response(
        ni, mol, grids, "b3lyp", dm1_dft, max_memory=max_memory
    )
    xc_matrix = xc_matrix if unrestricted else xc_matrix[None]
    reference = np.array(xc_grid_response, copy=True)
    for atom, (_, _, ao0, ao1) in enumerate(mol.aoslice_by_atom()):
        reference[atom] += 2 * parallel_einsum(
            "sxij,sij->x", xc_matrix[:, :, ao0:ao1], spin_dm[:, ao0:ao1]
        )
    print("Error force DFT: ", np.linalg.norm(force - reference))

    return {
        "rho_cube_dft": rho_cube_dft.reshape(
            len(grids.coords), grids.input_level, EDGE_SIZE, EDGE_SIZE, EDGE_SIZE
        ),
        "grad2force": grad2force,
        "grid_weight_response": weight_response,
    }
