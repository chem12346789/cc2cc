"""Shared restricted and unrestricted Hartree-Fock grid energies."""

import numpy as np
import pyscf.dft


def get_hf_energy(mol, grids, mf, dm1_hf, evaluate=False, *, unrestricted):
    """Calculate HF energy-density components on the supplied grid."""
    if evaluate and not unrestricted:
        return {}

    import gc
    import torch

    from cc2cc.utils.get_dft_energy_rks import (
        _ecp_energy_density,
        _nuclear_energy_density,
        _orbital_gradient_squared,
        block_loop_rdm2,
    )
    from cc2cc.utils.tensor_helper import parallel_einsum

    print("Using backend: numpy for get_hf_energy\n")
    ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
    ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
    ao_value = ao_value[:4]
    ao_1 = ao_value[1:4]

    if unrestricted:
        rho_alpha = pyscf.dft.numint.eval_rho(
            mol, ao_value, dm1_hf[0], xctype="GGA"
        )
        rho_beta = pyscf.dft.numint.eval_rho(
            mol, ao_value, dm1_hf[1], xctype="GGA"
        )
        if evaluate:
            return {}
        dm_total = dm1_hf[0] + dm1_hf[1]
        rho_total = rho_alpha[0] + rho_beta[0]
    else:
        dm_total = dm1_hf
        rho_total = pyscf.dft.numint.eval_rho(
            mol, ao_value, dm1_hf, xctype="GGA"
        )[0]
    ecp_hf_grids = _ecp_energy_density(mol, ao_value[0], dm_total)

    print("Start K part from HF...")
    exc_k_hf_grids = np.zeros_like(grids.weights)
    for i_slice, j_slice, k_slice, l_slice in block_loop_rdm2(mol.nao):
        # Exchange is spin-diagonal; RHF uses the spin-summed density.
        if unrestricted:
            dm12_hf = parallel_einsum(
                "pr,qs->pqrs",
                dm1_hf[0][i_slice, k_slice],
                dm1_hf[0][j_slice, l_slice],
            ) + parallel_einsum(
                "pr,qs->pqrs",
                dm1_hf[1][i_slice, k_slice],
                dm1_hf[1][j_slice, l_slice],
            )
        else:
            dm12_hf = None
        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
            ao_0_i = ao_value[0][i]
            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                if unrestricted:
                    exc_k_hf_grids[i] += -0.5 * parallel_einsum(
                        "ijkl,i,j,kl->",
                        dm12_hf,
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                    )
                else:
                    exc_k_hf_grids[i] += -0.25 * parallel_einsum(
                        "ik,jl,i,j,kl->",
                        dm1_hf[i_slice, k_slice],
                        dm1_hf[j_slice, l_slice],
                        ao_0_i[i_slice],
                        ao_0_i[j_slice],
                        rinv[k_slice, l_slice],
                    )
        gc.collect()
        torch.cuda.empty_cache()
    print("Exchange part done.\n")

    print("Start hatree part from HF...")
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    hatree_hf_grids = np.zeros_like(grids.weights)
    vele = parallel_einsum("pij,ij->p", int1e_grids, dm_total)
    hatree_hf_grids += 0.5 * vele * rho_total
    print("Hatree part done.\n")

    print("Start kinetic part from HF...")
    kin_hf_grids = np.zeros_like(grids.weights)
    kinl_hf_grids = np.zeros_like(grids.weights)
    factor = 0.5 if unrestricted else 1.0
    for i_spin in range(2 if unrestricted else 1):
        coeff = mf.mo_coeff[i_spin] if unrestricted else mf.mo_coeff
        for i in range(mol.nelec[i_spin]):
            part = parallel_einsum(
                "pm,m,n,pn->p",
                ao_value[0],
                coeff[:, i],
                coeff[:, i],
                ao_2_diag,
            )
            kin_hf_grids -= factor * part
            kinl_hf_grids += factor * _orbital_gradient_squared(ao_1, coeff[:, i])
    print("Kinetic part done.\n")

    print("Start nuclear part from HF...")
    nuc_hf_grids = _nuclear_energy_density(mol, grids, rho_total)
    print("Nuclear part done.\n")

    return {
        "exc_k_hf_grids": exc_k_hf_grids,
        "hatree_hf_grids": hatree_hf_grids,
        "kin_hf_grids": kin_hf_grids,
        "kinl_hf_grids": kinl_hf_grids,
        "nuc_hf_grids": nuc_hf_grids,
        "ecp_hf_grids": ecp_hf_grids,
    }
