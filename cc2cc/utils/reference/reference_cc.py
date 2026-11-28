"""Shared CCSD(T) settings, target densities, gradients, and grid energies."""

import numpy as np
from pyscf import cc, lib
from pyscf.cc import ccsd_rdm, ccsd_t_lambda, uccsd_rdm
from pyscf.grad import ccsd as ccsd_grad, uccsd as uccsd_grad

from cc2cc.utils.ccsd_t.pyscf_ccsd_t_rdm import gamma1_intermediates
from cc2cc.utils.ccsd_t.pyscf_uccsd_t_lambda import kernel as uccsd_t_lambda_kernel
from cc2cc.utils.ccsd_t.pyscf_uccsd_t_u_gamma1_intermediates import (
    u_gamma1_intermediates,
)

CC_MAX_CYCLE = 200
CC_CONV_TOL = 1e-12
CC_CONV_TOL_NORMT = 1e-8
LAMBDA_MAX_CYCLE = 200
LAMBDA_TOL = 1e-10


def cc_reference(mol, grids, mf, *, evaluate, unrestricted):
    """Generate CCSD(T) energy and training reference data."""
    import gc

    from pyscf.cc import ccsd_t, uccsd_t

    triples = uccsd_t if unrestricted else ccsd_t
    mycc = make_ccsd(mf, unrestricted=unrestricted)
    _require_no_frozen_orbitals(mycc, unrestricted=unrestricted)
    mycc.verbose = 4
    # To trace the usage of memory, set verbose as 9.
    _, t1, t2 = mycc.kernel()
    eris = mycc.ao2mo()
    e3ref = triples.kernel(mycc, eris, t1, t2)
    e_cc = mycc.e_tot + e3ref
    print(
        f"{'UCCSD(T)' if unrestricted else 'CCSD(T)'} energy (Hartree): {e_cc}",
        flush=unrestricted,
    )
    data = {"e_cc": e_cc}
    if evaluate:
        del t1, t2, eris, mycc
        gc.collect()
        return data

    l1, l2, d1, d1_grad, dm1_cc_mo, dm1_cc, d2, dm2_cc = ccsd_t_target_density(
        mycc, eris, unrestricted=unrestricted, with_rdm2=True
    )
    data["grad_cc"] = ccsd_t_target_gradient(
        mycc, eris, l1, l2, d1_grad, d2, unrestricted=unrestricted
    )

    del t1, t2, l1, l2
    gc.collect()

    print("DM1 done.", flush=True)
    print("DM2 done.", flush=True)
    del d1, d1_grad, d2, eris, mycc
    gc.collect()

    data.update(
        get_cc_energy(
            mol, grids, mf, dm1_cc, dm1_cc_mo, dm2_cc, unrestricted=unrestricted
        )
    )
    data["dm1_cc"] = dm1_cc
    gc.collect()
    return data


def make_ccsd(mf, *, unrestricted=False, reference=False):
    """Configure CCSD with identical generation/reference convergence settings.

    ``reference`` is retained for existing callers; it does not change settings.
    """
    mycc = cc.UCCSD(mf) if unrestricted else cc.CCSD(mf)
    mycc.max_cycle = CC_MAX_CYCLE
    mycc.conv_tol = CC_CONV_TOL
    mycc.conv_tol_normt = CC_CONV_TOL_NORMT
    return mycc


def ccsd_t_target_density(
    mycc, eris, *, unrestricted=False, reference=False, with_rdm2=False
):
    """Return l1, l2, density/gradient intermediates, MO density, AO density.

    Both intermediates entries contain full-block triples corrections. The
    returned densities include orbital relaxation; frozen orbitals are rejected.
    With ``with_rdm2=True``, also return d2 and the relaxed AO 2-RDM
    (an aa, ab, bb tuple for UHF), reusing the same triples intermediates.
    Reference mode requires converged CC and lambda equations.
    """
    from pyscf.cc.ccsd_t_rdm import _gamma2_intermediates
    from cc2cc.utils.ccsd_t.pyscf_uccsd_t_u_gamma2_intermediates import (
        u_gamma2_intermediates,
    )

    _require_no_frozen_orbitals(mycc, unrestricted=unrestricted)
    if reference and not mycc.converged:
        raise RuntimeError("Reference CCSD did not converge.")
    t1, t2 = mycc.t1, mycc.t2
    kernel = uccsd_t_lambda_kernel if unrestricted else ccsd_t_lambda.kernel
    time = (lib.logger.process_clock(), lib.logger.perf_counter())
    converged, l1, l2 = kernel(
        mycc,
        eris,
        t1,
        t2,
        max_cycle=LAMBDA_MAX_CYCLE,
        tol=LAMBDA_TOL,
    )
    time = lib.logger.timer(mycc, "CCSD(T) lambda", *time)
    if reference and not converged:
        raise RuntimeError("Reference CCSD(T) lambda equations did not converge.")

    if unrestricted:
        d1, (goo, gOO, gvv, gVV) = u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        (doo, dOO), (dov, dOV), (dvo, dVO), (dvv, dVV) = d1
        d1_grad = (
            (doo - goo, dOO - gOO),
            (dov, dOV),
            (dvo, dVO),
            (dvv + gvv, dVV + gVV),
        )
        rdm = uccsd_rdm
    else:
        d1, goo, gvv = gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        doo, dov, dvo, dvv = d1
        d1_grad = (doo - goo * 0.5, dov, dvo, dvv + gvv * 0.5)
        rdm = ccsd_rdm
    time = lib.logger.timer(mycc, "CCSD(T) gamma1", *time)

    gamma2 = u_gamma2_intermediates if unrestricted else _gamma2_intermediates
    d2 = gamma2(mycc, t1, t2, l1, l2, eris, compress_vvvv=False)
    dm2 = rdm._make_rdm2(
        mycc, d1_grad, d2, with_dm1=False, with_frozen=False, ao_repr=True
    )
    dm_mo = _orbital_relaxed_dm1(
        mycc, eris, d1_grad, dm2, unrestricted=unrestricted
    )
    coeff = np.asarray(mycc.mo_coeff)
    dm_ao = coeff @ dm_mo @ coeff.swapaxes(-1, -2).conj()
    time = lib.logger.timer(mycc, "CCSD(T) density", *time)
    result = (l1, l2, d1_grad, d1_grad, dm_mo, dm_ao)
    if with_rdm2:
        hf_dm = np.asarray(mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ))
        _add_separable_dm2(dm2, dm_ao, hf_dm, unrestricted=unrestricted)
        return (*result, d2, dm2)
    return result


def _require_no_frozen_orbitals(mycc, *, unrestricted):
    masks = mycc.get_frozen_mask()
    masks = masks if unrestricted else (masks,)
    if any(not np.all(mask) for mask in masks):
        raise ValueError("Frozen orbitals are not supported for CCSD(T) references.")


def _orbital_relaxed_dm1(mycc, eris, d1, dm2, *, unrestricted):
    """Build the same orbital-response RHS as the PySCF CC nuclear gradient."""
    mol, mf = mycc.mol, mycc._scf
    coeff = np.asarray(mycc.mo_coeff)
    rdm = uccsd_rdm if unrestricted else ccsd_rdm
    grad = uccsd_grad if unrestricted else ccsd_grad
    dm = np.asarray(rdm._make_rdm1(mycc, d1, with_frozen=False, with_mf=False))
    matrices = dm if unrestricted else dm[None]
    coefficients = coeff if unrestricted else coeff[None]
    occupations = mycc.mo_occ if unrestricted else (mycc.mo_occ,)
    spin_pairs = (
        ((dm2[0], dm2[1]), (dm2[2], dm2[1].transpose(2, 3, 0, 1)))
        if unrestricted else ((dm2,),)
    )
    imat = np.zeros((len(matrices), mol.nao, mol.nao), dtype=np.float64)
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    blksize = max(1, int(max_memory * 0.9e6 / (8 * mol.nao**3 * 4)))
    p1 = 0
    for b0, b1, nf in ccsd_grad._shell_prange(mol, 0, mol.nbas, blksize):
        p0, p1 = p1, p1 + nf
        eri = mol.intor(
            "int2e", shls_slice=(b0, b1, 0, mol.nbas, 0, mol.nbas, 0, mol.nbas)
        )
        for spin, pairs in enumerate(spin_pairs):
            for pair in pairs:
                # Only the ERI-symmetric part contributes to the orbital RHS.
                buf = pair[p0:p1] + pair[:, p0:p1].transpose(1, 0, 2, 3)
                imat[spin] -= 0.5 * lib.einsum("iprs,iqrs->pq", eri, buf)
        del eri, buf
    overlap = mf.get_ovlp()
    imat = [c.T @ im @ overlap @ c for c, im in zip(coefficients, imat)]
    for density, occ in zip(matrices, occupations):
        nocc = np.count_nonzero(occ > 0)
        # The gradient Z-vector replaces, rather than adds to, the
        # unrelaxed occupied-virtual blocks.
        density[:nocc, nocc:] = 0
        density[nocc:, :nocc] = 0
    dao = coeff @ dm @ coeff.swapaxes(-1, -2).conj()
    veff = np.asarray(mf.get_veff(mol, dao))
    potentials = veff if unrestricted else (veff * 2,)
    rhs = []
    for c, occ, v, im in zip(coefficients, occupations, potentials, imat):
        nocc = np.count_nonzero(occ > 0)
        rhs.append(
            c[:, nocc:].T @ v @ c[:, :nocc]
            + im[:nocc, nocc:].T - im[nocc:, :nocc]
        )
    dm += np.asarray(
        grad._response_dm1(mycc, tuple(rhs) if unrestricted else rhs[0], eris)
    )
    for density, occ in zip(matrices, occupations):
        density[np.diag_indices_from(density)] += occ
    return (dm + dm.swapaxes(-1, -2).conj()) * 0.5


def _add_separable_dm2(dm2, dm1, hf_dm1, *, unrestricted):
    """Complete the nonseparable AO 2-RDM in PySCF's chemist convention, in place."""
    # Half the HF density supplies the reference determinant once; the rest
    # gives the separable correlated and orbital-response contributions.
    correction = dm1 - hf_dm1 * 0.5
    same_spin = (
        zip((dm2[0], dm2[2]), correction, hf_dm1)
        if unrestricted else ((dm2, correction, hf_dm1),)
    )
    exchange = 1.0 if unrestricted else 0.5
    for pair, delta, hf in same_spin:
        for p in range(hf.shape[0]):
            pair[p] += np.einsum("q,rs->qrs", delta[p], hf)
            pair[p] += np.einsum("q,rs->qrs", hf[p], delta)
            pair[p] -= exchange * np.einsum("s,rq->qrs", delta[p], hf)
            pair[p] -= exchange * np.einsum("s,rq->qrs", hf[p], delta)
    if unrestricted:
        for p in range(hf_dm1[0].shape[0]):
            dm2[1][p] += np.einsum("q,rs->qrs", correction[0][p], hf_dm1[1])
            dm2[1][p] += np.einsum("q,rs->qrs", hf_dm1[0][p], correction[1])


def ccsd_t_target_gradient(mycc, eris, l1, l2, d1_grad, d2, *, unrestricted=False):
    """Compute the target nuclear gradient in Hartree/Bohr without modifying d2."""
    _require_no_frozen_orbitals(mycc, unrestricted=unrestricted)
    mol = mycc.mol
    if mol.natm == 1:
        gradient = np.zeros((mol.natm, 3), dtype=np.float64)
    elif mol.nelectron == 1:
        gradient = mycc._scf.nuc_grad_method().kernel()
    else:
        if unrestricted:
            (
                (dovov, dovOV, dOVov, dOVOV),
                (dvvvv, dvvVV, dVVvv, dVVVV),
                (doooo, dooOO, dOOoo, dOOOO),
                (doovv, dooVV, dOOvv, dOOVV),
                (dovvo, dovVO, dOVvo, dOVVO),
                (dvvov, dvvOV, dVVov, dVVOV),
                (dovvv, dovVV, dOVvv, dOVVV),
                (dooov, dooOV, dOOov, dOOOV),
            ) = d2
            _, _, nvira, nvirb = mycc.t2[1].shape
            idxa = np.tril_indices(nvira)
            idxa = idxa[0] * nvira + idxa[1]
            idxb = np.tril_indices(nvirb)
            idxb = idxb[0] * nvirb + idxb[1]
            dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
            dvvvv = lib.take_2d(dvvvv.reshape(nvira**2, nvira**2), idxa, idxa)
            dvvvv *= 0.5
            dvvVV = dvvVV + dvvVV.transpose(1, 0, 2, 3)
            dvvVV = lib.take_2d(dvvVV.reshape(nvira**2, nvirb**2), idxa, idxb)
            dVVVV = dVVVV + dVVVV.transpose(1, 0, 2, 3)
            dVVVV = lib.take_2d(dVVVV.reshape(nvirb**2, nvirb**2), idxb, idxb)
            dVVVV *= 0.5
            d2_grad = (
                (dovov, dovOV, dOVov, dOVOV),
                (dvvvv, dvvVV, dVVvv, dVVVV),
                (doooo, dooOO, dOOoo, dOOOO),
                (doovv, dooVV, dOOvv, dOOVV),
                (dovvo, dovVO, dOVvo, dOVVO),
                (dvvov, dvvOV, dVVov, dVVOV),
                (dovvv, dovVV, dOVvv, dOVVV),
                (dooov, dooOV, dOOov, dOOOV),
            )
            grad_module = uccsd_grad
        else:
            dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
            nvir = mycc.nmo - mycc.nocc
            idx = np.tril_indices(nvir)
            vidx = idx[0] * nvir + idx[1]
            dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
            dvvvv = dvvvv + dvvvv.transpose(0, 1, 3, 2)
            dvvvv = lib.take_2d(dvvvv.reshape(nvir**2, nvir**2), vidx, vidx)
            dvvvv *= 0.25
            d2_grad = (dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov)
            grad_module = ccsd_grad
        cc_grad = grad_module.Gradients(mycc)
        de = grad_module.grad_elec(
            cc_grad, mycc.t1, mycc.t2, l1, l2, eris, d1=d1_grad, d2=d2_grad
        )
        cc_grad.de = de + cc_grad.grad_nuc(atmlst=cc_grad.atmlst)
        if mol.symmetry:
            cc_grad.de = cc_grad.symmetrize(cc_grad.de, cc_grad.atmlst)
        cc_grad._finalize()
        gradient = cc_grad.de
    print("Grad done.", flush=True)
    return gradient


def get_cc_energy(
    mol, grids, mf, dm1_cc, dm1_cc_mo, dm2_cc, *, unrestricted,
):
    """Calculate CC energy-density components for training on the supplied grid."""
    import gc
    import torch
    import pyscf.dft

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
        dm_total = dm1_cc[0] + dm1_cc[1]
        rho_total = (
            pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[0], xctype="GGA")[0]
            + pyscf.dft.numint.eval_rho(mol, ao_value, dm1_cc[1], xctype="GGA")[0]
        )
    else:
        dm_total = dm1_cc
        rho_total = pyscf.dft.numint.eval_rho(
            mol, ao_value, dm1_cc, xctype="GGA"
        )[0]
    ecp_cc_grids = ecp_energy_density(mol, ao_value[0], dm_total)

    print("Start exchange part...")
    exc_cc_grids = np.zeros_like(grids.weights)
    for i_slice, j_slice, k_slice, l_slice in block_loop_rdm2(mol.nao):
        if unrestricted:
            # The beta-alpha block is the pair-transposed alpha-beta block.
            dm12_cc = (
                0.5 * dm2_cc[0][i_slice, j_slice, k_slice, l_slice]
                + 0.5 * dm2_cc[1][i_slice, j_slice, k_slice, l_slice]
                + 0.5
                * dm2_cc[1].transpose(2, 3, 0, 1)[
                    i_slice, j_slice, k_slice, l_slice
                ]
                + 0.5 * dm2_cc[2][i_slice, j_slice, k_slice, l_slice]
            )
        else:
            dm12_cc = 0.5 * dm2_cc[i_slice, j_slice, k_slice, l_slice]
        dm12_cc = dm12_cc - 0.5 * parallel_einsum(
            "pq,rs->pqrs",
            dm_total[i_slice, j_slice],
            dm_total[k_slice, l_slice],
        )

        for i, coord in enumerate(grids.coords):
            if i * 10 % len(grids.coords) == 0:
                print(f"Progress: {(i*100)/len(grids.coords):.1f}%", flush=True)
            ao_0_i = ao_value[0][i]
            with mol.with_rinv_origin(coord):
                rinv = mol.intor("int1e_rinv")
                exc_cc_grids[i] += parallel_einsum(
                    "ijkl,i,j,kl->",
                    dm12_cc,
                    ao_0_i[i_slice],
                    ao_0_i[j_slice],
                    rinv[k_slice, l_slice],
                )
        del dm12_cc
        gc.collect()
        torch.cuda.empty_cache()
    print("Exchange part done.\n")

    print("Start hatree part...")
    hatree_cc_grids = np.zeros_like(grids.weights)
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    vele = parallel_einsum("pij,ij->p", int1e_grids, dm_total)
    hatree_cc_grids += 0.5 * vele * rho_total
    print("Hatree part done.\n")

    print("Start kinetic part...")
    kin_cc_grids = np.zeros_like(grids.weights)
    kinl_cc_grids = np.zeros_like(grids.weights)
    for i_spin in range(2 if unrestricted else 1):
        coeff = mf.mo_coeff[i_spin] if unrestricted else mf.mo_coeff
        density = dm1_cc_mo[i_spin] if unrestricted else dm1_cc_mo
        eigs_e_dm1, eigs_v_dm1 = np.linalg.eigh(density)
        eigs_v_dm1 = coeff @ eigs_v_dm1
        for i in range(np.shape(eigs_v_dm1)[1]):
            part = parallel_einsum(
                "pm,m,n,pn->p",
                ao_value[0],
                eigs_v_dm1[:, i],
                eigs_v_dm1[:, i],
                ao_2_diag,
            )
            kin_cc_grids -= part * eigs_e_dm1[i] / 2
            kinl_cc_grids += (
                orbital_gradient_squared(ao_1, eigs_v_dm1[:, i])
                * eigs_e_dm1[i]
                / 2
            )
    print("Kinetic part done.\n")

    print("Start nuclear part...")
    nuc_cc_grids = nuclear_energy_density(mol, grids, rho_total)
    print("Nuclear part done.\n")

    tol_cc_grids = (
        exc_cc_grids + hatree_cc_grids + kin_cc_grids + nuc_cc_grids + ecp_cc_grids
    )
    return {
        "exc_cc_grids": exc_cc_grids,
        "hatree_cc_grids": hatree_cc_grids,
        "kin_cc_grids": kin_cc_grids,
        "kinl_cc_grids": kinl_cc_grids,
        "nuc_cc_grids": nuc_cc_grids,
        "ecp_cc_grids": ecp_cc_grids,
        "tol_cc_grids": tol_cc_grids,
    }
