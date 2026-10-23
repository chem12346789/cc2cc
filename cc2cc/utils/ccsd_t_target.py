"""Shared CCSD(T) settings, triples-corrected target densities and gradients."""

import numpy as np
from pyscf import cc, lib
from pyscf.cc import ccsd_rdm, ccsd_t_lambda, uccsd_rdm
from pyscf.grad import ccsd as ccsd_grad, uccsd as uccsd_grad

from cc2cc.utils.pyscf_ccsd_t_rdm import _gamma1_intermediates
from cc2cc.utils.pyscf_uccsd_t_lambda import kernel as uccsd_t_lambda_kernel
from cc2cc.utils.pyscf_uccsd_t_u_gamma1_intermediates import u_gamma1_intermediates

CC_MAX_CYCLE = 200
CC_CONV_TOL = 1e-12
CC_CONV_TOL_NORMT = 1e-10
LAMBDA_MAX_CYCLE = 200
LAMBDA_TOL = 1e-10


def make_ccsd(mf, *, unrestricted=False, reference=False):
    """Configure CCSD with identical generation/reference convergence settings.

    ``reference`` is retained for existing callers; it does not change settings.
    """
    mycc = cc.UCCSD(mf) if unrestricted else cc.CCSD(mf)
    mycc.max_cycle = CC_MAX_CYCLE
    mycc.conv_tol = CC_CONV_TOL
    mycc.conv_tol_normt = CC_CONV_TOL_NORMT
    return mycc


def ccsd_t_target_density(mycc, eris, *, unrestricted=False, reference=False):
    """Return l1, l2, density/gradient intermediates, MO density, AO density.

    The density uses diagonal triples corrections; the CC gradient requires
    full-block corrections instead. Keep separate arrays for these two uses.
    Reference mode requires converged CC and lambda equations.
    """
    if reference and not mycc.converged:
        raise RuntimeError("Reference CCSD did not converge.")
    t1, t2 = mycc.t1, mycc.t2
    kernel = uccsd_t_lambda_kernel if unrestricted else ccsd_t_lambda.kernel
    converged, l1, l2 = kernel(
        mycc,
        eris,
        t1,
        t2,
        max_cycle=LAMBDA_MAX_CYCLE,
        tol=LAMBDA_TOL,
    )
    if reference and not converged:
        raise RuntimeError("Reference CCSD(T) lambda equations did not converge.")
    if unrestricted:
        print("uccsd_t_lambda DONE", flush=True)
        d1, (goo, gOO, gvv, gVV) = u_gamma1_intermediates(
            mycc, t1, t2, l1, l2, eris
        )
        print("u_gamma1_intermediates DONE", flush=True)
        (doo, dOO), (dov, dOV), (dvo, dVO), (dvv, dVV) = d1
        d1_grad = (
            (doo - goo, dOO - gOO),
            (dov, dOV),
            (dvo, dVO),
            (dvv + gvv, dVV + gVV),
        )
        corrections = ((doo, -goo), (dOO, -gOO), (dvv, gvv), (dVV, gVV))
        rdm = uccsd_rdm
    else:
        d1, goo, gvv = _gamma1_intermediates(mycc, t1, t2, l1, l2, eris)
        doo, dov, dvo, dvv = d1
        d1_grad = (doo - goo * 0.5, dov, dvo, dvv + gvv * 0.5)
        corrections = ((doo, -goo * 0.5), (dvv, gvv * 0.5))
        rdm = ccsd_rdm
    for block, correction in corrections:
        block[np.diag_indices_from(block)] += correction.diagonal()
    dm_mo = np.asarray(
        rdm._make_rdm1(mycc, d1, with_frozen=True, ao_repr=False)
    )
    coeff = np.asarray(mycc.mo_coeff)
    dm_ao = coeff @ dm_mo @ coeff.swapaxes(-1, -2).conj()
    return l1, l2, d1, d1_grad, dm_mo, dm_ao


def ccsd_t_target_gradient(mycc, eris, l1, l2, d1_grad, d2, *, unrestricted=False):
    """Compute the target nuclear gradient in Hartree/Bohr without modifying d2."""
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
