import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm

from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum, w_einsum
from cc2cc.utils.ccsd_t.pyscf_uccsd_t_lambda import (
    mixed_spin_divide,
    mixed_spin_v,
    run_fused,
    same_spin_r6_divide,
    same_spin_v,
    triples_block_size,
)


def u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None, compress_vvvv=False):
    """Build unrestricted 2-RDM intermediates."""
    log = logger.Logger(mycc.stdout, mycc.verbose)
    d2 = uccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)

    if eris is None:
        eris = mycc.ao2mo()

    dovov, dovOV, dOVov, dOVOV = d2[0]
    dvvvv, dvvVV, dVVvv, dVVVV = d2[1]
    doooo, dooOO, dOOoo, dOOOO = d2[2]
    doovv, dooVV, dOOvv, dOOVV = d2[3]
    dovvo, dovVO, dOVvo, dOVVO = d2[4]
    dvvov, dvvOV, dVVov, dVVOV = d2[5]
    dovvv, dovVV, dOVvv, dOVVV = d2[6]
    dooov, dooOV, dOOov, dOOOV = d2[7]

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    has_alpha = nocca > 0 and nvira > 0
    has_beta = noccb > 0 and nvirb > 0
    has_mixed = has_alpha and has_beta
    nmoa = eris.focka.shape[0]
    nmob = eris.fockb.shape[0]
    mo_ea, mo_eb = eris.mo_energy

    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)
    eris_OVVV = numpy.asarray(eris.get_OVVV())
    eris_OVOO = numpy.asarray(eris.OVOO)
    eris_OVOV = numpy.asarray(eris.OVOV)
    eris_ovVV = numpy.asarray(eris.get_ovVV())
    eris_OVvv = numpy.asarray(eris.get_OVvv())
    eris_ovOO = numpy.asarray(eris.ovOO)
    eris_OVoo = numpy.asarray(eris.OVoo)
    eris_ovOV = numpy.asarray(eris.ovOV)

    eia = mo_ea[:nocca, None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:, :nocca]
    fVO = eris.fockb[noccb:, :noccb]

    # aaa
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = triples_block_size(
        max_memory, nvira, nocca**3 * nvira, 1 / 2
    )
    if blksize < nvira:
        blksize = min(blksize, (nvira + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 1 rdm _gamma2_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocca,
        nvira,
        blksize,
    )

    for b0, b1 in lib.prange(0, nvira if has_alpha else 0, blksize):
        for c0, c1 in lib.prange(0, nvira, blksize):
            w_blk = w_einsum(
                "ijae,kceb->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, c0:c1, :, b0:b1],
            )
            w_blk = w_einsum(
                "kice,jbea->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, b0:b1, :, :],
                out=w_blk,
            )
            w_blk = w_einsum(
                "jkbe,iaec->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "ikae,jbec->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, b0:b1, :, c0:c1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "kjce,iaeb->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, :, :, b0:b1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "jibe,kcea->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, c0:c1, :, :],
                out=w_blk,
            )
            w_blk = w_einsum(
                "mkbc,iajm->ijkabc",
                t2aa[:, :, b0:b1, c0:c1],
                eris_ovoo[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "mjab,kcim->ijkabc",
                t2aa[:, :, :, b0:b1],
                eris_ovoo[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "mica,jbkm->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovoo[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "mjcb,iakm->ijkabc",
                t2aa[:, :, c0:c1, b0:b1],
                eris_ovoo[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "miba,kcjm->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovoo[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "mkac,jbim->ijkabc",
                t2aa[:, :, :, c0:c1],
                eris_ovoo[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            v_blk = run_fused(
                same_spin_v, eris_ovov, t1a, t2aa, fvo, b0, b1, c0, c1
            )

            rw = run_fused(same_spin_r6_divide, w_blk, None, eia, b0, c0)
            wvd = run_fused(same_spin_r6_divide, w_blk, v_blk, eia, b0, c0)

            dovov[:, b0:b1, :, c0:c1] += (
                parallel_einsum("ia,ijkabc->jbkc", t1a, rw) * 0.25
            )
            # *(1/8) instead of (1/4) because ooov appears 4 times in the 2pdm tensor due
            # to symmetrization, and its contribution is scaled by 1/2 in Tr(H,2pdm)
            dooov -= (
                parallel_einsum("mkbc,ijkabc->jmia", t2aa[:, :, b0:b1, c0:c1], wvd)
                * 0.125
            )
            dovvv[:, :, :, b0:b1] += (
                parallel_einsum("kjcf,ijkabc->iafb", t2aa[:, :, c0:c1, :], wvd) * 0.125
            )

    # bbb
    blksize = triples_block_size(
        max_memory, nvirb, noccb**3 * nvirb, 1 / 2
    )
    if blksize < nvirb:
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm _gamma2_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for b0, b1 in lib.prange(0, nvirb if has_beta else 0, blksize):
        for c0, c1 in lib.prange(0, nvirb, blksize):
            w_blk = w_einsum(
                "ijae,kceb->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, c0:c1, :, b0:b1],
            )
            w_blk = w_einsum(
                "kice,jbea->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, b0:b1, :, :],
                out=w_blk,
            )
            w_blk = w_einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "ikae,jbec->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, b0:b1, :, c0:c1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "kjce,iaeb->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, :, :, b0:b1],
                out=w_blk,
            )
            w_blk = w_einsum(
                "jibe,kcea->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, c0:c1, :, :],
                out=w_blk,
            )
            w_blk = w_einsum(
                "imab,kcjm->ijkabc",
                t2bb[:, :, :, b0:b1],
                eris_OVOO[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, b0:b1, c0:c1],
                eris_OVOO[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "kmca,jbim->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVOO[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "imac,jbkm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_OVOO[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "kmcb,iajm->ijkabc",
                t2bb[:, :, c0:c1, b0:b1],
                eris_OVOO[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = w_einsum(
                "jmba,kcim->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVOO[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            v_blk = run_fused(
                same_spin_v, eris_OVOV, t1b, t2bb, fVO, b0, b1, c0, c1
            )

            rw = run_fused(same_spin_r6_divide, w_blk, None, eIA, b0, c0)
            wvd = run_fused(same_spin_r6_divide, w_blk, v_blk, eIA, b0, c0)

            dOVOV[:, b0:b1, :, c0:c1] += (
                parallel_einsum("ia,ijkabc->jbkc", t1b, rw) * 0.25
            )
            dOOOV -= (
                parallel_einsum("mkbc,ijkabc->jmia", t2bb[:, :, b0:b1, c0:c1], wvd)
                * 0.125
            )
            dOVVV[:, :, :, b0:b1] += (
                parallel_einsum("kjcf,ijkabc->iafb", t2bb[:, :, c0:c1, :], wvd) * 0.125
            )

    # baa
    blksize = triples_block_size(
        max_memory, nvirb, nocca**2 * noccb, 1 / 3
    )
    blksize = max(1, min(nvira, blksize))
    if blksize < nvirb:
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm _gamma2_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for a0, a1 in lib.prange(0, nvirb if has_mixed else 0, blksize):
        for b0, b1 in lib.prange(0, nvira, blksize):
            for c0, c1 in lib.prange(0, nvira, blksize):
                w_blk = w_einsum(
                    "jIeA,kceb->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    scale=2,
                )
                w_blk = w_einsum(
                    "jIbE,kcEA->IjkAbc",
                    t2ab[:, :, b0:b1, :],
                    eris_ovVV[:, c0:c1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                    out=w_blk,
                )
                w_blk = w_einsum(
                    "mIbA,kcjm->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovoo[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jMbA,kcIM->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovOO[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )
                v_blk = run_fused(
                    mixed_spin_v,
                    eris_ovov[:, b0:b1, :, c0:c1],
                    t1b[:, a0:a1],
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                    t2aa[:, :, b0:b1, c0:c1],
                    fVO[a0:a1, :],
                    t2ab[:, :, c0:c1, a0:a1],
                    fvo[b0:b1, :],
                )

                w_blk = w_einsum(
                    "kIeA,jceb->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kIbE,jcEA->IjkAbc",
                    t2ab[:, :, b0:b1, :],
                    eris_ovVV[:, c0:c1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                    out=w_blk, subtract=True,
                )
                w_blk = w_einsum(
                    "mIbA,jckm->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovoo[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kMbA,jcIM->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovOO[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk,
                )

                w_blk = w_einsum(
                    "kIeA,jbec->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kIcE,jbEA->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovVV[:, b0:b1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                    out=w_blk,
                )
                w_blk = w_einsum(
                    "mIcA,jbkm->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovoo[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kMcA,jbIM->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovOO[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )

                w_blk = w_einsum(
                    "jIeA,kbec->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jIcE,kbEA->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovVV[:, b0:b1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                    out=w_blk, subtract=True,
                )
                w_blk = w_einsum(
                    "mIcA,kbjm->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovoo[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "jMcA,kbIM->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovOO[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk,
                )

                run_fused(
                    mixed_spin_v,
                    eris_ovov[:, b0:b1, :, c0:c1],
                    t1b[:, a0:a1],
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                    t2aa[:, :, b0:b1, c0:c1],
                    fVO[a0:a1, :],
                    t2ab[:, :, c0:c1, a0:a1],
                    fvo[b0:b1, :],
                    v_blk.transpose(0, 2, 1, 3, 4, 5),
                    True,
                )
                run_fused(
                    mixed_spin_v,
                    eris_ovov[:, c0:c1, :, b0:b1],
                    t1b[:, a0:a1],
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                    t2aa[:, :, c0:c1, b0:b1],
                    fVO[a0:a1, :],
                    t2ab[:, :, b0:b1, a0:a1],
                    fvo[c0:c1, :],
                    v_blk.transpose(0, 2, 1, 3, 5, 4),
                    False,
                )
                run_fused(
                    mixed_spin_v,
                    eris_ovov[:, c0:c1, :, b0:b1],
                    t1b[:, a0:a1],
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                    t2aa[:, :, c0:c1, b0:b1],
                    fVO[a0:a1, :],
                    t2ab[:, :, b0:b1, a0:a1],
                    fvo[c0:c1, :],
                    v_blk.transpose(0, 1, 2, 3, 5, 4),
                    True,
                )

                rw = mixed_spin_divide(
                    w_blk, None, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                wvd = mixed_spin_divide(
                    w_blk, v_blk, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                dovvv[:, c0:c1, :, b0:b1] += (
                    parallel_einsum("jiea,ijkabc->kceb", t2ab[:, :, :, a0:a1], wvd)
                    * 0.25
                )
                dovVV[:, c0:c1, :, a0:a1] += (
                    parallel_einsum("jibe,ijkabc->kcea", t2ab[:, :, b0:b1, :], wvd)
                    * 0.25
                )
                dOVvv[:, a0:a1, :, c0:c1] += (
                    parallel_einsum("jkbe,ijkabc->iaec", t2aa[:, :, b0:b1, :], wvd)
                    * 0.125
                )
                dooov[:, :, :, c0:c1] -= (
                    parallel_einsum("miba,ijkabc->jmkc", t2ab[:, :, b0:b1, a0:a1], wvd)
                    * 0.25
                )
                dOOov[:, :, :, c0:c1] -= (
                    parallel_einsum("jmba,ijkabc->imkc", t2ab[:, :, b0:b1, a0:a1], wvd)
                    * 0.25
                )
                dooOV[:, :, :, a0:a1] -= (
                    parallel_einsum("jmbc,ijkabc->kmia", t2aa[:, :, b0:b1, c0:c1], wvd)
                    * 0.125
                )
                dovov[:, b0:b1, :, c0:c1] += (
                    parallel_einsum("ia,ijkabc->jbkc", t1b[:, a0:a1], rw) * 0.25
                )
                # dOVov[:, a0:a1, :, c0:c1] += parallel_einsum('jb,ijkabc->iakc', t1a, rw) * .25
                dovOV[:, c0:c1, :, a0:a1] += (
                    parallel_einsum("jb,ijkabc->kcia", t1a[:, b0:b1], rw) * 0.25
                )

    # bba
    blksize = triples_block_size(
        max_memory, nvirb, noccb**2 * nocca, 1 / 3
    )
    blksize = max(1, min(nvira, blksize))
    if blksize < nvira or blksize < nvirb:
        blksize = min(blksize, (nvira + 1) // 2)
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm _gamma2_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for a0, a1 in lib.prange(0, nvira if has_mixed else 0, blksize):
        for b0, b1 in lib.prange(0, nvirb, blksize):
            for c0, c1 in lib.prange(0, nvirb, blksize):
                w_blk = w_einsum(
                    "ijae,kceb->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, c0:c1, :, b0:b1],
                    scale=2,
                )
                w_blk = w_einsum(
                    "ijeb,kcea->ijkabc",
                    t2ab[:, :, :, b0:b1],
                    eris_OVvv[:, c0:c1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                    out=w_blk,
                )
                w_blk = w_einsum(
                    "imab,kcjm->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVOO[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "mjab,kcim->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVoo[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )
                v_blk = run_fused(
                    mixed_spin_v,
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                    eris_ovOV[:, a0:a1, :, c0:c1].transpose(2, 3, 0, 1),
                    t1b[:, b0:b1],
                    t2bb[:, :, b0:b1, c0:c1],
                    fvo[a0:a1, :],
                    t2ab[:, :, a0:a1, c0:c1].transpose(1, 0, 3, 2),
                    fVO[b0:b1, :],
                )

                w_blk = w_einsum(
                    "ikae,jceb->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, c0:c1, :, b0:b1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "ikeb,jcea->ijkabc",
                    t2ab[:, :, :, b0:b1],
                    eris_OVvv[:, c0:c1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                    out=w_blk, subtract=True,
                )
                w_blk = w_einsum(
                    "imab,jckm->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVOO[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "mkab,jcim->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVoo[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk,
                )

                w_blk = w_einsum(
                    "ikae,jbec->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, b0:b1, :, c0:c1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "ikec,jbea->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVvv[:, b0:b1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                    out=w_blk,
                )
                w_blk = w_einsum(
                    "imac,jbkm->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVOO[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "mkac,jbim->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVoo[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )

                w_blk = w_einsum(
                    "ijae,kbec->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, b0:b1, :, c0:c1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "ijec,kbea->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVvv[:, b0:b1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = w_einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                    out=w_blk, subtract=True,
                )
                w_blk = w_einsum(
                    "imac,kbjm->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVOO[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "mjac,kbim->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVoo[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = w_einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk,
                )

                run_fused(
                    mixed_spin_v,
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                    eris_ovOV[:, a0:a1, :, c0:c1].transpose(2, 3, 0, 1),
                    t1b[:, b0:b1],
                    t2bb[:, :, b0:b1, c0:c1],
                    fvo[a0:a1, :],
                    t2ab[:, :, a0:a1, c0:c1].transpose(1, 0, 3, 2),
                    fVO[b0:b1, :],
                    v_blk.transpose(0, 2, 1, 3, 4, 5),
                    True,
                )
                run_fused(
                    mixed_spin_v,
                    eris_OVOV[:, c0:c1, :, b0:b1],
                    t1a[:, a0:a1],
                    eris_ovOV[:, a0:a1, :, b0:b1].transpose(2, 3, 0, 1),
                    t1b[:, c0:c1],
                    t2bb[:, :, c0:c1, b0:b1],
                    fvo[a0:a1, :],
                    t2ab[:, :, a0:a1, b0:b1].transpose(1, 0, 3, 2),
                    fVO[c0:c1, :],
                    v_blk.transpose(0, 2, 1, 3, 5, 4),
                    False,
                )
                run_fused(
                    mixed_spin_v,
                    eris_OVOV[:, c0:c1, :, b0:b1],
                    t1a[:, a0:a1],
                    eris_ovOV[:, a0:a1, :, b0:b1].transpose(2, 3, 0, 1),
                    t1b[:, c0:c1],
                    t2bb[:, :, c0:c1, b0:b1],
                    fvo[a0:a1, :],
                    t2ab[:, :, a0:a1, b0:b1].transpose(1, 0, 3, 2),
                    fVO[c0:c1, :],
                    v_blk.transpose(0, 1, 2, 3, 5, 4),
                    True,
                )

                rw = mixed_spin_divide(
                    w_blk, None, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                wvd = mixed_spin_divide(
                    w_blk, v_blk, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                dOVVV[:, c0:c1, :, b0:b1] += (
                    parallel_einsum("ijae,ijkabc->kceb", t2ab[:, :, a0:a1, :], wvd)
                    * 0.25
                )
                dOVvv[:, c0:c1, :, a0:a1] += (
                    parallel_einsum("ijeb,ijkabc->kcea", t2ab[:, :, :, b0:b1], wvd)
                    * 0.25
                )
                dovVV[:, a0:a1, :, c0:c1] += (
                    parallel_einsum("jkbe,ijkabc->iaec", t2bb[:, :, b0:b1, :], wvd)
                    * 0.125
                )
                dOOOV[:, :, :, c0:c1] -= (
                    parallel_einsum("imab,ijkabc->jmkc", t2ab[:, :, a0:a1, b0:b1], wvd)
                    * 0.25
                )
                dooOV[:, :, :, c0:c1] -= (
                    parallel_einsum("mjab,ijkabc->imkc", t2ab[:, :, a0:a1, b0:b1], wvd)
                    * 0.25
                )
                dOOov[:, :, :, a0:a1] -= (
                    parallel_einsum("jmbc,ijkabc->kmia", t2bb[:, :, b0:b1, c0:c1], wvd)
                    * 0.125
                )
                dOVOV[:, b0:b1, :, c0:c1] += (
                    parallel_einsum("ia,ijkabc->jbkc", t1a[:, a0:a1], rw) * 0.25
                )
                dovOV[:, a0:a1, :, c0:c1] += (
                    parallel_einsum("jb,ijkabc->iakc", t1b[:, b0:b1], rw) * 0.25
                )
                # dOVov += parallel_einsum('jb,ijkabc->kcia', t1b, rw) * .25

    if compress_vvvv:
        nmoa, nmob = mycc.nmo
        nocca, noccb, nvira, nvirb = t2ab.shape
        idxa = numpy.tril_indices(nvira)
        idxa = idxa[0] * nvira + idxa[1]
        idxb = numpy.tril_indices(nvirb)
        idxb = idxb[0] * nvirb + idxb[1]
        dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
        dvvvv = lib.take_2d(dvvvv.reshape(nvira**2, nvira**2), idxa, idxa)
        dvvvv *= 0.5
        dvvVV = dvvVV + dvvVV.transpose(1, 0, 2, 3)
        dvvVV = lib.take_2d(dvvVV.reshape(nvira**2, nvirb**2), idxa, idxb)
        dVVVV = dVVVV + dVVVV.transpose(1, 0, 2, 3)
        dVVVV = lib.take_2d(dVVVV.reshape(nvirb**2, nvirb**2), idxb, idxb)
        dVVVV *= 0.5

        d2 = (
            (dovov, dovOV, dOVov, dOVOV),
            (dvvvv, dvvVV, dVVvv, dVVVV),
            (doooo, dooOO, dOOoo, dOOOO),
            (doovv, dooVV, dOOvv, dOOVV),
            (dovvo, dovVO, dOVvo, dOVVO),
            (dvvov, dvvOV, dVVov, dVVOV),
            (dovvv, dovVV, dOVvv, dOVVV),
            (dooov, dooOV, dOOov, dOOOV),
        )

    return d2


def r6(w):
    return (
        w
        + w.transpose(2, 0, 1, 3, 4, 5)
        + w.transpose(1, 2, 0, 3, 4, 5)
        - w.transpose(2, 1, 0, 3, 4, 5)
        - w.transpose(0, 2, 1, 3, 4, 5)
        - w.transpose(1, 0, 2, 3, 4, 5)
    )
