import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm
from pyscf.cc import ccsd_lambda
from pyscf.cc import uccsd_lambda


def parallel_einsum(subscripts, *tensors):
    return numpy.einsum(subscripts, *tensors, optimize="optimal")


def u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None, compress_vvvv=False):
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
    blksize = min(
        nvira, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocca**3 * nvira)) ** (1 / 2))
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

    time2 = logger.process_clock(), logger.perf_counter()
    for b0, b1 in lib.prange(0, nvira, blksize):
        for c0, c1 in lib.prange(0, nvira, blksize):
            w_blk = parallel_einsum(
                "ijae,kceb->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, c0:c1, :, b0:b1],
            )
            w_blk += parallel_einsum(
                "kice,jbea->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, b0:b1, :, :],
            )
            w_blk += parallel_einsum(
                "jkbe,iaec->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, :, :, c0:c1],
            )
            w_blk += parallel_einsum(
                "ikae,jbec->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, b0:b1, :, c0:c1],
            )
            w_blk += parallel_einsum(
                "kjce,iaeb->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, :, :, b0:b1],
            )
            w_blk += parallel_einsum(
                "jibe,kcea->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, c0:c1, :, :],
            )
            w_blk -= parallel_einsum(
                "mkbc,iajm->ijkabc",
                t2aa[:, :, b0:b1, c0:c1],
                eris_ovoo[:, :, :, :],
            )
            w_blk -= parallel_einsum(
                "mjab,kcim->ijkabc",
                t2aa[:, :, :, b0:b1],
                eris_ovoo[:, c0:c1, :, :],
            )
            w_blk -= parallel_einsum(
                "mica,jbkm->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovoo[:, b0:b1, :, :],
            )
            w_blk -= parallel_einsum(
                "mjcb,iakm->ijkabc",
                t2aa[:, :, c0:c1, b0:b1],
                eris_ovoo[:, :, :, :],
            )
            w_blk -= parallel_einsum(
                "miba,kcjm->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovoo[:, c0:c1, :, :],
            )
            w_blk -= parallel_einsum(
                "mkac,jbim->ijkabc",
                t2aa[:, :, :, c0:c1],
                eris_ovoo[:, b0:b1, :, :],
            )
            v_blk = parallel_einsum(
                "jbkc,ia->ijkabc",
                eris_ovov[:, b0:b1, :, c0:c1],
                t1a[:, :],
            )
            v_blk += parallel_einsum(
                "iajb,kc->ijkabc",
                eris_ovov[:, :, :, b0:b1],
                t1a[:, c0:c1],
            )
            v_blk += parallel_einsum(
                "kcia,jb->ijkabc",
                eris_ovov[:, c0:c1, :, :],
                t1a[:, b0:b1],
            )
            v_blk += parallel_einsum(
                "kcjb,ia->ijkabc",
                eris_ovov[:, c0:c1, :, b0:b1],
                t1a[:, :],
            )
            v_blk += parallel_einsum(
                "jbia,kc->ijkabc",
                eris_ovov[:, b0:b1, :, :],
                t1a[:, c0:c1],
            )
            v_blk += parallel_einsum(
                "iakc,jb->ijkabc",
                eris_ovov[:, :, :, c0:c1],
                t1a[:, b0:b1],
            )
            v_blk += (
                parallel_einsum(
                    "jkbc,ai->ijkabc",
                    t2aa[:, :, b0:b1, c0:c1],
                    fvo[:, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "ijab,ck->ijkabc",
                    t2aa[:, :, :, b0:b1],
                    fvo[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "kica,bj->ijkabc",
                    t2aa[:, :, c0:c1, :],
                    fvo[b0:b1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "kjcb,ai->ijkabc",
                    t2aa[:, :, c0:c1, b0:b1],
                    fvo[:, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "jiba,ck->ijkabc",
                    t2aa[:, :, b0:b1, :],
                    fvo[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "ikac,bj->ijkabc",
                    t2aa[:, :, :, c0:c1],
                    fvo[b0:b1, :],
                )
                * 0.5
            )

            d3 = lib.direct_sum(
                "ia+jb+kc->ijkabc", eia[:, :], eia[:, b0:b1], eia[:, c0:c1]
            )
            rw = r6(w_blk) / d3
            wvd = r6(w_blk * 2 + v_blk) / d3

            dovov[:, b0:b1, :, c0:c1] += parallel_einsum("ia,ijkabc->jbkc", t1a, rw) * 0.25
            # *(1/8) instead of (1/4) because ooov appears 4 times in the 2pdm tensor due
            # to symmetrization, and its contribution is scaled by 1/2 in Tr(H,2pdm)
            dooov -= (
                parallel_einsum("mkbc,ijkabc->jmia", t2aa[:, :, b0:b1, c0:c1], wvd) * 0.125
            )
            dovvv[:, :, :, b0:b1] += (
                parallel_einsum("kjcf,ijkabc->iafb", t2aa[:, :, c0:c1, :], wvd) * 0.125
            )
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma2_intermediates pass1 [%d:%d]" % (b0, b1), *time2
        )

    # bbb
    blksize = min(
        nvirb, int(((max_memory * 0.9e6 / 8) / 6.0 / (noccb**3 * nvirb)) ** (1 / 2))
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
    time2 = logger.process_clock(), logger.perf_counter()
    for b0, b1 in lib.prange(0, nvirb, blksize):
        for c0, c1 in lib.prange(0, nvirb, blksize):
            w_blk = parallel_einsum(
                "ijae,kceb->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, c0:c1, :, b0:b1],
            )
            w_blk += parallel_einsum(
                "kice,jbea->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, b0:b1, :, :],
            )
            w_blk += parallel_einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, :, :, c0:c1],
            )
            w_blk += parallel_einsum(
                "ikae,jbec->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, b0:b1, :, c0:c1],
            )
            w_blk += parallel_einsum(
                "kjce,iaeb->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, :, :, b0:b1],
            )
            w_blk += parallel_einsum(
                "jibe,kcea->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, c0:c1, :, :],
            )
            w_blk -= parallel_einsum(
                "imab,kcjm->ijkabc",
                t2bb[:, :, :, b0:b1],
                eris_OVOO[:, c0:c1, :, :],
            )
            w_blk -= parallel_einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, b0:b1, c0:c1],
                eris_OVOO[:, :, :, :],
            )
            w_blk -= parallel_einsum(
                "kmca,jbim->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVOO[:, b0:b1, :, :],
            )
            w_blk -= parallel_einsum(
                "imac,jbkm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_OVOO[:, b0:b1, :, :],
            )
            w_blk -= parallel_einsum(
                "kmcb,iajm->ijkabc",
                t2bb[:, :, c0:c1, b0:b1],
                eris_OVOO[:, :, :, :],
            )
            w_blk -= parallel_einsum(
                "jmba,kcim->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVOO[:, c0:c1, :, :],
            )
            v_blk = parallel_einsum(
                "jbkc,ia->ijkabc",
                eris_OVOV[:, b0:b1, :, c0:c1],
                t1b[:, :],
            )
            v_blk += parallel_einsum(
                "iajb,kc->ijkabc",
                eris_OVOV[:, :, :, b0:b1],
                t1b[:, c0:c1],
            )
            v_blk += parallel_einsum(
                "kcia,jb->ijkabc",
                eris_OVOV[:, c0:c1, :, :],
                t1b[:, b0:b1],
            )
            v_blk += parallel_einsum(
                "kcjb,ia->ijkabc",
                eris_OVOV[:, c0:c1, :, b0:b1],
                t1b[:, :],
            )
            v_blk += parallel_einsum(
                "jbia,kc->ijkabc",
                eris_OVOV[:, b0:b1, :, :],
                t1b[:, c0:c1],
            )
            v_blk += parallel_einsum(
                "iakc,jb->ijkabc",
                eris_OVOV[:, :, :, c0:c1],
                t1b[:, b0:b1],
            )
            v_blk += (
                parallel_einsum(
                    "jkbc,ai->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    fVO[:, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "ijab,ck->ijkabc",
                    t2bb[:, :, :, b0:b1],
                    fVO[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "kica,bj->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    fVO[b0:b1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "kjcb,ai->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    fVO[:, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "jiba,ck->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    fVO[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                parallel_einsum(
                    "ikac,bj->ijkabc",
                    t2bb[:, :, :, c0:c1],
                    fVO[b0:b1, :],
                )
                * 0.5
            )

            d3 = lib.direct_sum(
                "ia+jb+kc->ijkabc", eIA[:, :], eIA[:, b0:b1], eIA[:, c0:c1]
            )
            rw = r6(w_blk) / d3
            wvd = r6(w_blk * 2 + v_blk) / d3

            dOVOV[:, b0:b1, :, c0:c1] += parallel_einsum("ia,ijkabc->jbkc", t1b, rw) * 0.25
            dOOOV -= (
                parallel_einsum("mkbc,ijkabc->jmia", t2bb[:, :, b0:b1, c0:c1], wvd) * 0.125
            )
            dOVVV[:, :, :, b0:b1] += (
                parallel_einsum("kjcf,ijkabc->iafb", t2bb[:, :, c0:c1, :], wvd) * 0.125
            )
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma_intermediates pass2 [%d:%d]" % (b0, b1), *time2
        )

    # baa
    blksize = min(
        nvirb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocca**2 * noccb)) ** (1 / 3))
    )
    blksize = min(nvira, blksize)
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
    time2 = logger.process_clock(), logger.perf_counter()
    for a0, a1 in lib.prange(0, nvirb, blksize):
        for b0, b1 in lib.prange(0, nvira, blksize):
            for c0, c1 in lib.prange(0, nvira, blksize):
                w_blk = (
                    parallel_einsum(
                        "jIeA,kceb->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "jIbE,kcEA->IjkAbc",
                        t2ab[:, :, b0:b1, :],
                        eris_ovVV[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                )
                w_blk -= (
                    parallel_einsum(
                        "mIbA,kcjm->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "jMbA,kcIM->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                )
                v_blk = parallel_einsum(
                    "jbkc,IA->IjkAbc",
                    eris_ovov[:, b0:b1, :, c0:c1],
                    t1b[:, a0:a1],
                )
                v_blk += parallel_einsum(
                    "kcIA,jb->IjkAbc",
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                )
                v_blk += parallel_einsum(
                    "kcIA,jb->IjkAbc",
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                )
                v_blk += (
                    parallel_einsum(
                        "jkbc,AI->IjkAbc",
                        t2aa[:, :, b0:b1, c0:c1],
                        fVO[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk += (
                    parallel_einsum(
                        "kIcA,bj->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        fvo[b0:b1, :],
                    )
                    * 2
                )

                w_blk -= (
                    parallel_einsum(
                        "kIeA,jceb->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "kIbE,jcEA->IjkAbc",
                        t2ab[:, :, b0:b1, :],
                        eris_ovVV[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                )
                w_blk += (
                    parallel_einsum(
                        "mIbA,jckm->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "kMbA,jcIM->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                w_blk += (
                    parallel_einsum(
                        "kIeA,jbec->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "kIcE,jbEA->IjkAbc",
                        t2ab[:, :, c0:c1, :],
                        eris_ovVV[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                )
                w_blk -= (
                    parallel_einsum(
                        "mIcA,jbkm->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "kMcA,jbIM->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                w_blk -= (
                    parallel_einsum(
                        "jIeA,kbec->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "jIcE,kbEA->IjkAbc",
                        t2ab[:, :, c0:c1, :],
                        eris_ovVV[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                )
                w_blk += (
                    parallel_einsum(
                        "mIcA,kbjm->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "jMcA,kbIM->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                v_blk -= parallel_einsum(
                    "kbjc,IA->IjkAbc",
                    eris_ovov[:, b0:b1, :, c0:c1],
                    t1b[:, a0:a1],
                )
                v_blk -= parallel_einsum(
                    "jcIA,kb->IjkAbc",
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                )
                v_blk -= parallel_einsum(
                    "jcIA,kb->IjkAbc",
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                )
                v_blk -= (
                    parallel_einsum(
                        "kjbc,AI->IjkAbc",
                        t2aa[:, :, b0:b1, c0:c1],
                        fVO[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk -= (
                    parallel_einsum(
                        "jIcA,bk->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        fvo[b0:b1, :],
                    )
                    * 2
                )

                v_blk += parallel_einsum(
                    "kcjb,IA->IjkAbc",
                    eris_ovov[:, c0:c1, :, b0:b1],
                    t1b[:, a0:a1],
                )
                v_blk += parallel_einsum(
                    "jbIA,kc->IjkAbc",
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                )
                v_blk += parallel_einsum(
                    "jbIA,kc->IjkAbc",
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                )
                v_blk += (
                    parallel_einsum(
                        "kjcb,AI->IjkAbc",
                        t2aa[:, :, c0:c1, b0:b1],
                        fVO[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk += (
                    parallel_einsum(
                        "jIbA,ck->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        fvo[c0:c1, :],
                    )
                    * 2
                )

                v_blk -= parallel_einsum(
                    "jckb,IA->IjkAbc",
                    eris_ovov[:, c0:c1, :, b0:b1],
                    t1b[:, a0:a1],
                )
                v_blk -= parallel_einsum(
                    "kbIA,jc->IjkAbc",
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                )
                v_blk -= parallel_einsum(
                    "kbIA,jc->IjkAbc",
                    eris_ovOV[:, b0:b1, :, a0:a1],
                    t1a[:, c0:c1],
                )
                v_blk -= (
                    parallel_einsum(
                        "jkcb,AI->IjkAbc",
                        t2aa[:, :, c0:c1, b0:b1],
                        fVO[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk -= (
                    parallel_einsum(
                        "kIbA,cj->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        fvo[c0:c1, :],
                    )
                    * 2
                )

                d3 = lib.direct_sum(
                    "IA+jb+kc->IjkAbc", eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                rw = w_blk / d3
                wvd = (w_blk * 2 + v_blk) / d3
                dovvv[:, c0:c1, :, b0:b1] += (
                    parallel_einsum("jiea,ijkabc->kceb", t2ab[:, :, :, a0:a1], wvd) * 0.25
                )
                dovVV[:, c0:c1, :, a0:a1] += (
                    parallel_einsum("jibe,ijkabc->kcea", t2ab[:, :, b0:b1, :], wvd) * 0.25
                )
                dOVvv[:, a0:a1, :, c0:c1] += (
                    parallel_einsum("jkbe,ijkabc->iaec", t2aa[:, :, b0:b1, :], wvd) * 0.125
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
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma2_intermediates pass3 [%d:%d]" % (a0, a1), *time2
        )

    # bba
    blksize = min(
        nvirb, int(((max_memory * 0.9e6 / 8) / 6.0 / (noccb**2 * nocca)) ** (1 / 3))
    )
    blksize = min(nvira, blksize)
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
    time2 = logger.process_clock(), logger.perf_counter()
    for a0, a1 in lib.prange(0, nvira, blksize):
        for b0, b1 in lib.prange(0, nvirb, blksize):
            for c0, c1 in lib.prange(0, nvirb, blksize):
                w_blk = (
                    parallel_einsum(
                        "ijae,kceb->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "ijeb,kcea->ijkabc",
                        t2ab[:, :, :, b0:b1],
                        eris_OVvv[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                )
                w_blk -= (
                    parallel_einsum(
                        "imab,kcjm->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "mjab,kcim->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                )
                v_blk = parallel_einsum(
                    "jbkc,ia->ijkabc",
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                )
                v_blk += parallel_einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk += parallel_einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk += (
                    parallel_einsum(
                        "JKBC,ai->iJKaBC",
                        t2bb[:, :, b0:b1, c0:c1],
                        fvo[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk += (
                    parallel_einsum(
                        "iKaC,BJ->iJKaBC",
                        t2ab[:, :, a0:a1, c0:c1],
                        fVO[b0:b1, :],
                    )
                    * 2
                )

                w_blk -= (
                    parallel_einsum(
                        "ikae,jceb->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "ikeb,jcea->ijkabc",
                        t2ab[:, :, :, b0:b1],
                        eris_OVvv[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                )
                w_blk += (
                    parallel_einsum(
                        "imab,jckm->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "mkab,jcim->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                )

                w_blk += (
                    parallel_einsum(
                        "ikae,jbec->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "ikec,jbea->ijkabc",
                        t2ab[:, :, :, c0:c1],
                        eris_OVvv[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                )
                w_blk -= (
                    parallel_einsum(
                        "imac,jbkm->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "mkac,jbim->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                )

                w_blk -= (
                    parallel_einsum(
                        "ijae,kbec->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk -= (
                    parallel_einsum(
                        "ijec,kbea->ijkabc",
                        t2ab[:, :, :, c0:c1],
                        eris_OVvv[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= parallel_einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                )
                w_blk += (
                    parallel_einsum(
                        "imac,kbjm->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    parallel_einsum(
                        "mjac,kbim->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += parallel_einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                )

                v_blk -= parallel_einsum(
                    "kbjc,ia->ijkabc",
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                )
                v_blk -= parallel_einsum(
                    "iajc,kb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk -= parallel_einsum(
                    "iajc,kb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk -= (
                    parallel_einsum(
                        "KJBC,ai->iJKaBC",
                        t2bb[:, :, b0:b1, c0:c1],
                        fvo[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk -= (
                    parallel_einsum(
                        "iJaC,BK->iJKaBC",
                        t2ab[:, :, a0:a1, c0:c1],
                        fVO[b0:b1, :],
                    )
                    * 2
                )

                v_blk += parallel_einsum(
                    "kcjb,ia->ijkabc",
                    eris_OVOV[:, c0:c1, :, b0:b1],
                    t1a[:, a0:a1],
                )
                v_blk += parallel_einsum(
                    "iajb,kc->ijkabc",
                    eris_ovOV[:, a0:a1, :, b0:b1],
                    t1b[:, c0:c1],
                )
                v_blk += parallel_einsum(
                    "iajb,kc->ijkabc",
                    eris_ovOV[:, a0:a1, :, b0:b1],
                    t1b[:, c0:c1],
                )
                v_blk += (
                    parallel_einsum(
                        "KJCB,ai->iJKaBC",
                        t2bb[:, :, c0:c1, b0:b1],
                        fvo[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk += (
                    parallel_einsum(
                        "iJaB,CK->iJKaBC",
                        t2ab[:, :, a0:a1, b0:b1],
                        fVO[c0:c1, :],
                    )
                    * 2
                )

                v_blk -= parallel_einsum(
                    "jckb,ia->ijkabc",
                    eris_OVOV[:, c0:c1, :, b0:b1],
                    t1a[:, a0:a1],
                )
                v_blk -= parallel_einsum(
                    "iakb,jc->ijkabc",
                    eris_ovOV[:, a0:a1, :, b0:b1],
                    t1b[:, c0:c1],
                )
                v_blk -= parallel_einsum(
                    "iakb,jc->ijkabc",
                    eris_ovOV[:, a0:a1, :, b0:b1],
                    t1b[:, c0:c1],
                )
                v_blk -= (
                    parallel_einsum(
                        "JKCB,ai->iJKaBC",
                        t2bb[:, :, c0:c1, b0:b1],
                        fvo[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk -= (
                    parallel_einsum(
                        "iKaB,CJ->iJKaBC",
                        t2ab[:, :, a0:a1, b0:b1],
                        fVO[c0:c1, :],
                    )
                    * 2
                )

                d3 = lib.direct_sum(
                    "ia+jb+kc->ijkabc", eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                rw = w_blk / d3
                wvd = (w_blk * 2 + v_blk) / d3
                dOVVV[:, c0:c1, :, b0:b1] += (
                    parallel_einsum("ijae,ijkabc->kceb", t2ab[:, :, a0:a1, :], wvd) * 0.25
                )
                dOVvv[:, c0:c1, :, a0:a1] += (
                    parallel_einsum("ijeb,ijkabc->kcea", t2ab[:, :, :, b0:b1], wvd) * 0.25
                )
                dovVV[:, a0:a1, :, c0:c1] += (
                    parallel_einsum("jkbe,ijkabc->iaec", t2bb[:, :, b0:b1, :], wvd) * 0.125
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
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma2_intermediates pass4 [%d:%d]" % (a0, a1), *time2
        )

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
