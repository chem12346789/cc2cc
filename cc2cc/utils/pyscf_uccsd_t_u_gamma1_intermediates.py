import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm
from pyscf.cc import ccsd_lambda
from pyscf.cc import uccsd_lambda


def u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None, for_grad=False):
    log = logger.Logger(mycc.stdout, mycc.verbose)
    d1 = uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None:
        eris = mycc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
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

    goo = numpy.zeros((nocca, nocca), dtype=t1a.dtype)
    gvv = numpy.zeros((nvira, nvira), dtype=t1a.dtype)
    gvo = numpy.zeros((nvira, nocca), dtype=t1a.dtype)
    gOO = numpy.zeros((noccb, noccb), dtype=t1b.dtype)
    gVV = numpy.zeros((nvirb, nvirb), dtype=t1b.dtype)
    gVO = numpy.zeros((nvirb, noccb), dtype=t1b.dtype)

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
        "ccsd_t spin 1 rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocca,
        nvira,
        blksize,
    )

    time2 = logger.process_clock(), logger.perf_counter()
    for b0, b1 in lib.prange(0, nvira, blksize):
        for c0, c1 in lib.prange(0, nvira, blksize):
            w_blk = numpy.einsum(
                "ijae,kceb->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, c0:c1, :, b0:b1],
            )
            w_blk += numpy.einsum(
                "kice,jbea->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, b0:b1, :, :],
            )
            w_blk += numpy.einsum(
                "jkbe,iaec->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, :, :, c0:c1],
            )
            w_blk += numpy.einsum(
                "ikae,jbec->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, b0:b1, :, c0:c1],
            )
            w_blk += numpy.einsum(
                "kjce,iaeb->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, :, :, b0:b1],
            )
            w_blk += numpy.einsum(
                "jibe,kcea->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, c0:c1, :, :],
            )
            w_blk -= numpy.einsum(
                "mkbc,iajm->ijkabc",
                t2aa[:, :, b0:b1, c0:c1],
                eris_ovoo[:, :, :, :],
            )
            w_blk -= numpy.einsum(
                "mjab,kcim->ijkabc",
                t2aa[:, :, :, b0:b1],
                eris_ovoo[:, c0:c1, :, :],
            )
            w_blk -= numpy.einsum(
                "mica,jbkm->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovoo[:, b0:b1, :, :],
            )
            w_blk -= numpy.einsum(
                "mjcb,iakm->ijkabc",
                t2aa[:, :, c0:c1, b0:b1],
                eris_ovoo[:, :, :, :],
            )
            w_blk -= numpy.einsum(
                "miba,kcjm->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovoo[:, c0:c1, :, :],
            )
            w_blk -= numpy.einsum(
                "mkac,jbim->ijkabc",
                t2aa[:, :, :, c0:c1],
                eris_ovoo[:, b0:b1, :, :],
            )
            v_blk = numpy.einsum(
                "jbkc,ia->ijkabc",
                eris_ovov[:, b0:b1, :, c0:c1],
                t1a[:, :],
            )
            v_blk += numpy.einsum(
                "iajb,kc->ijkabc",
                eris_ovov[:, :, :, b0:b1],
                t1a[:, c0:c1],
            )
            v_blk += numpy.einsum(
                "kcia,jb->ijkabc",
                eris_ovov[:, c0:c1, :, :],
                t1a[:, b0:b1],
            )
            v_blk += numpy.einsum(
                "kcjb,ia->ijkabc",
                eris_ovov[:, c0:c1, :, b0:b1],
                t1a[:, :],
            )
            v_blk += numpy.einsum(
                "jbia,kc->ijkabc",
                eris_ovov[:, b0:b1, :, :],
                t1a[:, c0:c1],
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_ovov[:, :, :, c0:c1],
                t1a[:, b0:b1],
            )
            v_blk += (
                numpy.einsum(
                    "jkbc,ai->ijkabc",
                    t2aa[:, :, b0:b1, c0:c1],
                    fvo[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "ijab,ck->ijkabc",
                    t2aa[:, :, :, b0:b1],
                    fvo[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "kica,bj->ijkabc",
                    t2aa[:, :, c0:c1, :],
                    fvo[b0:b1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "kjcb,ai->ijkabc",
                    t2aa[:, :, c0:c1, b0:b1],
                    fvo[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "jiba,ck->ijkabc",
                    t2aa[:, :, b0:b1, :],
                    fvo[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
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
            wvd = (w_blk + v_blk) / d3

            goo += numpy.einsum("iklabc,jklabc->ij", wvd, rw) * 0.125
            gvv += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.125
            gvo += numpy.einsum("jkbc,ijkabc->ai", t2aa[:, :, b0:b1, c0:c1], rw) * 0.125
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma1_intermediates pass2 [%d:%d]" % (b0, b1), *time2
        )

    # bbb
    blksize = min(
        nvirb, int(((max_memory * 0.9e6 / 8) / 6.0 / (noccb**3 * nvirb)) ** (1 / 2))
    )
    if blksize < nvirb:
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    time2 = logger.process_clock(), logger.perf_counter()
    for b0, b1 in lib.prange(0, nvirb, blksize):
        for c0, c1 in lib.prange(0, nvirb, blksize):
            w_blk = numpy.einsum(
                "ijae,kceb->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, c0:c1, :, b0:b1],
            )
            w_blk += numpy.einsum(
                "kice,jbea->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, b0:b1, :, :],
            )
            w_blk += numpy.einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, :, :, c0:c1],
            )
            w_blk += numpy.einsum(
                "ikae,jbec->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, b0:b1, :, c0:c1],
            )
            w_blk += numpy.einsum(
                "kjce,iaeb->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, :, :, b0:b1],
            )
            w_blk += numpy.einsum(
                "jibe,kcea->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, c0:c1, :, :],
            )
            w_blk -= numpy.einsum(
                "imab,kcjm->ijkabc",
                t2bb[:, :, :, b0:b1],
                eris_OVOO[:, c0:c1, :, :],
            )
            w_blk -= numpy.einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, b0:b1, c0:c1],
                eris_OVOO[:, :, :, :],
            )
            w_blk -= numpy.einsum(
                "kmca,jbim->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVOO[:, b0:b1, :, :],
            )
            w_blk -= numpy.einsum(
                "imac,jbkm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_OVOO[:, b0:b1, :, :],
            )
            w_blk -= numpy.einsum(
                "kmcb,iajm->ijkabc",
                t2bb[:, :, c0:c1, b0:b1],
                eris_OVOO[:, :, :, :],
            )
            w_blk -= numpy.einsum(
                "jmba,kcim->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVOO[:, c0:c1, :, :],
            )
            v_blk = numpy.einsum(
                "jbkc,ia->ijkabc",
                eris_OVOV[:, b0:b1, :, c0:c1],
                t1b[:, :],
            )
            v_blk += numpy.einsum(
                "iajb,kc->ijkabc",
                eris_OVOV[:, :, :, b0:b1],
                t1b[:, c0:c1],
            )
            v_blk += numpy.einsum(
                "kcia,jb->ijkabc",
                eris_OVOV[:, c0:c1, :, :],
                t1b[:, b0:b1],
            )
            v_blk += numpy.einsum(
                "kcjb,ia->ijkabc",
                eris_OVOV[:, c0:c1, :, b0:b1],
                t1b[:, :],
            )
            v_blk += numpy.einsum(
                "jbia,kc->ijkabc",
                eris_OVOV[:, b0:b1, :, :],
                t1b[:, c0:c1],
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_OVOV[:, :, :, c0:c1],
                t1b[:, b0:b1],
            )
            v_blk += (
                numpy.einsum(
                    "jkbc,ai->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    fVO[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "ijab,ck->ijkabc",
                    t2bb[:, :, :, b0:b1],
                    fVO[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "kica,bj->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    fVO[b0:b1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "kjcb,ai->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    fVO[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "jiba,ck->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    fVO[c0:c1, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
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
            wvd = (w_blk + v_blk) / d3

            gOO += numpy.einsum("iklabc,jklabc->ij", wvd, rw) * 0.125
            gVV += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.125
            gVO += numpy.einsum("jkbc,ijkabc->ai", t2bb[:, :, b0:b1, c0:c1], rw) * 0.125
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma1_intermediates pass2 [%d:%d]" % (b0, b1), *time2
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
        "ccsd_t spin 2 rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
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
                    numpy.einsum(
                        "jIeA,kceb->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "jIbE,kcEA->IjkAbc",
                        t2ab[:, :, b0:b1, :],
                        eris_ovVV[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                )
                w_blk -= (
                    numpy.einsum(
                        "mIbA,kcjm->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "jMbA,kcIM->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                )
                v_blk = numpy.einsum(
                    "jbkc,IA->IjkAbc", eris_ovov[:, b0:b1, :, c0:c1], t1b[:, a0:a1]
                )
                v_blk += numpy.einsum(
                    "kcIA,jb->IjkAbc", eris_ovOV[:, c0:c1, :, a0:a1], t1a[:, b0:b1]
                )
                v_blk += numpy.einsum(
                    "kcIA,jb->IjkAbc", eris_ovOV[:, c0:c1, :, a0:a1], t1a[:, b0:b1]
                )
                v_blk += (
                    numpy.einsum(
                        "jkbc,AI->IjkAbc", t2aa[:, :, b0:b1, c0:c1], fVO[a0:a1, :]
                    )
                    * 0.5
                )
                v_blk += (
                    numpy.einsum(
                        "kIcA,bj->IjkAbc", t2ab[:, :, c0:c1, a0:a1], fvo[b0:b1, :]
                    )
                    * 2
                )

                d3 = lib.direct_sum(
                    "IA+jb+kc->IjkAbc", eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                wvd = (w_blk + v_blk) / d3

                w_blk -= (
                    numpy.einsum(
                        "kIeA,jceb->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "kIbE,jcEA->IjkAbc",
                        t2ab[:, :, b0:b1, :],
                        eris_ovVV[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                )
                w_blk += (
                    numpy.einsum(
                        "mIbA,jckm->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "kMbA,jcIM->IjkAbc",
                        t2ab[:, :, b0:b1, a0:a1],
                        eris_ovOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                w_blk += (
                    numpy.einsum(
                        "kIeA,jbec->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "kIcE,jbEA->IjkAbc",
                        t2ab[:, :, c0:c1, :],
                        eris_ovVV[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                )
                w_blk -= (
                    numpy.einsum(
                        "mIcA,jbkm->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "kMcA,jbIM->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                w_blk -= (
                    numpy.einsum(
                        "jIeA,kbec->IjkAbc",
                        t2ab[:, :, :, a0:a1],
                        eris_ovvv[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "jIcE,kbEA->IjkAbc",
                        t2ab[:, :, c0:c1, :],
                        eris_ovVV[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                )
                w_blk += (
                    numpy.einsum(
                        "mIcA,kbjm->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "jMcA,kbIM->IjkAbc",
                        t2ab[:, :, c0:c1, a0:a1],
                        eris_ovOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                )

                rw = w_blk / d3
                goo += numpy.einsum("kilabc,kjlabc->ij", wvd, rw) * 0.25
                goo += numpy.einsum("kliabc,kljabc->ij", wvd, rw) * 0.25
                gOO += numpy.einsum("iklabc,jklabc->ij", wvd, rw) * 0.25
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma1_intermediates pass2 [%d:%d]" % (a0, a1), *time2
        )

    blksize = min(
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nvira**2 * nvirb)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    if blksize < nocca or blksize < noccb:
        blksize = min(blksize, (nocca + 1) // 2)
        blksize = min(blksize, (noccb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = (%d,%d), (%d,%d)  blksize = %d",
        max_memory,
        nocca,
        noccb,
        nvira,
        nvirb,
        blksize,
    )
    time2 = logger.process_clock(), logger.perf_counter()
    for i0, i1 in lib.prange(0, noccb, blksize):
        for j0, j1 in lib.prange(0, nocca, blksize):
            for k0, k1 in lib.prange(0, nocca, blksize):
                w_blk = (
                    numpy.einsum(
                        "jIeA,kceb->IjkAbc",
                        t2ab[j0:j1, i0:i1, :, :],
                        eris_ovvv[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "jIbE,kcEA->IjkAbc",
                        t2ab[j0:j1, i0:i1, :, :],
                        eris_ovVV[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[j0:j1, k0:k1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                )
                w_blk -= (
                    numpy.einsum(
                        "mIbA,kcjm->IjkAbc",
                        t2ab[:, i0:i1, :, :],
                        eris_ovoo[k0:k1, :, j0:j1, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "jMbA,kcIM->IjkAbc",
                        t2ab[j0:j1, :, :, :],
                        eris_ovOO[k0:k1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[j0:j1, :, :, :],
                    eris_OVoo[i0:i1, :, k0:k1:, :],
                )
                v_blk = numpy.einsum(
                    "jbkc,IA->IjkAbc",
                    eris_ovov[j0:j1, :, k0:k1, :],
                    t1b[i0:i1, :],
                )
                v_blk += numpy.einsum(
                    "kcIA,jb->IjkAbc",
                    eris_ovOV[k0:k1, :, i0:i1, :],
                    t1a[j0:j1, :],
                )
                v_blk += numpy.einsum(
                    "kcIA,jb->IjkAbc",
                    eris_ovOV[k0:k1, :, i0:i1, :],
                    t1a[j0:j1, :],
                )
                v_blk += (
                    numpy.einsum(
                        "jkbc,AI->IjkAbc",
                        t2aa[j0:j1, k0:k1, :, :],
                        fVO[:, i0:i1],
                    )
                    * 0.5
                )
                v_blk += (
                    numpy.einsum(
                        "kIcA,bj->IjkAbc",
                        t2ab[k0:k1, i0:i1, :, :],
                        fvo[:, j0:j1],
                    )
                    * 2
                )
                d3 = lib.direct_sum(
                    "IA+jb+kc->IjkAbc", eIA[i0:i1, :], eia[j0:j1, :], eia[k0:k1, :]
                )
                wvd = (w_blk + v_blk) / d3

                w_blk -= (
                    numpy.einsum(
                        "kIeA,jceb->IjkAbc",
                        t2ab[k0:k1, i0:i1, :, :],
                        eris_ovvv[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "kIbE,jcEA->IjkAbc",
                        t2ab[k0:k1, i0:i1, :, :],
                        eris_ovVV[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[k0:k1, j0:j1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                )
                w_blk += (
                    numpy.einsum(
                        "mIbA,jckm->IjkAbc",
                        t2ab[:, i0:i1, :, :],
                        eris_ovoo[j0:j1, :, k0:k1, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "kMbA,jcIM->IjkAbc",
                        t2ab[k0:k1, :, :, :],
                        eris_ovOO[j0:j1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[k0:k1, :, :, :],
                    eris_OVoo[i0:i1, :, j0:j1:, :],
                )

                w_blk += (
                    numpy.einsum(
                        "kIeA,jbec->IjkAbc",
                        t2ab[k0:k1, i0:i1, :, :],
                        eris_ovvv[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "kIcE,jbEA->IjkAbc",
                        t2ab[k0:k1, i0:i1, :, :],
                        eris_ovVV[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[k0:k1, j0:j1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                )
                w_blk -= (
                    numpy.einsum(
                        "mIcA,jbkm->IjkAbc",
                        t2ab[:, i0:i1, :, :],
                        eris_ovoo[j0:j1, :, k0:k1, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "kMcA,jbIM->IjkAbc",
                        t2ab[k0:k1, :, :, :],
                        eris_ovOO[j0:j1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[k0:k1, :, :, :],
                    eris_OVoo[i0:i1, :, j0:j1:, :],
                )

                w_blk -= (
                    numpy.einsum(
                        "jIeA,kbec->IjkAbc",
                        t2ab[j0:j1, i0:i1, :, :],
                        eris_ovvv[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "jIcE,kbEA->IjkAbc",
                        t2ab[j0:j1, i0:i1, :, :],
                        eris_ovVV[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[j0:j1, k0:k1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                )
                w_blk += (
                    numpy.einsum(
                        "mIcA,kbjm->IjkAbc",
                        t2ab[:, i0:i1, :, :],
                        eris_ovoo[k0:k1, :, j0:j1, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "jMcA,kbIM->IjkAbc",
                        t2ab[j0:j1, :, :, :],
                        eris_ovOO[k0:k1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[j0:j1, :, :, :],
                    eris_OVoo[i0:i1, :, k0:k1:, :],
                )
                rw = w_blk / d3
                gvv += numpy.einsum("ijkcad,ijkcbd->ab", wvd, rw) * 0.25
                gvv += numpy.einsum("ijkcda,ijkcdb->ab", wvd, rw) * 0.25
                gVV += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.25
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma1_intermediates pass2 [%d:%d]" % (i0, i1), *time2
        )

    blksize = min(
        noccb,
        int(
            ((max_memory * 0.9e6 / 8) / 6.0 / (nocca * noccb * nvira * nvirb))
            ** (1 / 3)
        ),
    )
    blksize = min(nocca, blksize)
    if blksize < noccb or blksize < nocca:
        blksize = min(blksize, (nocca + 1) // 2)
        blksize = min(blksize, (noccb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = (%d,%d), (%d,%d)  blksize = %d",
        max_memory,
        nocca,
        noccb,
        nvira,
        nvirb,
        blksize,
    )
    time2 = logger.process_clock(), logger.perf_counter()
    for c0, c1 in lib.prange(0, nvira, blksize):
        for k0, k1 in lib.prange(0, nocca, blksize):
            w_blk = (
                numpy.einsum(
                    "jIeA,kceb->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovvv[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "jIbE,kcEA->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovVV[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "jkbe,IAec->IjkAbc",
                t2aa[:, k0:k1, :, :],
                eris_OVvv[:, :, :, c0:c1],
            )
            w_blk -= (
                numpy.einsum(
                    "mIbA,kcjm->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovoo[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "jMbA,kcIM->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovOO[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "jmbc,IAkm->IjkAbc",
                t2aa[:, :, :, c0:c1],
                eris_OVoo[:, :, k0:k1, :],
            )
            v_blk = numpy.einsum(
                "jbkc,IA->IjkAbc",
                eris_ovov[:, :, k0:k1, c0:c1],
                t1b[:, :],
            )
            v_blk += numpy.einsum(
                "kcIA,jb->IjkAbc",
                eris_ovOV[k0:k1, c0:c1, :, :],
                t1a[:, :],
            )
            v_blk += numpy.einsum(
                "kcIA,jb->IjkAbc",
                eris_ovOV[k0:k1, c0:c1, :, :],
                t1a[:, :],
            )
            v_blk += (
                numpy.einsum(
                    "jkbc,AI->IjkAbc",
                    t2aa[:, k0:k1, :, c0:c1],
                    fVO[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "kIcA,bj->IjkAbc",
                    t2ab[k0:k1, :, c0:c1, :],
                    fvo[:, :],
                )
                * 2
            )
            d3 = lib.direct_sum(
                "IA+jb+kc->IjkAbc", eIA[:, :], eia[:, :], eia[k0:k1, c0:c1]
            )
            wvd = (w_blk + v_blk) / d3

            w_blk -= (
                numpy.einsum(
                    "kIeA,jceb->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovvv[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "kIbE,jcEA->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovVV[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kjbe,IAec->IjkAbc",
                t2aa[k0:k1, :, :, :],
                eris_OVvv[:, :, :, c0:c1],
            )
            w_blk += (
                numpy.einsum(
                    "mIbA,jckm->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovoo[:, c0:c1, k0:k1, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "kMbA,jcIM->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovOO[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kmbc,IAjm->IjkAbc",
                t2aa[k0:k1, :, :, c0:c1],
                eris_OVoo[:, :, :, :],
            )

            w_blk += (
                numpy.einsum(
                    "kIeA,jbec->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovvv[:, :, :, c0:c1],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "kIcE,jbEA->IjkAbc",
                    t2ab[k0:k1, :, c0:c1, :],
                    eris_ovVV[:, :, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kjce,IAeb->IjkAbc",
                t2aa[k0:k1, :, c0:c1, :],
                eris_OVvv[:, :, :, :],
            )
            w_blk -= (
                numpy.einsum(
                    "mIcA,jbkm->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovoo[:, :, k0:k1, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "kMcA,jbIM->IjkAbc",
                    t2ab[k0:k1, :, c0:c1, :],
                    eris_ovOO[:, :, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kmcb,IAjm->IjkAbc",
                t2aa[k0:k1, :, c0:c1, :],
                eris_OVoo[:, :, :, :],
            )

            w_blk -= (
                numpy.einsum(
                    "jIeA,kbec->IjkAbc",
                    t2ab[:, :, :, :],
                    eris_ovvv[k0:k1, :, :, c0:c1],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "jIcE,kbEA->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovVV[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "jkce,IAeb->IjkAbc",
                t2aa[:, k0:k1, c0:c1, :],
                eris_OVvv[:, :, :, :],
            )
            w_blk += (
                numpy.einsum(
                    "mIcA,kbjm->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovoo[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "jMcA,kbIM->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovOO[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "jmcb,IAkm->IjkAbc",
                t2aa[:, :, c0:c1, :],
                eris_OVoo[:, :, k0:k1, :],
            )
            rw = w_blk / d3
            gvo += numpy.einsum("kica,ijkabc->bj", t2ab[k0:k1, :, c0:c1, :], rw) * 0.5
            gVO += numpy.einsum("jkbc,ijkabc->ai", t2aa[:, k0:k1, :, c0:c1], rw) * 0.125
        time2 = log.timer_debug1(
            "uccsd_t rdm _gamma1_intermediates pass2 [%d:%d]" % (c0, c1), *time2
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
        "ccsd_t spin 2 rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
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
                    numpy.einsum(
                        "ijae,kceb->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "ijeb,kcea->ijkabc",
                        t2ab[:, :, :, b0:b1],
                        eris_OVvv[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                )
                w_blk -= (
                    numpy.einsum(
                        "imab,kcjm->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "mjab,kcim->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                )
                v_blk = numpy.einsum(
                    "jbkc,ia->ijkabc",
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                )
                v_blk += numpy.einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk += numpy.einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[:, a0:a1, :, c0:c1],
                    t1b[:, b0:b1],
                )
                v_blk += (
                    numpy.einsum(
                        "JKBC,ai->iJKaBC",
                        t2bb[:, :, b0:b1, c0:c1],
                        fvo[a0:a1, :],
                    )
                    * 0.5
                )
                v_blk += (
                    numpy.einsum(
                        "iKaC,BJ->iJKaBC",
                        t2ab[:, :, a0:a1, c0:c1],
                        fVO[b0:b1, :],
                    )
                    * 2
                )
                d3 = lib.direct_sum(
                    "ia+jb+kc->ijkabc", eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                wvd = (w_blk + v_blk) / d3

                w_blk -= (
                    numpy.einsum(
                        "ikae,jceb->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, c0:c1, :, b0:b1],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "ikeb,jcea->ijkabc",
                        t2ab[:, :, :, b0:b1],
                        eris_OVvv[:, c0:c1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                )
                w_blk += (
                    numpy.einsum(
                        "imab,jckm->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVOO[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "mkab,jcim->ijkabc",
                        t2ab[:, :, a0:a1, b0:b1],
                        eris_OVoo[:, c0:c1, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                )

                w_blk += (
                    numpy.einsum(
                        "ikae,jbec->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "ikec,jbea->ijkabc",
                        t2ab[:, :, :, c0:c1],
                        eris_OVvv[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                )
                w_blk -= (
                    numpy.einsum(
                        "imac,jbkm->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "mkac,jbim->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                )

                w_blk -= (
                    numpy.einsum(
                        "ijae,kbec->ijkabc",
                        t2ab[:, :, a0:a1, :],
                        eris_OVVV[:, b0:b1, :, c0:c1],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "ijec,kbea->ijkabc",
                        t2ab[:, :, :, c0:c1],
                        eris_OVvv[:, b0:b1, :, a0:a1],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                )
                w_blk += (
                    numpy.einsum(
                        "imac,kbjm->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVOO[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "mjac,kbim->ijkabc",
                        t2ab[:, :, a0:a1, c0:c1],
                        eris_OVoo[:, b0:b1, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                )
                rw = w_blk / d3
                goo += numpy.einsum("iklabc,jklabc->ij", wvd, rw) * 0.25
                gOO += numpy.einsum("kilabc,kjlabc->ij", wvd, rw) * 0.25
                gOO += numpy.einsum("kliabc,kljabc->ij", wvd, rw) * 0.25
        time2 = log.timer_debug1(
            "uccsd_t spin 2 rdm _gamma1_intermediates pass2 [%d:%d]" % (a0, a1), *time2
        )

    blksize = min(
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nvirb**2 * nvira)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    if blksize < nocca or blksize < noccb:
        blksize = min(blksize, (nocca + 1) // 2)
        blksize = min(blksize, (noccb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = (%d,%d), (%d,%d)  blksize = %d",
        max_memory,
        nocca,
        noccb,
        nvira,
        nvirb,
        blksize,
    )
    time2 = logger.process_clock(), logger.perf_counter()
    for i0, i1 in lib.prange(0, nocca, blksize):
        for j0, j1 in lib.prange(0, noccb, blksize):
            for k0, k1 in lib.prange(0, noccb, blksize):
                w_blk = (
                    numpy.einsum(
                        "ijae,kceb->ijkabc",
                        t2ab[i0:i1, j0:j1, :, :],
                        eris_OVVV[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "ijeb,kcea->ijkabc",
                        t2ab[i0:i1, j0:j1, :, :],
                        eris_OVvv[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[j0:j1, k0:k1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                )
                w_blk -= (
                    numpy.einsum(
                        "imab,kcjm->ijkabc",
                        t2ab[i0:i1, :, :, :],
                        eris_OVOO[k0:k1, :, j0:j1, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "mjab,kcim->ijkabc",
                        t2ab[:, j0:j1, :, :],
                        eris_OVoo[k0:k1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[j0:j1, :, :, :],
                    eris_ovOO[i0:i1, :, k0:k1, :],
                )
                v_blk = numpy.einsum(
                    "jbkc,ia->ijkabc",
                    eris_OVOV[j0:j1, :, k0:k1, :],
                    t1a[i0:i1, :],
                )
                v_blk += numpy.einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[i0:i1, :, k0:k1, :],
                    t1b[j0:j1, :],
                )
                v_blk += numpy.einsum(
                    "iakc,jb->ijkabc",
                    eris_ovOV[i0:i1, :, k0:k1, :],
                    t1b[j0:j1, :],
                )
                v_blk += (
                    numpy.einsum(
                        "JKBC,ai->iJKaBC",
                        t2bb[j0:j1, k0:k1, :, :],
                        fvo[:, i0:i1],
                    )
                    * 0.5
                )
                v_blk += (
                    numpy.einsum(
                        "iKaC,BJ->iJKaBC",
                        t2ab[i0:i1, k0:k1, :, :],
                        fVO[:, j0:j1],
                    )
                    * 2
                )
                d3 = lib.direct_sum(
                    "ia+jb+kc->ijkabc", eia[i0:i1, :], eIA[j0:j1, :], eIA[k0:k1, :]
                )
                wvd = (w_blk + v_blk) / d3

                w_blk -= (
                    numpy.einsum(
                        "ikae,jceb->ijkabc",
                        t2ab[i0:i1, k0:k1, :, :],
                        eris_OVVV[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "ikeb,jcea->ijkabc",
                        t2ab[i0:i1, k0:k1, :, :],
                        eris_OVvv[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[k0:k1, j0:j1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                )
                w_blk += (
                    numpy.einsum(
                        "imab,jckm->ijkabc",
                        t2ab[i0:i1, :, :, :],
                        eris_OVOO[j0:j1, :, k0:k1, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "mkab,jcim->ijkabc",
                        t2ab[:, k0:k1, :, :],
                        eris_OVoo[j0:j1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[k0:k1, :, :, :],
                    eris_ovOO[i0:i1, :, j0:j1, :],
                )

                w_blk += (
                    numpy.einsum(
                        "ikae,jbec->ijkabc",
                        t2ab[i0:i1, k0:k1, :, :],
                        eris_OVVV[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "ikec,jbea->ijkabc",
                        t2ab[i0:i1, k0:k1, :, :],
                        eris_OVvv[j0:j1, :, :, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[k0:k1, j0:j1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                )
                w_blk -= (
                    numpy.einsum(
                        "imac,jbkm->ijkabc",
                        t2ab[i0:i1, :, :, :],
                        eris_OVOO[j0:j1, :, k0:k1, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "mkac,jbim->ijkabc",
                        t2ab[:, k0:k1, :, :],
                        eris_OVoo[j0:j1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[k0:k1, :, :, :],
                    eris_ovOO[i0:i1, :, j0:j1, :],
                )

                w_blk -= (
                    numpy.einsum(
                        "ijae,kbec->ijkabc",
                        t2ab[i0:i1, j0:j1, :, :],
                        eris_OVVV[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk -= (
                    numpy.einsum(
                        "ijec,kbea->ijkabc",
                        t2ab[i0:i1, j0:j1, :, :],
                        eris_OVvv[k0:k1, :, :, :],
                    )
                    * 2
                )
                w_blk -= numpy.einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[j0:j1, k0:k1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                )
                w_blk += (
                    numpy.einsum(
                        "imac,kbjm->ijkabc",
                        t2ab[i0:i1, :, :, :],
                        eris_OVOO[k0:k1, :, j0:j1, :],
                    )
                    * 2
                )
                w_blk += (
                    numpy.einsum(
                        "mjac,kbim->ijkabc",
                        t2ab[:, j0:j1, :, :],
                        eris_OVoo[k0:k1, :, i0:i1, :],
                    )
                    * 2
                )
                w_blk += numpy.einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[j0:j1, :, :, :],
                    eris_ovOO[i0:i1, :, k0:k1, :],
                )
                rw = w_blk / d3
                gvv += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.25
                gVV += numpy.einsum("ijkcad,ijkcbd->ab", wvd, rw) * 0.25
                gVV += numpy.einsum("ijkcda,ijkcdb->ab", wvd, rw) * 0.25

    blksize = min(
        noccb,
        int(
            ((max_memory * 0.9e6 / 8) / 6.0 / (nocca * noccb * nvira * nvirb))
            ** (1 / 3)
        ),
    )
    blksize = min(nocca, blksize)
    if blksize < noccb or blksize < nocca:
        blksize = min(blksize, (nocca + 1) // 2)
        blksize = min(blksize, (noccb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = (%d,%d), (%d,%d)  blksize = %d",
        max_memory,
        nocca,
        noccb,
        nvira,
        nvirb,
        blksize,
    )
    time2 = logger.process_clock(), logger.perf_counter()
    for c0, c1 in lib.prange(0, nvirb, blksize):
        for k0, k1 in lib.prange(0, noccb, blksize):
            w_blk = (
                numpy.einsum(
                    "ijae,kceb->ijkabc",
                    t2ab,
                    eris_OVVV[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "ijeb,kcea->ijkabc",
                    t2ab,
                    eris_OVvv[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, k0:k1, :, :],
                eris_ovVV[:, :, :, c0:c1],
            )
            w_blk -= (
                numpy.einsum(
                    "imab,kcjm->ijkabc",
                    t2ab,
                    eris_OVOO[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "mjab,kcim->ijkabc",
                    t2ab,
                    eris_OVoo[k0:k1, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_ovOO[:, :, k0:k1, :],
            )
            v_blk = numpy.einsum(
                "jbkc,ia->ijkabc",
                eris_OVOV[:, :, k0:k1, c0:c1],
                t1a[:, :],
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_ovOV[:, :, k0:k1, c0:c1],
                t1b[:, :],
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_ovOV[:, :, k0:k1, c0:c1],
                t1b[:, :],
            )
            v_blk += (
                numpy.einsum(
                    "JKBC,ai->iJKaBC",
                    t2bb[:, k0:k1, :, c0:c1],
                    fvo[:, :],
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "iKaC,BJ->iJKaBC",
                    t2ab[:, k0:k1, :, c0:c1],
                    fVO[:, :],
                )
                * 2
            )
            d3 = lib.direct_sum(
                "ia+jb+kc->ijkabc", eia[:, :], eIA[:, :], eIA[k0:k1, c0:c1]
            )
            wvd = (w_blk + v_blk) / d3

            w_blk -= (
                numpy.einsum(
                    "ikae,jceb->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVVV[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "ikeb,jcea->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVvv[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kjbe,iaec->ijkabc",
                t2bb[k0:k1, :, :, :],
                eris_ovVV[:, :, :, c0:c1],
            )
            w_blk += (
                numpy.einsum(
                    "imab,jckm->ijkabc",
                    t2ab[:, :, :, :],
                    eris_OVOO[:, c0:c1, k0:k1, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "mkab,jcim->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVoo[:, c0:c1, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kmbc,iajm->ijkabc",
                t2bb[k0:k1, :, :, c0:c1],
                eris_ovOO[:, :, :, :],
            )

            w_blk += (
                numpy.einsum(
                    "ikae,jbec->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVVV[:, :, :, c0:c1],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "ikec,jbea->ijkabc",
                    t2ab[:, k0:k1, :, c0:c1],
                    eris_OVvv[:, :, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kjce,iaeb->ijkabc",
                t2bb[k0:k1, :, c0:c1, :],
                eris_ovVV[:, :, :, :],
            )
            w_blk -= (
                numpy.einsum(
                    "imac,jbkm->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVOO[:, :, k0:k1, :],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "mkac,jbim->ijkabc",
                    t2ab[:, k0:k1, :, c0:c1],
                    eris_OVoo[:, :, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kmcb,iajm->ijkabc",
                t2bb[k0:k1, :, c0:c1, :],
                eris_ovOO[:, :, :, :],
            )

            w_blk -= (
                numpy.einsum(
                    "ijae,kbec->ijkabc",
                    t2ab[:, :, :, :],
                    eris_OVVV[k0:k1, :, :, c0:c1],
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "ijec,kbea->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVvv[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "jkce,iaeb->ijkabc",
                t2bb[:, k0:k1, c0:c1, :],
                eris_ovVV[:, :, :, :],
            )
            w_blk += (
                numpy.einsum(
                    "imac,kbjm->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVOO[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "mjac,kbim->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVoo[k0:k1, :, :, :],
                )
                * 2
            )
            w_blk += numpy.einsum(
                "jmcb,iakm->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_ovOO[:, :, k0:k1, :],
            )
            rw = w_blk / d3
            gVO += numpy.einsum("ikac,ijkabc->bj", t2ab[:, k0:k1, :, c0:c1], rw) * 0.5
            gvo += numpy.einsum("jkbc,ijkabc->ai", t2bb[:, k0:k1, :, c0:c1], rw) * 0.125

    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]

    # if for_grad:
    #     doo -= goo
    #     dOO -= gOO
    #     dvv += gvv
    #     dVV += gVV
    # else:
    #     doo[numpy.diag_indices(nocca)] -= goo.diagonal()
    #     dOO[numpy.diag_indices(noccb)] -= gOO.diagonal()
    #     dvv[numpy.diag_indices(nvira)] += gvv.diagonal()
    #     dVV[numpy.diag_indices(nvirb)] += gVV.diagonal()

    dvo += gvo
    dVO += gVO

    return d1, (goo, gOO, gvv, gVV)


def r6(w):
    return (
        w
        + w.transpose(2, 0, 1, 3, 4, 5)
        + w.transpose(1, 2, 0, 3, 4, 5)
        - w.transpose(2, 1, 0, 3, 4, 5)
        - w.transpose(0, 2, 1, 3, 4, 5)
        - w.transpose(1, 0, 2, 3, 4, 5)
    )
