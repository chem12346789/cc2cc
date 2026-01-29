import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm


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
    blksize = 5
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
    blksize = 5
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
    blksize = 5
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
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocca**2 * noccb)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    blksize = 5
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
    for i0, i1 in lib.prange(0, nvirb, blksize):
        for j0, j1 in lib.prange(0, nvira, blksize):
            for k0, k1 in lib.prange(0, nvira, blksize):
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
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocca**2 * noccb)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    blksize = 5
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
        for k0, k1 in lib.prange(0, nvira, blksize):
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
    goo = numpy.zeros((nocca, nocca), dtype=t1a.dtype)
    gOO = numpy.zeros((noccb, noccb), dtype=t1b.dtype)
    gvv = numpy.zeros((nvira, nvira), dtype=t1a.dtype)
    gVV = numpy.zeros((nvirb, nvirb), dtype=t1b.dtype)
    gvo = numpy.zeros((nvira, nocca), dtype=t1a.dtype)
    gVO = numpy.zeros((nvirb, noccb), dtype=t1b.dtype)

    blksize = min(
        nvirb, int(((max_memory * 0.9e6 / 8) / 6.0 / (noccb**2 * nocca)) ** (1 / 3))
    )
    blksize = min(nvira, blksize)
    blksize = 5
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
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (noccb**2 * nocca)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    blksize = 5
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
    for i0, i1 in lib.prange(0, nvira, blksize):
        for j0, j1 in lib.prange(0, nvirb, blksize):
            for k0, k1 in lib.prange(0, nvirb, blksize):
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
        noccb, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocca**2 * noccb)) ** (1 / 3))
    )
    blksize = min(nocca, blksize)
    blksize = 5
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
        for k0, k1 in lib.prange(0, nvira, blksize):
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
                eris_OVOV,
                t1a,
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_ovOV,
                t1b,
            )
            v_blk += numpy.einsum(
                "iakc,jb->ijkabc",
                eris_ovOV,
                t1b,
            )
            v_blk += (
                numpy.einsum(
                    "JKBC,ai->iJKaBC",
                    t2bb,
                    fvo,
                )
                * 0.5
            )
            v_blk += (
                numpy.einsum(
                    "iKaC,BJ->iJKaBC",
                    t2ab,
                    fVO,
                )
                * 2
            )

            w_blk -= (
                numpy.einsum(
                    "ikae,jceb->ijkabc",
                    t2ab,
                    eris_OVVV,
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "ikeb,jcea->ijkabc",
                    t2ab,
                    eris_OVvv,
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kjbe,iaec->ijkabc",
                t2bb,
                eris_ovVV,
            )
            w_blk += (
                numpy.einsum(
                    "imab,jckm->ijkabc",
                    t2ab,
                    eris_OVOO,
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "mkab,jcim->ijkabc",
                    t2ab,
                    eris_OVoo,
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kmbc,iajm->ijkabc",
                t2bb,
                eris_ovOO,
            )

            w_blk += (
                numpy.einsum(
                    "ikae,jbec->ijkabc",
                    t2ab,
                    eris_OVVV,
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "ikec,jbea->ijkabc",
                    t2ab,
                    eris_OVvv,
                )
                * 2
            )
            w_blk += numpy.einsum(
                "kjce,iaeb->ijkabc",
                t2bb,
                eris_ovVV,
            )
            w_blk -= (
                numpy.einsum(
                    "imac,jbkm->ijkabc",
                    t2ab,
                    eris_OVOO,
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "mkac,jbim->ijkabc",
                    t2ab,
                    eris_OVoo,
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "kmcb,iajm->ijkabc",
                t2bb,
                eris_ovOO,
            )

            w_blk -= (
                numpy.einsum(
                    "ijae,kbec->ijkabc",
                    t2ab,
                    eris_OVVV,
                )
                * 2
            )
            w_blk -= (
                numpy.einsum(
                    "ijec,kbea->ijkabc",
                    t2ab,
                    eris_OVvv,
                )
                * 2
            )
            w_blk -= numpy.einsum(
                "jkce,iaeb->ijkabc",
                t2bb,
                eris_ovVV,
            )
            w_blk += (
                numpy.einsum(
                    "imac,kbjm->ijkabc",
                    t2ab,
                    eris_OVOO,
                )
                * 2
            )
            w_blk += (
                numpy.einsum(
                    "mjac,kbim->ijkabc",
                    t2ab,
                    eris_OVoo,
                )
                * 2
            )
            w_blk += numpy.einsum(
                "jmcb,iakm->ijkabc",
                t2bb,
                eris_ovOO,
            )

    w = numpy.einsum("ijae,kceb->ijkabc", t2ab, eris_OVVV) * 2
    w += numpy.einsum("ijeb,kcea->ijkabc", t2ab, eris_OVvv) * 2
    w += numpy.einsum("jkbe,iaec->ijkabc", t2bb, eris_ovVV)
    w -= numpy.einsum("imab,kcjm->ijkabc", t2ab, eris_OVOO) * 2
    w -= numpy.einsum("mjab,kcim->ijkabc", t2ab, eris_OVoo) * 2
    w -= numpy.einsum("jmbc,iakm->ijkabc", t2bb, eris_ovOO)
    v = numpy.einsum("jbkc,ia->ijkabc", eris_OVOV, t1a)
    v += numpy.einsum("iakc,jb->ijkabc", eris_ovOV, t1b)
    v += numpy.einsum("iakc,jb->ijkabc", eris_ovOV, t1b)
    v += numpy.einsum("JKBC,ai->iJKaBC", t2bb, fvo) * 0.5
    v += numpy.einsum("iKaC,BJ->iJKaBC", t2ab, fVO) * 2
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eIA, eIA)
    wvd = (w + v) / d3
    rw = r4(w) / d3
    print(numpy.allclose(gvv, numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.25))
    print(
        numpy.allclose(
            gVV,
            numpy.einsum("ijkcad,ijkcbd->ab", wvd, rw) * 0.25
            + numpy.einsum("ijkcda,ijkcdb->ab", wvd, rw) * 0.25,
        )
    )
    raise Exception("stop here")
    # gVO += numpy.einsum("ikac,ijkabc->bj", t2ab, rw) * 0.5
    # gvo += numpy.einsum("jkbc,ijkabc->ai", t2bb, rw) * 0.125

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


def u_gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None, compress_vvvv=False):
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
    eia = mo_ea[:nocca, None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:, :nocca]
    fVO = eris.fockb[noccb:, :noccb]

    # aaa
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eia, eia)
    w = numpy.einsum("ijae,kceb->ijkabc", t2aa, numpy.asarray(eris.get_ovvv()))
    w -= numpy.einsum("mkbc,iajm->ijkabc", t2aa, numpy.asarray(eris.ovoo))
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.ovov), t1a)
    v += numpy.einsum("jkbc,ai->ijkabc", t2aa, fvo) * 0.5

    rw = r6(p6(w)) / d3
    wvd = r6(p6(w * 2 + v)) / d3
    dovov += numpy.einsum("ia,ijkabc->jbkc", t1a, rw) * 0.25
    # *(1/8) instead of (1/4) because ooov appears 4 times in the 2pdm tensor due
    # to symmetrization, and its contribution is scaled by 1/2 in Tr(H,2pdm)
    dooov -= numpy.einsum("mkbc,ijkabc->jmia", t2aa, wvd) * 0.125
    dovvv += numpy.einsum("kjcf,ijkabc->iafb", t2aa, wvd) * 0.125

    # bbb
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eIA, eIA)
    w = numpy.einsum("ijae,kceb->ijkabc", t2bb, numpy.asarray(eris.get_OVVV()))
    w -= numpy.einsum("imab,kcjm->ijkabc", t2bb, numpy.asarray(eris.OVOO))
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV), t1b)
    v += numpy.einsum("jkbc,ai->ijkabc", t2bb, fVO) * 0.5

    rw = r6(p6(w)) / d3
    wvd = r6(p6(w * 2 + v)) / d3
    dOVOV += numpy.einsum("ia,ijkabc->jbkc", t1b, rw) * 0.25
    dOOOV -= numpy.einsum("mkbc,ijkabc->jmia", t2bb, wvd) * 0.125
    dOVVV += numpy.einsum("kjcf,ijkabc->iafb", t2bb, wvd) * 0.125

    # baa
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eia, eia)
    w = numpy.einsum("jIeA,kceb->IjkAbc", t2ab, numpy.asarray(eris.get_ovvv())) * 2
    w += numpy.einsum("jIbE,kcEA->IjkAbc", t2ab, numpy.asarray(eris.get_ovVV())) * 2
    w += numpy.einsum("jkbe,IAec->IjkAbc", t2aa, numpy.asarray(eris.get_OVvv()))
    w -= numpy.einsum("mIbA,kcjm->IjkAbc", t2ab, numpy.asarray(eris.ovoo)) * 2
    w -= numpy.einsum("jMbA,kcIM->IjkAbc", t2ab, numpy.asarray(eris.ovOO)) * 2
    w -= numpy.einsum("jmbc,IAkm->IjkAbc", t2aa, numpy.asarray(eris.OVoo))
    v = numpy.einsum("jbkc,IA->IjkAbc", numpy.asarray(eris.ovov), t1b)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV), t1a)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV), t1a)
    v += numpy.einsum("jkbc,AI->IjkAbc", t2aa, fVO) * 0.5
    v += numpy.einsum("kIcA,bj->IjkAbc", t2ab, fvo) * 2

    rw = r4(w) / d3
    wvd = r4(w * 2 + v) / d3
    dovvv += numpy.einsum("jiea,ijkabc->kceb", t2ab, wvd) * 0.25
    dovVV += numpy.einsum("jibe,ijkabc->kcea", t2ab, wvd) * 0.25
    dOVvv += numpy.einsum("jkbe,ijkabc->iaec", t2aa, wvd) * 0.125
    dooov -= numpy.einsum("miba,ijkabc->jmkc", t2ab, wvd) * 0.25
    dOOov -= numpy.einsum("jmba,ijkabc->imkc", t2ab, wvd) * 0.25
    dooOV -= numpy.einsum("jmbc,ijkabc->kmia", t2aa, wvd) * 0.125
    dovov += numpy.einsum("ia,ijkabc->jbkc", t1b, rw) * 0.25
    # dOVov += numpy.einsum('jb,ijkabc->iakc', t1a, rw) * .25
    dovOV += numpy.einsum("jb,ijkabc->kcia", t1a, rw) * 0.25

    # bba
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eIA, eIA)
    w = numpy.einsum("ijae,kceb->ijkabc", t2ab, numpy.asarray(eris.get_OVVV())) * 2
    w += numpy.einsum("ijeb,kcea->ijkabc", t2ab, numpy.asarray(eris.get_OVvv())) * 2
    w += numpy.einsum("jkbe,iaec->ijkabc", t2bb, numpy.asarray(eris.get_ovVV()))
    w -= numpy.einsum("imab,kcjm->ijkabc", t2ab, numpy.asarray(eris.OVOO)) * 2
    w -= numpy.einsum("mjab,kcim->ijkabc", t2ab, numpy.asarray(eris.OVoo)) * 2
    w -= numpy.einsum("jmbc,iakm->ijkabc", t2bb, numpy.asarray(eris.ovOO))
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV), t1a)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV), t1b)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV), t1b)
    v += numpy.einsum("JKBC,ai->iJKaBC", t2bb, fvo) * 0.5
    v += numpy.einsum("iKaC,BJ->iJKaBC", t2ab, fVO) * 2

    rw = r4(w) / d3
    wvd = r4(w * 2 + v) / d3
    dOVVV += numpy.einsum("ijae,ijkabc->kceb", t2ab, wvd) * 0.25
    dOVvv += numpy.einsum("ijeb,ijkabc->kcea", t2ab, wvd) * 0.25
    dovVV += numpy.einsum("jkbe,ijkabc->iaec", t2bb, wvd) * 0.125
    dOOOV -= numpy.einsum("imab,ijkabc->jmkc", t2ab, wvd) * 0.25
    dooOV -= numpy.einsum("mjab,ijkabc->imkc", t2ab, wvd) * 0.25
    dOOov -= numpy.einsum("jmbc,ijkabc->kmia", t2bb, wvd) * 0.125
    dOVOV += numpy.einsum("ia,ijkabc->jbkc", t1a, rw) * 0.25
    dovOV += numpy.einsum("jb,ijkabc->iakc", t1b, rw) * 0.25
    # dOVov += numpy.einsum('jb,ijkabc->kcia', t1b, rw) * .25

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


def p6(t):
    return (
        t
        + t.transpose(1, 2, 0, 4, 5, 3)
        + t.transpose(2, 0, 1, 5, 3, 4)
        + t.transpose(0, 2, 1, 3, 5, 4)
        + t.transpose(2, 1, 0, 5, 4, 3)
        + t.transpose(1, 0, 2, 4, 3, 5)
    )


def r6(w):
    return (
        w
        + w.transpose(2, 0, 1, 3, 4, 5)
        + w.transpose(1, 2, 0, 3, 4, 5)
        - w.transpose(2, 1, 0, 3, 4, 5)
        - w.transpose(0, 2, 1, 3, 4, 5)
        - w.transpose(1, 0, 2, 3, 4, 5)
    )


def r4(w):
    w = w - w.transpose(0, 2, 1, 3, 4, 5)
    w = w + w.transpose(0, 2, 1, 3, 5, 4)
    return w
