import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import uccsd_rdm

from cc2cc.utils.tensor_helper import parallel_einsum, _w_einsum
from cc2cc.utils.pyscf_uccsd_t_lambda import (
    _mixed_spin_divide,
    _mixed_spin_v,
    _run_fused,
    _same_spin_r6_divide,
    _same_spin_v,
)


def u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None, for_grad=False):
    """Build unrestricted 1-RDM intermediates."""
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

    for b0, b1 in lib.prange(0, nvira, blksize):
        for c0, c1 in lib.prange(0, nvira, blksize):
            w_blk = _w_einsum(
                "ijae,kceb->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, c0:c1, :, b0:b1],
            )
            w_blk = _w_einsum(
                "kice,jbea->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, b0:b1, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "jkbe,iaec->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "ikae,jbec->ijkabc",
                t2aa[:, :, :, :],
                eris_ovvv[:, b0:b1, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "kjce,iaeb->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovvv[:, :, :, b0:b1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "jibe,kcea->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovvv[:, c0:c1, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "mkbc,iajm->ijkabc",
                t2aa[:, :, b0:b1, c0:c1],
                eris_ovoo[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mjab,kcim->ijkabc",
                t2aa[:, :, :, b0:b1],
                eris_ovoo[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mica,jbkm->ijkabc",
                t2aa[:, :, c0:c1, :],
                eris_ovoo[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mjcb,iakm->ijkabc",
                t2aa[:, :, c0:c1, b0:b1],
                eris_ovoo[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "miba,kcjm->ijkabc",
                t2aa[:, :, b0:b1, :],
                eris_ovoo[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mkac,jbim->ijkabc",
                t2aa[:, :, :, c0:c1],
                eris_ovoo[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            v_blk = _run_fused(
                _same_spin_v, eris_ovov, t1a, t2aa, fvo, b0, b1, c0, c1
            )

            rw = _run_fused(_same_spin_r6_divide, w_blk, None, eia, b0, c0)
            wvd = _run_fused(
                _same_spin_r6_divide, w_blk, v_blk, eia, b0, c0, False
            )

            goo += parallel_einsum("iklabc,jklabc->ij", wvd, rw) * 0.125
            gvv += parallel_einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.125
            gvo += parallel_einsum("jkbc,ijkabc->ai", t2aa[:, :, b0:b1, c0:c1], rw) * 0.125

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
    for b0, b1 in lib.prange(0, nvirb, blksize):
        for c0, c1 in lib.prange(0, nvirb, blksize):
            w_blk = _w_einsum(
                "ijae,kceb->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, c0:c1, :, b0:b1],
            )
            w_blk = _w_einsum(
                "kice,jbea->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, b0:b1, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "ikae,jbec->ijkabc",
                t2bb[:, :, :, :],
                eris_OVVV[:, b0:b1, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "kjce,iaeb->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVVV[:, :, :, b0:b1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "jibe,kcea->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVVV[:, c0:c1, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "imab,kcjm->ijkabc",
                t2bb[:, :, :, b0:b1],
                eris_OVOO[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, b0:b1, c0:c1],
                eris_OVOO[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "kmca,jbim->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_OVOO[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "imac,jbkm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_OVOO[:, b0:b1, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "kmcb,iajm->ijkabc",
                t2bb[:, :, c0:c1, b0:b1],
                eris_OVOO[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "jmba,kcim->ijkabc",
                t2bb[:, :, b0:b1, :],
                eris_OVOO[:, c0:c1, :, :],
                out=w_blk, subtract=True,
            )
            v_blk = _run_fused(
                _same_spin_v, eris_OVOV, t1b, t2bb, fVO, b0, b1, c0, c1
            )

            rw = _run_fused(_same_spin_r6_divide, w_blk, None, eIA, b0, c0)
            wvd = _run_fused(
                _same_spin_r6_divide, w_blk, v_blk, eIA, b0, c0, False
            )

            gOO += parallel_einsum("iklabc,jklabc->ij", wvd, rw) * 0.125
            gVV += parallel_einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.125
            gVO += parallel_einsum("jkbc,ijkabc->ai", t2bb[:, :, b0:b1, c0:c1], rw) * 0.125

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
    for a0, a1 in lib.prange(0, nvirb, blksize):
        for b0, b1 in lib.prange(0, nvira, blksize):
            for c0, c1 in lib.prange(0, nvira, blksize):
                w_blk = _w_einsum(
                    "jIeA,kceb->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    scale=2,
                )
                w_blk = _w_einsum(
                    "jIbE,kcEA->IjkAbc",
                    t2ab[:, :, b0:b1, :],
                    eris_ovVV[:, c0:c1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "mIbA,kcjm->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovoo[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jMbA,kcIM->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovOO[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )
                v_blk = _run_fused(
                    _mixed_spin_v,
                    eris_ovov[:, b0:b1, :, c0:c1],
                    t1b[:, a0:a1],
                    eris_ovOV[:, c0:c1, :, a0:a1],
                    t1a[:, b0:b1],
                    t2aa[:, :, b0:b1, c0:c1],
                    fVO[a0:a1, :],
                    t2ab[:, :, c0:c1, a0:a1],
                    fvo[b0:b1, :],
                )

                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1], 1
                )

                w_blk = _w_einsum(
                    "kIeA,jceb->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kIbE,jcEA->IjkAbc",
                    t2ab[:, :, b0:b1, :],
                    eris_ovVV[:, c0:c1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[:, :, b0:b1, :],
                    eris_OVvv[:, a0:a1, :, c0:c1],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "mIbA,jckm->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovoo[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kMbA,jcIM->IjkAbc",
                    t2ab[:, :, b0:b1, a0:a1],
                    eris_ovOO[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[:, :, b0:b1, c0:c1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk,
                )

                w_blk = _w_einsum(
                    "kIeA,jbec->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kIcE,jbEA->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovVV[:, b0:b1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "mIcA,jbkm->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovoo[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kMcA,jbIM->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovOO[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )

                w_blk = _w_einsum(
                    "jIeA,kbec->IjkAbc",
                    t2ab[:, :, :, a0:a1],
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jIcE,kbEA->IjkAbc",
                    t2ab[:, :, c0:c1, :],
                    eris_ovVV[:, b0:b1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[:, :, c0:c1, :],
                    eris_OVvv[:, a0:a1, :, b0:b1],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "mIcA,kbjm->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovoo[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jMcA,kbIM->IjkAbc",
                    t2ab[:, :, c0:c1, a0:a1],
                    eris_ovOO[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[:, :, c0:c1, b0:b1],
                    eris_OVoo[:, a0:a1, :, :],
                    out=w_blk,
                )

                rw = _mixed_spin_divide(
                    w_blk, None, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                goo += parallel_einsum("kilabc,kjlabc->ij", wvd, rw) * 0.25
                goo += parallel_einsum("kliabc,kljabc->ij", wvd, rw) * 0.25
                gOO += parallel_einsum("iklabc,jklabc->ij", wvd, rw) * 0.25

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
    for i0, i1 in lib.prange(0, noccb, blksize):
        for j0, j1 in lib.prange(0, nocca, blksize):
            for k0, k1 in lib.prange(0, nocca, blksize):
                w_blk = _w_einsum(
                    "jIeA,kceb->IjkAbc",
                    t2ab[j0:j1, i0:i1, :, :],
                    eris_ovvv[k0:k1, :, :, :],
                    scale=2,
                )
                w_blk = _w_einsum(
                    "jIbE,kcEA->IjkAbc",
                    t2ab[j0:j1, i0:i1, :, :],
                    eris_ovVV[k0:k1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jkbe,IAec->IjkAbc",
                    t2aa[j0:j1, k0:k1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "mIbA,kcjm->IjkAbc",
                    t2ab[:, i0:i1, :, :],
                    eris_ovoo[k0:k1, :, j0:j1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jMbA,kcIM->IjkAbc",
                    t2ab[j0:j1, :, :, :],
                    eris_ovOO[k0:k1, :, i0:i1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jmbc,IAkm->IjkAbc",
                    t2aa[j0:j1, :, :, :],
                    eris_OVoo[i0:i1, :, k0:k1:, :],
                    out=w_blk, subtract=True,
                )
                v_blk = _run_fused(
                    _mixed_spin_v,
                    eris_ovov[j0:j1, :, k0:k1, :],
                    t1b[i0:i1, :],
                    eris_ovOV[k0:k1, :, i0:i1, :],
                    t1a[j0:j1, :],
                    t2aa[j0:j1, k0:k1, :, :],
                    fVO[:, i0:i1],
                    t2ab[k0:k1, i0:i1, :, :],
                    fvo[:, j0:j1],
                )
                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eIA[i0:i1, :], eia[j0:j1, :], eia[k0:k1, :], 1
                )

                w_blk = _w_einsum(
                    "kIeA,jceb->IjkAbc",
                    t2ab[k0:k1, i0:i1, :, :],
                    eris_ovvv[j0:j1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kIbE,jcEA->IjkAbc",
                    t2ab[k0:k1, i0:i1, :, :],
                    eris_ovVV[j0:j1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kjbe,IAec->IjkAbc",
                    t2aa[k0:k1, j0:j1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "mIbA,jckm->IjkAbc",
                    t2ab[:, i0:i1, :, :],
                    eris_ovoo[j0:j1, :, k0:k1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kMbA,jcIM->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovOO[j0:j1, :, i0:i1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kmbc,IAjm->IjkAbc",
                    t2aa[k0:k1, :, :, :],
                    eris_OVoo[i0:i1, :, j0:j1:, :],
                    out=w_blk,
                )

                w_blk = _w_einsum(
                    "kIeA,jbec->IjkAbc",
                    t2ab[k0:k1, i0:i1, :, :],
                    eris_ovvv[j0:j1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kIcE,jbEA->IjkAbc",
                    t2ab[k0:k1, i0:i1, :, :],
                    eris_ovVV[j0:j1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kjce,IAeb->IjkAbc",
                    t2aa[k0:k1, j0:j1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "mIcA,jbkm->IjkAbc",
                    t2ab[:, i0:i1, :, :],
                    eris_ovoo[j0:j1, :, k0:k1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kMcA,jbIM->IjkAbc",
                    t2ab[k0:k1, :, :, :],
                    eris_ovOO[j0:j1, :, i0:i1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kmcb,IAjm->IjkAbc",
                    t2aa[k0:k1, :, :, :],
                    eris_OVoo[i0:i1, :, j0:j1:, :],
                    out=w_blk, subtract=True,
                )

                w_blk = _w_einsum(
                    "jIeA,kbec->IjkAbc",
                    t2ab[j0:j1, i0:i1, :, :],
                    eris_ovvv[k0:k1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jIcE,kbEA->IjkAbc",
                    t2ab[j0:j1, i0:i1, :, :],
                    eris_ovVV[k0:k1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jkce,IAeb->IjkAbc",
                    t2aa[j0:j1, k0:k1, :, :],
                    eris_OVvv[i0:i1, :, :, :],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "mIcA,kbjm->IjkAbc",
                    t2ab[:, i0:i1, :, :],
                    eris_ovoo[k0:k1, :, j0:j1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jMcA,kbIM->IjkAbc",
                    t2ab[j0:j1, :, :, :],
                    eris_ovOO[k0:k1, :, i0:i1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jmcb,IAkm->IjkAbc",
                    t2aa[j0:j1, :, :, :],
                    eris_OVoo[i0:i1, :, k0:k1:, :],
                    out=w_blk,
                )
                rw = _mixed_spin_divide(
                    w_blk, None, eIA[i0:i1, :], eia[j0:j1, :], eia[k0:k1, :]
                )
                gvv += parallel_einsum("ijkcad,ijkcbd->ab", wvd, rw) * 0.25
                gvv += parallel_einsum("ijkcda,ijkcdb->ab", wvd, rw) * 0.25
                gVV += parallel_einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.25

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
    for c0, c1 in lib.prange(0, nvira, blksize):
        for k0, k1 in lib.prange(0, nocca, blksize):
            w_blk = _w_einsum(
                "jIeA,kceb->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovvv[k0:k1, c0:c1, :, :],
                scale=2,
            )
            w_blk = _w_einsum(
                "jIbE,kcEA->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovVV[k0:k1, c0:c1, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "jkbe,IAec->IjkAbc",
                t2aa[:, k0:k1, :, :],
                eris_OVvv[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "mIbA,kcjm->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovoo[k0:k1, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jMbA,kcIM->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovOO[k0:k1, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jmbc,IAkm->IjkAbc",
                t2aa[:, :, :, c0:c1],
                eris_OVoo[:, :, k0:k1, :],
                out=w_blk, subtract=True,
            )
            v_blk = _run_fused(
                _mixed_spin_v,
                eris_ovov[:, :, k0:k1, c0:c1],
                t1b[:, :],
                eris_ovOV[k0:k1, c0:c1, :, :],
                t1a[:, :],
                t2aa[:, k0:k1, :, c0:c1],
                fVO[:, :],
                t2ab[k0:k1, :, c0:c1, :],
                fvo[:, :],
            )
            wvd = _mixed_spin_divide(
                w_blk, v_blk, eIA[:, :], eia[:, :], eia[k0:k1, c0:c1], 1
            )

            w_blk = _w_einsum(
                "kIeA,jceb->IjkAbc",
                t2ab[k0:k1, :, :, :],
                eris_ovvv[:, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kIbE,jcEA->IjkAbc",
                t2ab[k0:k1, :, :, :],
                eris_ovVV[:, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kjbe,IAec->IjkAbc",
                t2aa[k0:k1, :, :, :],
                eris_OVvv[:, :, :, c0:c1],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mIbA,jckm->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovoo[:, c0:c1, k0:k1, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kMbA,jcIM->IjkAbc",
                t2ab[k0:k1, :, :, :],
                eris_ovOO[:, c0:c1, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kmbc,IAjm->IjkAbc",
                t2aa[k0:k1, :, :, c0:c1],
                eris_OVoo[:, :, :, :],
                out=w_blk,
            )

            w_blk = _w_einsum(
                "kIeA,jbec->IjkAbc",
                t2ab[k0:k1, :, :, :],
                eris_ovvv[:, :, :, c0:c1],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kIcE,jbEA->IjkAbc",
                t2ab[k0:k1, :, c0:c1, :],
                eris_ovVV[:, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kjce,IAeb->IjkAbc",
                t2aa[k0:k1, :, c0:c1, :],
                eris_OVvv[:, :, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "mIcA,jbkm->IjkAbc",
                t2ab[:, :, c0:c1, :],
                eris_ovoo[:, :, k0:k1, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kMcA,jbIM->IjkAbc",
                t2ab[k0:k1, :, c0:c1, :],
                eris_ovOO[:, :, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kmcb,IAjm->IjkAbc",
                t2aa[k0:k1, :, c0:c1, :],
                eris_OVoo[:, :, :, :],
                out=w_blk, subtract=True,
            )

            w_blk = _w_einsum(
                "jIeA,kbec->IjkAbc",
                t2ab[:, :, :, :],
                eris_ovvv[k0:k1, :, :, c0:c1],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jIcE,kbEA->IjkAbc",
                t2ab[:, :, c0:c1, :],
                eris_ovVV[k0:k1, :, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jkce,IAeb->IjkAbc",
                t2aa[:, k0:k1, c0:c1, :],
                eris_OVvv[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "mIcA,kbjm->IjkAbc",
                t2ab[:, :, c0:c1, :],
                eris_ovoo[k0:k1, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "jMcA,kbIM->IjkAbc",
                t2ab[:, :, c0:c1, :],
                eris_ovOO[k0:k1, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "jmcb,IAkm->IjkAbc",
                t2aa[:, :, c0:c1, :],
                eris_OVoo[:, :, k0:k1, :],
                out=w_blk,
            )
            rw = _mixed_spin_divide(
                w_blk, None, eIA[:, :], eia[:, :], eia[k0:k1, c0:c1]
            )
            gvo += parallel_einsum("kica,ijkabc->bj", t2ab[k0:k1, :, c0:c1, :], rw) * 0.5
            gVO += parallel_einsum("jkbc,ijkabc->ai", t2aa[:, k0:k1, :, c0:c1], rw) * 0.125

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
    for a0, a1 in lib.prange(0, nvira, blksize):
        for b0, b1 in lib.prange(0, nvirb, blksize):
            for c0, c1 in lib.prange(0, nvirb, blksize):
                w_blk = _w_einsum(
                    "ijae,kceb->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, c0:c1, :, b0:b1],
                    scale=2,
                )
                w_blk = _w_einsum(
                    "ijeb,kcea->ijkabc",
                    t2ab[:, :, :, b0:b1],
                    eris_OVvv[:, c0:c1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "imab,kcjm->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVOO[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "mjab,kcim->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVoo[:, c0:c1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )
                v_blk = _run_fused(
                    _mixed_spin_v,
                    eris_OVOV[:, b0:b1, :, c0:c1],
                    t1a[:, a0:a1],
                    eris_ovOV[:, a0:a1, :, c0:c1].transpose(2, 3, 0, 1),
                    t1b[:, b0:b1],
                    t2bb[:, :, b0:b1, c0:c1],
                    fvo[a0:a1, :],
                    t2ab[:, :, a0:a1, c0:c1].transpose(1, 0, 3, 2),
                    fVO[b0:b1, :],
                )
                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1], 1
                )

                w_blk = _w_einsum(
                    "ikae,jceb->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, c0:c1, :, b0:b1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "ikeb,jcea->ijkabc",
                    t2ab[:, :, :, b0:b1],
                    eris_OVvv[:, c0:c1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[:, :, b0:b1, :],
                    eris_ovVV[:, a0:a1, :, c0:c1],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "imab,jckm->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVOO[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "mkab,jcim->ijkabc",
                    t2ab[:, :, a0:a1, b0:b1],
                    eris_OVoo[:, c0:c1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[:, :, b0:b1, c0:c1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk,
                )

                w_blk = _w_einsum(
                    "ikae,jbec->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, b0:b1, :, c0:c1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "ikec,jbea->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVvv[:, b0:b1, :, a0:a1],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "imac,jbkm->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVOO[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "mkac,jbim->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVoo[:, b0:b1, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk, subtract=True,
                )

                w_blk = _w_einsum(
                    "ijae,kbec->ijkabc",
                    t2ab[:, :, a0:a1, :],
                    eris_OVVV[:, b0:b1, :, c0:c1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "ijec,kbea->ijkabc",
                    t2ab[:, :, :, c0:c1],
                    eris_OVvv[:, b0:b1, :, a0:a1],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[:, :, c0:c1, :],
                    eris_ovVV[:, a0:a1, :, b0:b1],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "imac,kbjm->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVOO[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "mjac,kbim->ijkabc",
                    t2ab[:, :, a0:a1, c0:c1],
                    eris_OVoo[:, b0:b1, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[:, :, c0:c1, b0:b1],
                    eris_ovOO[:, a0:a1, :, :],
                    out=w_blk,
                )
                rw = _mixed_spin_divide(
                    w_blk, None, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                goo += parallel_einsum("iklabc,jklabc->ij", wvd, rw) * 0.25
                gOO += parallel_einsum("kilabc,kjlabc->ij", wvd, rw) * 0.25
                gOO += parallel_einsum("kliabc,kljabc->ij", wvd, rw) * 0.25

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
    for i0, i1 in lib.prange(0, nocca, blksize):
        for j0, j1 in lib.prange(0, noccb, blksize):
            for k0, k1 in lib.prange(0, noccb, blksize):
                w_blk = _w_einsum(
                    "ijae,kceb->ijkabc",
                    t2ab[i0:i1, j0:j1, :, :],
                    eris_OVVV[k0:k1, :, :, :],
                    scale=2,
                )
                w_blk = _w_einsum(
                    "ijeb,kcea->ijkabc",
                    t2ab[i0:i1, j0:j1, :, :],
                    eris_OVvv[k0:k1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jkbe,iaec->ijkabc",
                    t2bb[j0:j1, k0:k1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "imab,kcjm->ijkabc",
                    t2ab[i0:i1, :, :, :],
                    eris_OVOO[k0:k1, :, j0:j1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "mjab,kcim->ijkabc",
                    t2ab[:, j0:j1, :, :],
                    eris_OVoo[k0:k1, :, i0:i1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jmbc,iakm->ijkabc",
                    t2bb[j0:j1, :, :, :],
                    eris_ovOO[i0:i1, :, k0:k1, :],
                    out=w_blk, subtract=True,
                )
                v_blk = _run_fused(
                    _mixed_spin_v,
                    eris_OVOV[j0:j1, :, k0:k1, :],
                    t1a[i0:i1, :],
                    eris_ovOV[i0:i1, :, k0:k1, :].transpose(2, 3, 0, 1),
                    t1b[j0:j1, :],
                    t2bb[j0:j1, k0:k1, :, :],
                    fvo[:, i0:i1],
                    t2ab[i0:i1, k0:k1, :, :].transpose(1, 0, 3, 2),
                    fVO[:, j0:j1],
                )
                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eia[i0:i1, :], eIA[j0:j1, :], eIA[k0:k1, :], 1
                )

                w_blk = _w_einsum(
                    "ikae,jceb->ijkabc",
                    t2ab[i0:i1, k0:k1, :, :],
                    eris_OVVV[j0:j1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "ikeb,jcea->ijkabc",
                    t2ab[i0:i1, k0:k1, :, :],
                    eris_OVvv[j0:j1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kjbe,iaec->ijkabc",
                    t2bb[k0:k1, j0:j1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "imab,jckm->ijkabc",
                    t2ab[i0:i1, :, :, :],
                    eris_OVOO[j0:j1, :, k0:k1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "mkab,jcim->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVoo[j0:j1, :, i0:i1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kmbc,iajm->ijkabc",
                    t2bb[k0:k1, :, :, :],
                    eris_ovOO[i0:i1, :, j0:j1, :],
                    out=w_blk,
                )

                w_blk = _w_einsum(
                    "ikae,jbec->ijkabc",
                    t2ab[i0:i1, k0:k1, :, :],
                    eris_OVVV[j0:j1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "ikec,jbea->ijkabc",
                    t2ab[i0:i1, k0:k1, :, :],
                    eris_OVvv[j0:j1, :, :, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "kjce,iaeb->ijkabc",
                    t2bb[k0:k1, j0:j1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                    out=w_blk,
                )
                w_blk = _w_einsum(
                    "imac,jbkm->ijkabc",
                    t2ab[i0:i1, :, :, :],
                    eris_OVOO[j0:j1, :, k0:k1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "mkac,jbim->ijkabc",
                    t2ab[:, k0:k1, :, :],
                    eris_OVoo[j0:j1, :, i0:i1, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "kmcb,iajm->ijkabc",
                    t2bb[k0:k1, :, :, :],
                    eris_ovOO[i0:i1, :, j0:j1, :],
                    out=w_blk, subtract=True,
                )

                w_blk = _w_einsum(
                    "ijae,kbec->ijkabc",
                    t2ab[i0:i1, j0:j1, :, :],
                    eris_OVVV[k0:k1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "ijec,kbea->ijkabc",
                    t2ab[i0:i1, j0:j1, :, :],
                    eris_OVvv[k0:k1, :, :, :],
                    out=w_blk, subtract=True, scale=2,
                )
                w_blk = _w_einsum(
                    "jkce,iaeb->ijkabc",
                    t2bb[j0:j1, k0:k1, :, :],
                    eris_ovVV[i0:i1, :, :, :],
                    out=w_blk, subtract=True,
                )
                w_blk = _w_einsum(
                    "imac,kbjm->ijkabc",
                    t2ab[i0:i1, :, :, :],
                    eris_OVOO[k0:k1, :, j0:j1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "mjac,kbim->ijkabc",
                    t2ab[:, j0:j1, :, :],
                    eris_OVoo[k0:k1, :, i0:i1, :],
                    out=w_blk, scale=2,
                )
                w_blk = _w_einsum(
                    "jmcb,iakm->ijkabc",
                    t2bb[j0:j1, :, :, :],
                    eris_ovOO[i0:i1, :, k0:k1, :],
                    out=w_blk,
                )
                rw = _mixed_spin_divide(
                    w_blk, None, eia[i0:i1, :], eIA[j0:j1, :], eIA[k0:k1, :]
                )
                gvv += parallel_einsum("ijkacd,ijkbcd->ab", wvd, rw) * 0.25
                gVV += parallel_einsum("ijkcad,ijkcbd->ab", wvd, rw) * 0.25
                gVV += parallel_einsum("ijkcda,ijkcdb->ab", wvd, rw) * 0.25

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
    for c0, c1 in lib.prange(0, nvirb, blksize):
        for k0, k1 in lib.prange(0, noccb, blksize):
            w_blk = _w_einsum(
                "ijae,kceb->ijkabc",
                t2ab,
                eris_OVVV[k0:k1, c0:c1, :, :],
                scale=2,
            )
            w_blk = _w_einsum(
                "ijeb,kcea->ijkabc",
                t2ab,
                eris_OVvv[k0:k1, c0:c1, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "jkbe,iaec->ijkabc",
                t2bb[:, k0:k1, :, :],
                eris_ovVV[:, :, :, c0:c1],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "imab,kcjm->ijkabc",
                t2ab,
                eris_OVOO[k0:k1, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "mjab,kcim->ijkabc",
                t2ab,
                eris_OVoo[k0:k1, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jmbc,iakm->ijkabc",
                t2bb[:, :, :, c0:c1],
                eris_ovOO[:, :, k0:k1, :],
                out=w_blk, subtract=True,
            )
            v_blk = _run_fused(
                _mixed_spin_v,
                eris_OVOV[:, :, k0:k1, c0:c1],
                t1a[:, :],
                eris_ovOV[:, :, k0:k1, c0:c1].transpose(2, 3, 0, 1),
                t1b[:, :],
                t2bb[:, k0:k1, :, c0:c1],
                fvo[:, :],
                t2ab[:, k0:k1, :, c0:c1].transpose(1, 0, 3, 2),
                fVO[:, :],
            )
            wvd = _mixed_spin_divide(
                w_blk, v_blk, eia[:, :], eIA[:, :], eIA[k0:k1, c0:c1], 1
            )

            w_blk = _w_einsum(
                "ikae,jceb->ijkabc",
                t2ab[:, k0:k1, :, :],
                eris_OVVV[:, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "ikeb,jcea->ijkabc",
                t2ab[:, k0:k1, :, :],
                eris_OVvv[:, c0:c1, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kjbe,iaec->ijkabc",
                t2bb[k0:k1, :, :, :],
                eris_ovVV[:, :, :, c0:c1],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "imab,jckm->ijkabc",
                t2ab[:, :, :, :],
                eris_OVOO[:, c0:c1, k0:k1, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "mkab,jcim->ijkabc",
                t2ab[:, k0:k1, :, :],
                eris_OVoo[:, c0:c1, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kmbc,iajm->ijkabc",
                t2bb[k0:k1, :, :, c0:c1],
                eris_ovOO[:, :, :, :],
                out=w_blk,
            )

            w_blk = _w_einsum(
                "ikae,jbec->ijkabc",
                t2ab[:, k0:k1, :, :],
                eris_OVVV[:, :, :, c0:c1],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "ikec,jbea->ijkabc",
                t2ab[:, k0:k1, :, c0:c1],
                eris_OVvv[:, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "kjce,iaeb->ijkabc",
                t2bb[k0:k1, :, c0:c1, :],
                eris_ovVV[:, :, :, :],
                out=w_blk,
            )
            w_blk = _w_einsum(
                "imac,jbkm->ijkabc",
                t2ab[:, :, :, c0:c1],
                eris_OVOO[:, :, k0:k1, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "mkac,jbim->ijkabc",
                t2ab[:, k0:k1, :, c0:c1],
                eris_OVoo[:, :, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "kmcb,iajm->ijkabc",
                t2bb[k0:k1, :, c0:c1, :],
                eris_ovOO[:, :, :, :],
                out=w_blk, subtract=True,
            )

            w_blk = _w_einsum(
                "ijae,kbec->ijkabc",
                t2ab[:, :, :, :],
                eris_OVVV[k0:k1, :, :, c0:c1],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "ijec,kbea->ijkabc",
                t2ab[:, :, :, c0:c1],
                eris_OVvv[k0:k1, :, :, :],
                out=w_blk, subtract=True, scale=2,
            )
            w_blk = _w_einsum(
                "jkce,iaeb->ijkabc",
                t2bb[:, k0:k1, c0:c1, :],
                eris_ovVV[:, :, :, :],
                out=w_blk, subtract=True,
            )
            w_blk = _w_einsum(
                "imac,kbjm->ijkabc",
                t2ab[:, :, :, c0:c1],
                eris_OVOO[k0:k1, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "mjac,kbim->ijkabc",
                t2ab[:, :, :, c0:c1],
                eris_OVoo[k0:k1, :, :, :],
                out=w_blk, scale=2,
            )
            w_blk = _w_einsum(
                "jmcb,iakm->ijkabc",
                t2bb[:, :, c0:c1, :],
                eris_ovOO[:, :, k0:k1, :],
                out=w_blk,
            )
            rw = _mixed_spin_divide(
                w_blk, None, eia[:, :], eIA[:, :], eIA[k0:k1, c0:c1]
            )
            gVO += parallel_einsum("ikac,ijkabc->bj", t2ab[:, k0:k1, :, c0:c1], rw) * 0.5
            gvo += parallel_einsum("jkbc,ijkabc->ai", t2bb[:, k0:k1, :, c0:c1], rw) * 0.125

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
