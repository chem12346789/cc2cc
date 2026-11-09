import numpy
from numba import get_num_threads, njit, prange, set_num_threads
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda
from pyscf.cc import uccsd_lambda

from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum, _w_einsum


def _triples_block_size(max_memory, extent, denominator, exponent):
    """Keep the loop step positive when a spin sector is empty."""
    if not denominator:
        return 1
    return max(
        1,
        min(
            extent,
            int(((max_memory * 0.9e6 / 8) / 6.0 / denominator) ** exponent),
        ),
    )


@njit(parallel=True, cache=False)
def _same_spin_v(ovov, t1, t2, fvo, b0, b1, c0, c1):
    nocc, nvir = t1.shape
    nb, nc = b1 - b0, c1 - c0
    # Match the ij-a / k-c-b GEMM layout of the connected triples tensor.
    out = numpy.empty((nocc, nc, nb, nocc, nocc, nvir), dtype=t1.dtype)
    for row in prange(nocc * nc * nb):
        b = row % nb
        c = (row // nb) % nc
        k = row // (nb * nc)
        B, C = b0 + b, c0 + c
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    value = ovov[j, B, k, C] * t1[i, a]
                    value += ovov[i, a, j, B] * t1[k, C]
                    value += ovov[k, C, i, a] * t1[j, B]
                    value += ovov[k, C, j, B] * t1[i, a]
                    value += ovov[j, B, i, a] * t1[k, C]
                    value += ovov[i, a, k, C] * t1[j, B]
                    value += (t2[j, k, B, C] * fvo[a, i]) * 0.5
                    value += (t2[i, j, a, B] * fvo[C, k]) * 0.5
                    value += (t2[k, i, C, a] * fvo[B, j]) * 0.5
                    value += (t2[k, j, C, B] * fvo[a, i]) * 0.5
                    value += (t2[j, i, B, a] * fvo[C, k]) * 0.5
                    value += (t2[i, k, a, C] * fvo[B, j]) * 0.5
                    out[k, c, b, i, j, a] = value
    return out.transpose(3, 4, 0, 5, 2, 1)


@njit(parallel=True, cache=False, error_model="numpy")
def _same_spin_r6_divide(w, v, eia, b0, c0, symmetrize=True):
    nocc, _, _, nvir, nb, nc = w.shape
    out = numpy.empty((nocc, nc, nb, nocc, nocc, nvir), dtype=w.dtype)
    for row in prange(nocc * nc * nb):
        b = row % nb
        c = (row // nb) % nc
        k = row // (nb * nc)
        for i in range(nocc):
            for j in range(nocc):
                for a in range(nvir):
                    if v is not None and not symmetrize:
                        denominator = (eia[i, a] + eia[j, b0 + b]) + eia[k, c0 + c]
                        out[k, c, b, i, j, a] = (
                            w[i, j, k, a, b, c] + v[i, j, k, a, b, c]
                        ) / denominator
                        continue
                    x0 = w[i, j, k, a, b, c]
                    x1 = w[j, k, i, a, b, c]
                    x2 = w[k, i, j, a, b, c]
                    x3 = w[k, j, i, a, b, c]
                    x4 = w[i, k, j, a, b, c]
                    x5 = w[j, i, k, a, b, c]
                    if v is not None:
                        x0 = x0 * 2 + v[i, j, k, a, b, c]
                        x1 = x1 * 2 + v[j, k, i, a, b, c]
                        x2 = x2 * 2 + v[k, i, j, a, b, c]
                        x3 = x3 * 2 + v[k, j, i, a, b, c]
                        x4 = x4 * 2 + v[i, k, j, a, b, c]
                        x5 = x5 * 2 + v[j, i, k, a, b, c]
                    denominator = (eia[i, a] + eia[j, b0 + b]) + eia[k, c0 + c]
                    out[k, c, b, i, j, a] = (
                        x0 + x1 + x2 - x3 - x4 - x5
                    ) / denominator
    return out.transpose(3, 4, 0, 5, 2, 1)


@njit(parallel=True, cache=False)
def _mixed_spin_v(
    ovov, t1, ovOV, t1_same, t2_same, f_other, t2_mixed, f_same,
    out=None, subtract=False,
):
    """Accumulate one mixed-spin outer-product group in IjkAbc order."""
    nj, nb, nk, nc = ovov.shape
    ni, na = t1.shape
    if out is None:
        # Match the outer-product layout; subsequent groups use permuted views.
        storage = numpy.empty((nj, nb, nk, nc, ni, na), dtype=ovov.dtype)
        out = storage.transpose(4, 0, 2, 5, 1, 3)
        initialize = True
    else:
        initialize = False
    for row in prange(nj * nb * nk * nc):
        c = row % nc
        k = (row // nc) % nk
        b = (row // (nc * nk)) % nb
        j = row // (nc * nk * nb)
        for i in range(ni):
            for a in range(na):
                first = ovov[j, b, k, c] * t1[i, a]
                repeated = ovOV[k, c, i, a] * t1_same[j, b]
                same = (t2_same[j, k, b, c] * f_other[a, i]) * 0.5
                mixed = (t2_mixed[k, i, c, a] * f_same[b, j]) * 2
                if initialize:
                    value = first
                elif subtract:
                    value = out[i, j, k, a, b, c] - first
                else:
                    value = out[i, j, k, a, b, c] + first
                if subtract:
                    value = value - repeated
                    value = value - repeated
                    value = value - same
                    value = value - mixed
                else:
                    value = value + repeated
                    value = value + repeated
                    value = value + same
                    value = value + mixed
                out[i, j, k, a, b, c] = value
    return out


@njit(parallel=True, cache=False, error_model="numpy")
def _mixed_spin_divide_kernel(w, v, e0, e1, e2, inverse, factor):
    out = numpy.empty_like(w)
    n0, n1, n2, n3, n4, n5 = w.shape
    for row in prange(n0 * n1 * n2):
        x0 = row // (n1 * n2)
        x1 = (row // n2) % n1
        x2 = row % n2
        for x3 in range(n3):
            for x4 in range(n4):
                for x5 in range(n5):
                    index = (x0, x1, x2, x3, x4, x5)
                    denominator = (
                        e0[index[inverse[0]], index[inverse[3]]]
                        + e1[index[inverse[1]], index[inverse[4]]]
                    ) + e2[index[inverse[2]], index[inverse[5]]]
                    value = w[x0, x1, x2, x3, x4, x5]
                    if v is not None:
                        if factor == 2:
                            value = value * 2
                        value = value + v[x0, x1, x2, x3, x4, x5]
                    out[x0, x1, x2, x3, x4, x5] = value / denominator
    return out


def _mixed_spin_divide(w, v, e0, e1, e2, factor=2):
    axes = tuple(sorted(range(6), key=lambda axis: abs(w.strides[axis]), reverse=True))
    inverse = tuple(axes.index(axis) for axis in range(6))
    result = _run_fused(
        _mixed_spin_divide_kernel, w.transpose(axes),
        None if v is None else v.transpose(axes),
        e0, e1, e2, inverse, factor,
    )
    return result.transpose(inverse)


def _run_fused(kernel, *args):
    threads = get_num_threads()
    set_num_threads(min(threads, lib.num_threads()))
    try:
        return kernel(*args)
    finally:
        set_num_threads(threads)


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


def kernel(
    mycc,
    eris=None,
    t1=None,
    t2=None,
    l1=None,
    l2=None,
    max_cycle=50,
    tol=1e-8,
    verbose=logger.INFO,
):
    return ccsd_lambda.kernel(
        mycc,
        eris,
        t1,
        t2,
        l1,
        l2,
        max_cycle,
        tol,
        verbose,
        make_intermediates,
        update_lambda,
    )


def make_intermediates(mycc, t1, t2, eris):
    """Build unrestricted lambda intermediates."""
    log = logger.Logger(mycc.stdout, mycc.verbose)
    imds = uccsd_lambda.make_intermediates(mycc, t1, t2, eris)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    nocca, noccb = t2ab.shape[:2]

    has_alpha = nocca > 0 and nvira > 0
    has_beta = noccb > 0 and nvirb > 0
    has_mixed = has_alpha and has_beta
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

    l1a_t = numpy.zeros((nocca, nvira))
    l2aa_t = numpy.zeros((nocca, nocca, nvira, nvira))

    # aaa
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = _triples_block_size(
        max_memory, nvira, nocca**3 * nvira, 1 / 2
    )
    if blksize < nvira:
        blksize = min(blksize, (nvira + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 1 rdm make_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocca,
        nvira,
        blksize,
    )

    for b0, b1 in lib.prange(0, nvira if has_alpha else 0, blksize):
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
            l1a_t += parallel_einsum(
                "ijkabc,jbkc->ia",
                rw.conj(),
                eris_ovov[:, b0:b1, :, c0:c1],
            )
            wvd = _run_fused(_same_spin_r6_divide, w_blk, v_blk, eia, b0, c0)
            l2vv = parallel_einsum(
                "ijkabc,kceb->ijae", wvd, eris_ovvv.conj()[:, c0:c1, :, b0:b1]
            )
            l2aa_t += l2vv
            l2oo = parallel_einsum(
                "ijkabc,iajm->mkbc", wvd, eris_ovoo.conj()
            )
            l2aa_t[:, :, b0:b1, c0:c1] -= l2oo
            l2aa_t += l2vv.transpose(1, 0, 3, 2)
            l2aa_t[:, :, c0:c1, b0:b1] -= l2oo.transpose(1, 0, 3, 2)
            del l2vv, l2oo

            l2aa_t[:, :, b0:b1, c0:c1] += parallel_einsum("ijkabc,ai->jkbc", rw, fvo)
    imds.l1a_t = l1a_t / eia * 0.25
    imds.l2aa_t = l2aa_t.conj() / lib.direct_sum("ia+jb->ijab", eia, eia) * 0.5

    # bbb
    l1b_t = numpy.zeros((noccb, nvirb))
    l2bb_t = numpy.zeros((noccb, noccb, nvirb, nvirb))

    blksize = _triples_block_size(
        max_memory, nvirb, noccb**3 * nvirb, 1 / 2
    )
    if blksize < nvirb:
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm make_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for b0, b1 in lib.prange(0, nvirb if has_beta else 0, blksize):
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
            l1b_t += parallel_einsum(
                "ijkabc,jbkc->ia",
                rw.conj(),
                eris_OVOV[:, b0:b1, :, c0:c1],
            )
            wvd = _run_fused(_same_spin_r6_divide, w_blk, v_blk, eIA, b0, c0)
            l2vv = parallel_einsum(
                "ijkabc,kceb->ijae",
                wvd,
                eris_OVVV.conj()[:, c0:c1, :, b0:b1],
            )
            l2bb_t += l2vv
            l2oo = parallel_einsum(
                "ijkabc,iajm->mkbc",
                wvd,
                eris_OVOO.conj(),
            )
            l2bb_t[:, :, b0:b1, c0:c1] -= l2oo
            l2bb_t += l2vv.transpose(1, 0, 3, 2)
            l2bb_t[:, :, c0:c1, b0:b1] -= l2oo.transpose(1, 0, 3, 2)
            del l2vv, l2oo
            l2bb_t[:, :, b0:b1, c0:c1] += parallel_einsum(
                "ijkabc,ai->jkbc",
                rw,
                fVO,
            )

    imds.l1b_t = l1b_t / eIA * 0.25
    imds.l2bb_t = l2bb_t.conj() / lib.direct_sum("ia+jb->ijab", eIA, eIA) * 0.5

    # baa
    l1a_t = numpy.zeros((nocca, nvira))
    l1b_t = numpy.zeros((noccb, nvirb))
    l2aa_t = numpy.zeros((nocca, nocca, nvira, nvira))
    l2ab_t = numpy.zeros((nocca, noccb, nvira, nvirb))

    blksize = _triples_block_size(
        max_memory, nvirb, nocca**2 * noccb, 1 / 3
    )
    blksize = max(1, min(nvira, blksize))
    if blksize < nvirb:
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm make_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for a0, a1 in lib.prange(0, nvirb if has_mixed else 0, blksize):
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

                _run_fused(
                    _mixed_spin_v,
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
                _run_fused(
                    _mixed_spin_v,
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
                _run_fused(
                    _mixed_spin_v,
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

                rw = _mixed_spin_divide(
                    w_blk, None, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )
                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eIA[:, a0:a1], eia[:, b0:b1], eia[:, c0:c1]
                )

                l1a_t[:, b0:b1] += parallel_einsum(
                    "ijkabc,kcia->jb",
                    rw.conj(),
                    eris_ovOV[:, c0:c1, :, a0:a1],
                )
                l1b_t[:, a0:a1] += parallel_einsum(
                    "ijkabc,jbkc->ia",
                    rw.conj(),
                    eris_ovov[:, b0:b1, :, c0:c1],
                )
                l2vv = parallel_einsum(
                    "ijkabc,iaec->jkbe",
                    wvd,
                    eris_OVvv.conj()[:, a0:a1, :, c0:c1],
                )
                l2aa_t[:, :, b0:b1, :] += l2vv
                l2oo = parallel_einsum(
                    "ijkabc,iakm->jmbc",
                    wvd,
                    eris_OVoo.conj()[:, a0:a1, :, :],
                )
                l2aa_t[:, :, b0:b1, c0:c1] -= l2oo
                l2aa_t[:, :, :, b0:b1] += l2vv.transpose(1, 0, 3, 2)
                l2aa_t[:, :, c0:c1, b0:b1] -= l2oo.transpose(1, 0, 3, 2)
                del l2vv, l2oo
                l2aa_t[:, :, b0:b1, c0:c1] += parallel_einsum(
                    "ijkabc,ai->jkbc",
                    rw,
                    fVO[a0:a1, :],
                )

                l2ab_t[:, :, :, a0:a1] += parallel_einsum(
                    "ijkabc,kceb->jiea",
                    wvd,
                    eris_ovvv.conj()[:, c0:c1, :, b0:b1],
                )
                l2ab_t[:, :, b0:b1, :] += parallel_einsum(
                    "ijkabc,kcea->jibe",
                    wvd,
                    eris_ovVV.conj()[:, c0:c1, :, a0:a1],
                )
                l2ab_t[:, :, b0:b1, a0:a1] -= parallel_einsum(
                    "ijkabc,kcjm->miba",
                    wvd,
                    eris_ovoo.conj()[:, c0:c1, :, :],
                )
                l2ab_t[:, :, b0:b1, a0:a1] -= parallel_einsum(
                    "ijkabc,kcim->jmba",
                    wvd,
                    eris_ovOO.conj()[:, c0:c1, :, :],
                )
                l2ab_t[:, :, c0:c1, a0:a1] += parallel_einsum(
                    "ijkabc,bj->kica",
                    rw,
                    fvo[b0:b1, :],
                )

    imds.l1a_t += l1a_t / eia * 0.5
    imds.l1b_t += l1b_t / eIA * 0.25
    imds.l2aa_t += l2aa_t.conj() / lib.direct_sum("ia+jb->ijab", eia, eia) * 0.5
    imds.l2ab_t = l2ab_t.conj() / lib.direct_sum("ia+jb->ijab", eia, eIA) * 0.5

    # bba
    l1a_t = numpy.zeros((nocca, nvira))
    l1b_t = numpy.zeros((noccb, nvirb))
    l2bb_t = numpy.zeros((noccb, noccb, nvirb, nvirb))
    l2ab_t = numpy.zeros((nocca, noccb, nvira, nvirb))

    blksize = _triples_block_size(
        max_memory, nvirb, noccb**2 * nocca, 1 / 3
    )
    blksize = max(1, min(nvira, blksize))
    if blksize < nvira or blksize < nvirb:
        blksize = min(blksize, (nvira + 1) // 2)
        blksize = min(blksize, (nvirb + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t spin 2 rdm make_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        noccb,
        nvirb,
        blksize,
    )
    for a0, a1 in lib.prange(0, nvira if has_mixed else 0, blksize):
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

                _run_fused(
                    _mixed_spin_v,
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
                _run_fused(
                    _mixed_spin_v,
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
                _run_fused(
                    _mixed_spin_v,
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

                rw = _mixed_spin_divide(
                    w_blk, None, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                wvd = _mixed_spin_divide(
                    w_blk, v_blk, eia[:, a0:a1], eIA[:, b0:b1], eIA[:, c0:c1]
                )
                l1a_t[:, a0:a1] += parallel_einsum(
                    "ijkabc,jbkc->ia",
                    rw.conj(),
                    eris_OVOV[:, b0:b1, :, c0:c1],
                )
                l1b_t[:, b0:b1] += parallel_einsum(
                    "ijkabc,iakc->jb",
                    rw.conj(),
                    eris_ovOV[:, a0:a1, :, c0:c1],
                )

                l2vv = parallel_einsum(
                    "ijkabc,iaec->jkbe",
                    wvd,
                    eris_ovVV[:, a0:a1, :, c0:c1],
                )
                l2bb_t[:, :, b0:b1, :] += l2vv
                l2oo = parallel_einsum(
                    "ijkabc,iakm->jmbc",
                    wvd,
                    eris_ovOO[:, a0:a1, :, :],
                )
                l2bb_t[:, :, b0:b1, c0:c1] -= l2oo
                l2bb_t[:, :, :, b0:b1] += l2vv.transpose(1, 0, 3, 2)
                l2bb_t[:, :, c0:c1, b0:b1] -= l2oo.transpose(1, 0, 3, 2)
                del l2vv, l2oo
                l2bb_t[:, :, b0:b1, c0:c1] += parallel_einsum(
                    "ijkabc,ai->jkbc",
                    rw,
                    fvo[a0:a1, :],
                )

                l2ab_t[:, :, a0:a1, :] += parallel_einsum(
                    "ijkabc,kceb->ijae",
                    wvd,
                    eris_OVVV.conj()[:, c0:c1, :, b0:b1],
                )
                l2ab_t[:, :, :, b0:b1] += parallel_einsum(
                    "ijkabc,kcea->ijeb",
                    wvd,
                    eris_OVvv.conj()[:, c0:c1, :, a0:a1],
                )
                l2ab_t[:, :, a0:a1, b0:b1] -= parallel_einsum(
                    "ijkabc,kcjm->imab",
                    wvd,
                    eris_OVOO.conj()[:, c0:c1, :, :],
                )
                l2ab_t[:, :, a0:a1, b0:b1] -= parallel_einsum(
                    "ijkabc,kcim->mjab",
                    wvd,
                    eris_OVoo.conj()[:, c0:c1, :, :],
                )
                l2ab_t[:, :, a0:a1, c0:c1] += parallel_einsum(
                    "ijkabc,bj->ikac", rw, fVO[b0:b1, :]
                )

    imds.l1a_t += l1a_t / eia * 0.25
    imds.l1b_t += l1b_t / eIA * 0.5
    imds.l2bb_t += l2bb_t.conj() / lib.direct_sum("ia+jb->ijab", eIA, eIA) * 0.5
    imds.l2ab_t += l2ab_t.conj() / lib.direct_sum("ia+jb->ijab", eia, eIA) * 0.5

    return imds


def update_lambda(mycc, t1, t2, l1, l2, eris=None, imds=None):
    if eris is None:
        eris = mycc.ao2mo()
    if imds is None:
        imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = uccsd_lambda.update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    l1a += imds.l1a_t
    l1b += imds.l1b_t
    l2aa += imds.l2aa_t
    l2ab += imds.l2ab_t
    l2bb += imds.l2bb_t
    return (l1a, l1b), (l2aa, l2ab, l2bb)
