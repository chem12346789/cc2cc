#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Yu Jin <jinyuchem@uchicago.edu>

import ctypes
import numpy
from numba import njit, prange
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_rdm, _ccsd

from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum, _w_einsum
from cc2cc.utils.ccsd_t.pyscf_uccsd_t_lambda import _run_fused


@njit(parallel=True, cache=False)
def _restricted_v_divide(
    w, v, ovov, t1, t2, fvo, eia, start0, start1, start2, occupied, w_factor
):
    n0, n1, n2, n3, n4, n5 = w.shape
    for row in prange(n0 * n1 * n2):
        x = row // (n1 * n2)
        y = (row // n2) % n1
        z = row % n2
        for p in range(n3):
            if occupied:
                i, a = start0 + x, p
            else:
                i, a = p, start0 + x
            for q in range(n4):
                if occupied:
                    j, b = start1 + y, q
                else:
                    j, b = q, start1 + y
                for r in range(n5):
                    if occupied:
                        k, c = start2 + z, r
                    else:
                        k, c = r, start2 + z
                    value = ovov[i, a, j, b] * t1[k, c]
                    value += ovov[i, a, k, c] * t1[j, b]
                    value += ovov[j, b, k, c] * t1[i, a]
                    value += fvo[c, k] * t2[i, j, a, b]
                    value += fvo[a, i] * t2[j, k, b, c]
                    value += fvo[b, j] * t2[k, i, c, a]
                    denominator = (eia[i, a] + eia[j, b]) + eia[k, c]
                    w_value = w[x, y, z, p, q, r] / denominator
                    v[x, y, z, p, q, r] = value / denominator + w_factor * w_value
                    w[x, y, z, p, q, r] = w_value


def _gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None):
    """Build restricted 1-RDM intermediates with fused triples updates."""
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if (
        numpy.iscomplexobj(t1)
        or numpy.iscomplexobj(t2)
        or numpy.iscomplexobj(eris)
        or numpy.iscomplexobj(l1)
        or numpy.iscomplexobj(l2)
    ):
        raise ValueError(
            "_gamma1_intermediates does not support complex-valued inputs (t1, t2, l1, l2, or eris)"
        )

    doo, dov, dvo, dvv = ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None:
        eris = mycc.ao2mo()
    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum("i-a->ia", mo_e[:nocc], mo_e[nocc:])

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocc**3)) ** (1 / 3)))
    if blksize < nvir:
        blksize = min(blksize, (nvir + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocc,
        nvir,
        blksize,
    )

    goo = numpy.zeros((nocc, nocc), dtype=t1.dtype)
    w_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)
    v_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)

    for a0, a1 in lib.prange(0, nvir, blksize):
        ba = a1 - a0
        for b0, b1 in lib.prange(0, nvir, blksize):
            bb = b1 - b0
            for c0, c1 in lib.prange(0, nvir, blksize):
                bc = c1 - c0

                w_blk[:ba, :bb, :bc] = parallel_einsum(
                    "iafb,kjcf->abcijk",
                    eris_ovvv[:, a0:a1, :, b0:b1],
                    t2[:, :, c0:c1, :],
                )
                _w_einsum(
                    "iafc,jkbf->abcijk",
                    eris_ovvv[:, a0:a1, :, c0:c1],
                    t2[:, :, b0:b1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "jbfa,kicf->abcijk",
                    eris_ovvv[:, b0:b1, :, a0:a1],
                    t2[:, :, c0:c1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "jbfc,ikaf->abcijk",
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    t2[:, :, a0:a1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "kcfa,jibf->abcijk",
                    eris_ovvv[:, c0:c1, :, a0:a1],
                    t2[:, :, b0:b1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "kcfb,ijaf->abcijk",
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    t2[:, :, a0:a1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "iajm,mkbc->abcijk",
                    eris_ovoo[:, a0:a1, :, :],
                    t2[:, :, b0:b1, c0:c1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "iakm,mjcb->abcijk",
                    eris_ovoo[:, a0:a1, :, :],
                    t2[:, :, c0:c1, b0:b1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "jbim,mkac->abcijk",
                    eris_ovoo[:, b0:b1, :, :],
                    t2[:, :, a0:a1, c0:c1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "jbkm,mica->abcijk",
                    eris_ovoo[:, b0:b1, :, :],
                    t2[:, :, c0:c1, a0:a1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "kcim,mjab->abcijk",
                    eris_ovoo[:, c0:c1, :, :],
                    t2[:, :, a0:a1, b0:b1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "kcjm,miba->abcijk",
                    eris_ovoo[:, c0:c1, :, :],
                    t2[:, :, b0:b1, a0:a1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )

                _run_fused(
                    _restricted_v_divide,
                    w_blk[:ba, :bb, :bc], v_blk[:ba, :bb, :bc],
                    eris_ovov, t1, t2, eris.fock[nocc:, :nocc], eia,
                    a0, b0, c0, False, 1.0,
                )

                t3_symm_ip_py(w_blk, blksize**3, nocc, "4-2-211-2", 1.0, 0.0)
                goo[:, :] += parallel_einsum(
                    "abcikl,abcjkl->ij",
                    v_blk[:ba, :bb, :bc, :, :, :],
                    w_blk[:ba, :bb, :bc, :, :, :],
                )


    w_blk = None
    v_blk = None

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nocc, int(((max_memory * 0.9e6 / 8) / 6.0 / (nvir**3)) ** (1 / 3)))
    if blksize < nocc:
        blksize = min(blksize, (nocc + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma1_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocc,
        nvir,
        blksize,
    )

    gvv = numpy.zeros((nvir, nvir), dtype=t1.dtype)
    w_blk = numpy.empty((blksize, blksize, blksize, nvir, nvir, nvir), dtype=t1.dtype)
    v_blk = numpy.empty((blksize, blksize, blksize, nvir, nvir, nvir), dtype=t1.dtype)

    for k0, k1 in lib.prange(0, nocc, blksize):
        bk = k1 - k0
        for j0, j1 in lib.prange(0, nocc, blksize):
            bj = j1 - j0
            for i0, i1 in lib.prange(0, nocc, blksize):
                bi = i1 - i0

                w_blk[:bi, :bj, :bk] = parallel_einsum(
                    "iafb,kjcf->ijkabc",
                    eris_ovvv[i0:i1, :, :, :],
                    t2[k0:k1, j0:j1, :, :],
                )
                _w_einsum(
                    "iafc,jkbf->ijkabc",
                    eris_ovvv[i0:i1, :, :, :],
                    t2[j0:j1, k0:k1, :, :],
                    out=w_blk[:bi, :bj, :bk],
                )
                _w_einsum(
                    "jbfa,kicf->ijkabc",
                    eris_ovvv[j0:j1, :, :, :],
                    t2[k0:k1, i0:i1, :, :],
                    out=w_blk[:bi, :bj, :bk],
                )
                _w_einsum(
                    "jbfc,ikaf->ijkabc",
                    eris_ovvv[j0:j1, :, :, :],
                    t2[i0:i1, k0:k1, :, :],
                    out=w_blk[:bi, :bj, :bk],
                )
                _w_einsum(
                    "kcfa,jibf->ijkabc",
                    eris_ovvv[k0:k1, :, :, :],
                    t2[j0:j1, i0:i1, :, :],
                    out=w_blk[:bi, :bj, :bk],
                )
                _w_einsum(
                    "kcfb,ijaf->ijkabc",
                    eris_ovvv[k0:k1, :, :, :],
                    t2[i0:i1, j0:j1, :, :],
                    out=w_blk[:bi, :bj, :bk],
                )
                _w_einsum(
                    "iajm,mkbc->ijkabc",
                    eris_ovoo[i0:i1, :, j0:j1, :],
                    t2[:, k0:k1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )
                _w_einsum(
                    "iakm,mjcb->ijkabc",
                    eris_ovoo[i0:i1, :, k0:k1, :],
                    t2[:, j0:j1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )
                _w_einsum(
                    "jbim,mkac->ijkabc",
                    eris_ovoo[j0:j1, :, i0:i1, :],
                    t2[:, k0:k1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )
                _w_einsum(
                    "jbkm,mica->ijkabc",
                    eris_ovoo[j0:j1, :, k0:k1, :],
                    t2[:, i0:i1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )
                _w_einsum(
                    "kcim,mjab->ijkabc",
                    eris_ovoo[k0:k1, :, i0:i1, :],
                    t2[:, j0:j1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )
                _w_einsum(
                    "kcjm,miba->ijkabc",
                    eris_ovoo[k0:k1, :, j0:j1, :],
                    t2[:, i0:i1, :, :],
                    out=w_blk[:bi, :bj, :bk], subtract=True,
                )

                _run_fused(
                    _restricted_v_divide,
                    w_blk[:bi, :bj, :bk], v_blk[:bi, :bj, :bk],
                    eris_ovov, t1, t2, eris.fock[nocc:, :nocc], eia,
                    i0, j0, k0, True, 1.0,
                )

                t3_symm_ip_py(w_blk, blksize**3, nvir, "4-2-211-2", 1.0, 0.0)

                gvv[:, :] += parallel_einsum(
                    "ijkacd,ijkbcd->ab",
                    v_blk[:bi, :bj, :bk, :, :, :],
                    w_blk[:bi, :bj, :bk, :, :, :],
                )
                dvo[:, k0:k1] += 0.5 * parallel_einsum(
                    "ijab,ijkabc->ck",
                    t2[i0:i1, j0:j1, :, :],
                    w_blk[:bi, :bj, :bk, :, :, :],
                )


    w_blk = None
    v_blk = None

    # if not for_grad:
    #     # t3 amplitudes in CCSD(T) is computed non-iteratively. The
    #     # off-diagonal blocks of fock matrix does not contribute to CCSD(T)
    #     # energy. To make Tr(H,D) consistent to the CCSD(T) total energy, the
    #     # density matrix off-diagonal parts are excluded.
    #     doo[numpy.diag_indices(nocc)] -= goo.diagonal() * 0.5
    #     dvv[numpy.diag_indices(nvir)] += gvv.diagonal() * 0.5

    # else:
    #     # The off-diagonal blocks of fock matrix have small contributions to
    #     # analytical nuclear gradients.
    #     doo -= goo * 0.5
    #     dvv += gvv * 0.5

    return (doo, dov, dvo, dvv), goo, gvv


def _gamma2_intermediates(mycc, t1, t2, l1, l2, eris=None, compress_vvvv=False):
    """Build Chemist-ordered 2-RDM intermediates with fused triples updates."""
    log = logger.Logger(mycc.stdout, mycc.verbose)

    if (
        numpy.iscomplexobj(t1)
        or numpy.iscomplexobj(t2)
        or numpy.iscomplexobj(eris)
        or numpy.iscomplexobj(l1)
        or numpy.iscomplexobj(l2)
    ):
        raise ValueError(
            "_gamma2_intermediates does not support complex-valued inputs (t1, t2, l1, l2, or eris)"
        )

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = (
        ccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
    )
    if eris is None:
        eris = mycc.ao2mo()

    nocc, nvir = t1.shape
    eris_ovvv = numpy.asarray(eris.get_ovvv())
    eris_ovoo = numpy.asarray(eris.ovoo)
    eris_ovov = numpy.asarray(eris.ovov)

    mo_e = eris.mo_energy
    eia = lib.direct_sum("i-a->ia", mo_e[:nocc], mo_e[nocc:])

    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)
    blksize = min(nvir, int(((max_memory * 0.9e6 / 8) / 6.0 / (nocc**3)) ** (1 / 3)))
    if blksize < nvir:
        blksize = min(blksize, (nvir + 1) // 2)
        blksize = max(blksize, 1)
    log.debug1(
        "ccsd_t rdm _gamma2_intermediates: max_memory %d MB,  nocc,nvir = %d,%d  blksize = %d",
        max_memory,
        nocc,
        nvir,
        blksize,
    )

    w_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)
    v_blk = numpy.empty((blksize, blksize, blksize, nocc, nocc, nocc), dtype=t1.dtype)

    for c0, c1 in lib.prange(0, nvir, blksize):
        bc = c1 - c0
        for b0, b1 in lib.prange(0, nvir, blksize):
            bb = b1 - b0
            for a0, a1 in lib.prange(0, nvir, blksize):
                ba = a1 - a0

                w_blk[:ba, :bb, :bc] = parallel_einsum(
                    "iafb,kjcf->abcijk",
                    eris_ovvv[:, a0:a1, :, b0:b1],
                    t2[:, :, c0:c1, :],
                )
                _w_einsum(
                    "iafc,jkbf->abcijk",
                    eris_ovvv[:, a0:a1, :, c0:c1],
                    t2[:, :, b0:b1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "jbfa,kicf->abcijk",
                    eris_ovvv[:, b0:b1, :, a0:a1],
                    t2[:, :, c0:c1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "jbfc,ikaf->abcijk",
                    eris_ovvv[:, b0:b1, :, c0:c1],
                    t2[:, :, a0:a1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "kcfa,jibf->abcijk",
                    eris_ovvv[:, c0:c1, :, a0:a1],
                    t2[:, :, b0:b1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "kcfb,ijaf->abcijk",
                    eris_ovvv[:, c0:c1, :, b0:b1],
                    t2[:, :, a0:a1, :],
                    out=w_blk[:ba, :bb, :bc],
                )
                _w_einsum(
                    "iajm,mkbc->abcijk",
                    eris_ovoo[:, a0:a1, :, :],
                    t2[:, :, b0:b1, c0:c1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "iakm,mjcb->abcijk",
                    eris_ovoo[:, a0:a1, :, :],
                    t2[:, :, c0:c1, b0:b1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "jbim,mkac->abcijk",
                    eris_ovoo[:, b0:b1, :, :],
                    t2[:, :, a0:a1, c0:c1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "jbkm,mica->abcijk",
                    eris_ovoo[:, b0:b1, :, :],
                    t2[:, :, c0:c1, a0:a1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "kcim,mjab->abcijk",
                    eris_ovoo[:, c0:c1, :, :],
                    t2[:, :, a0:a1, b0:b1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )
                _w_einsum(
                    "kcjm,miba->abcijk",
                    eris_ovoo[:, c0:c1, :, :],
                    t2[:, :, b0:b1, a0:a1],
                    out=w_blk[:ba, :bb, :bc], subtract=True,
                )

                _run_fused(
                    _restricted_v_divide,
                    w_blk[:ba, :bb, :bc], v_blk[:ba, :bb, :bc],
                    eris_ovov, t1, t2, eris.fock[nocc:, :nocc], eia,
                    a0, b0, c0, False, 2.0,
                )
                t3_symm_ip_py(v_blk, blksize**3, nocc, "4-2-211-2", 1.0, 0.0)
                t3_symm_ip_py(w_blk, blksize**3, nocc, "4-2-211-2", 0.5, 0.0)

                dovov[:, a0:a1, :, b0:b1] += parallel_einsum(
                    "kc,abcijk->iajb", t1[:, c0:c1], w_blk[:ba, :bb, :bc, :, :, :]
                )
                dooov[:, :, :, a0:a1] -= parallel_einsum(
                    "mkbc,abcijk->jmia",
                    t2[:, :, b0:b1, c0:c1],
                    v_blk[:ba, :bb, :bc, :, :, :],
                )
                dovvv[:, a0:a1, :, b0:b1] += parallel_einsum(
                    "kjcf,abcijk->iafb",
                    t2[:, :, c0:c1, :],
                    v_blk[:ba, :bb, :bc, :, :, :],
                )


    w_blk = None
    v_blk = None

    dvvov = dovvv.transpose(2, 3, 0, 1)

    if compress_vvvv:
        nvir = mycc.nmo - mycc.nocc
        idx = numpy.tril_indices(nvir)
        vidx = idx[0] * nvir + idx[1]
        dvvvv = dvvvv + dvvvv.transpose(1, 0, 2, 3)
        dvvvv = dvvvv + dvvvv.transpose(0, 1, 3, 2)
        dvvvv = lib.take_2d(dvvvv.reshape(nvir**2, nvir**2), vidx, vidx)
        dvvvv *= 0.25

    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov


def t3_symm_ip_py(A, nocc3, nvir, pattern, alpha=1.0, beta=0.0):
    assert (
        A.dtype == numpy.float64 and A.flags["C_CONTIGUOUS"]
    ), "A must be a contiguous float64 array"

    pattern_c = pattern.encode("utf-8")

    drv = _ccsd.libcc.t3_symm_ip
    drv(
        A.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(nocc3),
        ctypes.c_int64(nvir),
        ctypes.c_char_p(pattern_c),
        ctypes.c_double(alpha),
        ctypes.c_double(beta),
    )
    return A
