import numpy
from pyscf import lib
from pyscf.cc import uccsd_rdm


def u_gamma1_intermediates(mycc, t1, t2, l1, l2, eris=None, for_grad=False):
    d1 = uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)

    if eris is None:
        eris = mycc.ao2mo()

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    nocca, noccb, nvira, nvirb = t2ab.shape
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca, None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb, None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:, :nocca]
    fVO = eris.fockb[noccb:, :noccb]

    # aaa
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eia, eia)
    w = numpy.einsum("ijae,kceb->ijkabc", t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w -= numpy.einsum("mkbc,iajm->ijkabc", t2aa, numpy.asarray(eris.ovoo).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.ovov).conj(), t1a)
    v += numpy.einsum("jkbc,ai->ijkabc", t2aa, fvo) * 0.5

    rw = p6(r6(w)) / d3
    wvd = p6(w + v) / d3
    goo = numpy.einsum("iklabc,jklabc->ij", wvd.conj(), rw) * 0.125
    gvv = numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw.conj()) * 0.125
    gvo = numpy.einsum("jkbc,ijkabc->ai", t2aa.conj(), rw) * 0.125

    # bbb
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eIA, eIA)
    w = numpy.einsum("ijae,kceb->ijkabc", t2bb, numpy.asarray(eris.get_OVVV()).conj())
    w -= numpy.einsum("imab,kcjm->ijkabc", t2bb, numpy.asarray(eris.OVOO).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV).conj(), t1b)
    v += numpy.einsum("jkbc,ai->ijkabc", t2bb, fVO) * 0.5

    rw = p6(r6(w)) / d3
    wvd = p6(w + v) / d3
    gOO = numpy.einsum("iklabc,jklabc->ij", wvd.conj(), rw) * 0.125
    gVV = numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw.conj()) * 0.125
    gVO = numpy.einsum("jkbc,ijkabc->ai", t2bb.conj(), rw) * 0.125

    # baa
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eia, eia)
    w = (
        numpy.einsum("jIeA,kceb->IjkAbc", t2ab, numpy.asarray(eris.get_ovvv()).conj())
        * 2
    )
    w += (
        numpy.einsum("jIbE,kcEA->IjkAbc", t2ab, numpy.asarray(eris.get_ovVV()).conj())
        * 2
    )
    w += numpy.einsum("jkbe,IAec->IjkAbc", t2aa, numpy.asarray(eris.get_OVvv()).conj())
    w -= numpy.einsum("mIbA,kcjm->IjkAbc", t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    w -= numpy.einsum("jMbA,kcIM->IjkAbc", t2ab, numpy.asarray(eris.ovOO).conj()) * 2
    w -= numpy.einsum("jmbc,IAkm->IjkAbc", t2aa, numpy.asarray(eris.OVoo).conj())
    v = numpy.einsum("jbkc,IA->IjkAbc", numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum("jkbc,AI->IjkAbc", t2aa, fVO) * 0.5
    v += numpy.einsum("kIcA,bj->IjkAbc", t2ab, fvo) * 2

    rw = r4(w) / d3
    wvd = (w + v) / d3
    goo += numpy.einsum("kilabc,kjlabc->ij", wvd.conj(), rw) * 0.25
    goo += numpy.einsum("kliabc,kljabc->ij", wvd.conj(), rw) * 0.25
    gOO += numpy.einsum("iklabc,jklabc->ij", wvd.conj(), rw) * 0.25
    gvv += numpy.einsum("ijkcad,ijkcbd->ab", wvd, rw.conj()) * 0.25
    gvv += numpy.einsum("ijkcda,ijkcdb->ab", wvd, rw.conj()) * 0.25
    gVV += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw.conj()) * 0.25
    gvo += numpy.einsum("kica,ijkabc->bj", t2ab.conj(), rw) * 0.5
    gVO += numpy.einsum("jkbc,ijkabc->ai", t2aa.conj(), rw) * 0.125

    # bba
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eIA, eIA)
    w = (
        numpy.einsum("ijae,kceb->ijkabc", t2ab, numpy.asarray(eris.get_OVVV()).conj())
        * 2
    )
    w += (
        numpy.einsum("ijeb,kcea->ijkabc", t2ab, numpy.asarray(eris.get_OVvv()).conj())
        * 2
    )
    w += numpy.einsum("jkbe,iaec->ijkabc", t2bb, numpy.asarray(eris.get_ovVV()).conj())
    w -= numpy.einsum("imab,kcjm->ijkabc", t2ab, numpy.asarray(eris.OVOO).conj()) * 2
    w -= numpy.einsum("mjab,kcim->ijkabc", t2ab, numpy.asarray(eris.OVoo).conj()) * 2
    w -= numpy.einsum("jmbc,iakm->ijkabc", t2bb, numpy.asarray(eris.ovOO).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV).conj(), t1a)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum("JKBC,ai->iJKaBC", t2bb, fvo) * 0.5
    v += numpy.einsum("iKaC,BJ->iJKaBC", t2ab, fVO) * 2

    rw = r4(w) / d3
    wvd = (w + v) / d3
    goo += numpy.einsum("iklabc,jklabc->ij", wvd.conj(), rw) * 0.25
    gOO += numpy.einsum("kilabc,kjlabc->ij", wvd.conj(), rw) * 0.25
    gOO += numpy.einsum("kliabc,kljabc->ij", wvd.conj(), rw) * 0.25
    gvv += numpy.einsum("ijkacd,ijkbcd->ab", wvd, rw.conj()) * 0.25
    gVV += numpy.einsum("ijkcad,ijkcbd->ab", wvd, rw.conj()) * 0.25
    gVV += numpy.einsum("ijkcda,ijkcdb->ab", wvd, rw.conj()) * 0.25
    gVO += numpy.einsum("ikac,ijkabc->bj", t2ab.conj(), rw) * 0.5
    gvo += numpy.einsum("jkbc,ijkabc->ai", t2bb.conj(), rw) * 0.125

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
    w = numpy.einsum("ijae,kceb->ijkabc", t2aa, numpy.asarray(eris.get_ovvv()).conj())
    w -= numpy.einsum("mkbc,iajm->ijkabc", t2aa, numpy.asarray(eris.ovoo).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.ovov).conj(), t1a)
    v += numpy.einsum("jkbc,ai->ijkabc", t2aa, fvo) * 0.5

    rw = r6(p6(w)) / d3
    wvd = r6(p6(w * 2 + v)) / d3
    dovov += numpy.einsum("ia,ijkabc->jbkc", t1a, rw.conj()) * 0.25
    # *(1/8) instead of (1/4) because ooov appears 4 times in the 2pdm tensor due
    # to symmetrization, and its contribution is scaled by 1/2 in Tr(H,2pdm)
    dooov -= numpy.einsum("mkbc,ijkabc->jmia", t2aa, wvd.conj()) * 0.125
    dovvv += numpy.einsum("kjcf,ijkabc->iafb", t2aa, wvd.conj()) * 0.125

    # bbb
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eIA, eIA)
    w = numpy.einsum("ijae,kceb->ijkabc", t2bb, numpy.asarray(eris.get_OVVV()).conj())
    w -= numpy.einsum("imab,kcjm->ijkabc", t2bb, numpy.asarray(eris.OVOO).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV).conj(), t1b)
    v += numpy.einsum("jkbc,ai->ijkabc", t2bb, fVO) * 0.5

    rw = r6(p6(w)) / d3
    wvd = r6(p6(w * 2 + v)) / d3
    dOVOV += numpy.einsum("ia,ijkabc->jbkc", t1b, rw.conj()) * 0.25
    dOOOV -= numpy.einsum("mkbc,ijkabc->jmia", t2bb, wvd.conj()) * 0.125
    dOVVV += numpy.einsum("kjcf,ijkabc->iafb", t2bb, wvd.conj()) * 0.125

    # baa
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eIA, eia, eia)
    w = (
        numpy.einsum("jIeA,kceb->IjkAbc", t2ab, numpy.asarray(eris.get_ovvv()).conj())
        * 2
    )
    w += (
        numpy.einsum("jIbE,kcEA->IjkAbc", t2ab, numpy.asarray(eris.get_ovVV()).conj())
        * 2
    )
    w += numpy.einsum("jkbe,IAec->IjkAbc", t2aa, numpy.asarray(eris.get_OVvv()).conj())
    w -= numpy.einsum("mIbA,kcjm->IjkAbc", t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    w -= numpy.einsum("jMbA,kcIM->IjkAbc", t2ab, numpy.asarray(eris.ovOO).conj()) * 2
    w -= numpy.einsum("jmbc,IAkm->IjkAbc", t2aa, numpy.asarray(eris.OVoo).conj())
    v = numpy.einsum("jbkc,IA->IjkAbc", numpy.asarray(eris.ovov).conj(), t1b)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum("kcIA,jb->IjkAbc", numpy.asarray(eris.ovOV).conj(), t1a)
    v += numpy.einsum("jkbc,AI->IjkAbc", t2aa, fVO) * 0.5
    v += numpy.einsum("kIcA,bj->IjkAbc", t2ab, fvo) * 2

    rw = r4(w) / d3
    wvd = r4(w * 2 + v) / d3
    dovvv += numpy.einsum("jiea,ijkabc->kceb", t2ab, wvd.conj()) * 0.25
    dovVV += numpy.einsum("jibe,ijkabc->kcea", t2ab, wvd.conj()) * 0.25
    dOVvv += numpy.einsum("jkbe,ijkabc->iaec", t2aa, wvd.conj()) * 0.125
    dooov -= numpy.einsum("miba,ijkabc->jmkc", t2ab, wvd.conj()) * 0.25
    dOOov -= numpy.einsum("jmba,ijkabc->imkc", t2ab, wvd.conj()) * 0.25
    dooOV -= numpy.einsum("jmbc,ijkabc->kmia", t2aa, wvd.conj()) * 0.125
    dovov += numpy.einsum("ia,ijkabc->jbkc", t1b, rw.conj()) * 0.25
    # dOVov += numpy.einsum('jb,ijkabc->iakc', t1a, rw.conj()) * .25
    dovOV += numpy.einsum("jb,ijkabc->kcia", t1a, rw.conj()) * 0.25

    # bba
    d3 = lib.direct_sum("ia+jb+kc->ijkabc", eia, eIA, eIA)
    w = (
        numpy.einsum("ijae,kceb->ijkabc", t2ab, numpy.asarray(eris.get_OVVV()).conj())
        * 2
    )
    w += (
        numpy.einsum("ijeb,kcea->ijkabc", t2ab, numpy.asarray(eris.get_OVvv()).conj())
        * 2
    )
    w += numpy.einsum("jkbe,iaec->ijkabc", t2bb, numpy.asarray(eris.get_ovVV()).conj())
    w -= numpy.einsum("imab,kcjm->ijkabc", t2ab, numpy.asarray(eris.OVOO).conj()) * 2
    w -= numpy.einsum("mjab,kcim->ijkabc", t2ab, numpy.asarray(eris.OVoo).conj()) * 2
    w -= numpy.einsum("jmbc,iakm->ijkabc", t2bb, numpy.asarray(eris.ovOO).conj())
    v = numpy.einsum("jbkc,ia->ijkabc", numpy.asarray(eris.OVOV).conj(), t1a)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum("iakc,jb->ijkabc", numpy.asarray(eris.ovOV).conj(), t1b)
    v += numpy.einsum("JKBC,ai->iJKaBC", t2bb, fvo) * 0.5
    v += numpy.einsum("iKaC,BJ->iJKaBC", t2ab, fVO) * 2

    rw = r4(w) / d3
    wvd = r4(w * 2 + v) / d3
    dOVVV += numpy.einsum("ijae,ijkabc->kceb", t2ab, wvd.conj()) * 0.25
    dOVvv += numpy.einsum("ijeb,ijkabc->kcea", t2ab, wvd.conj()) * 0.25
    dovVV += numpy.einsum("jkbe,ijkabc->iaec", t2bb, wvd.conj()) * 0.125
    dOOOV -= numpy.einsum("imab,ijkabc->jmkc", t2ab, wvd.conj()) * 0.25
    dooOV -= numpy.einsum("mjab,ijkabc->imkc", t2ab, wvd.conj()) * 0.25
    dOOov -= numpy.einsum("jmbc,ijkabc->kmia", t2bb, wvd.conj()) * 0.125
    dOVOV += numpy.einsum("ia,ijkabc->jbkc", t1a, rw.conj()) * 0.25
    dovOV += numpy.einsum("jb,ijkabc->iakc", t1b, rw.conj()) * 0.25
    # dOVov += numpy.einsum('jb,ijkabc->kcia', t1b, rw.conj()) * .25

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
