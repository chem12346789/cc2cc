# pylint: disable=W0212

import types

import numpy as np
import torch

from pyscf import lib
from pyscf.dft.numint import (
    _dot_ao_ao,
    _scale_ao_sparse,
    _dot_ao_ao_sparse,
    _tau_dot_sparse,
    _format_uks_dm,
    MGGA_DENSITY_LAPL,
)
from pyscf.dft.gen_grid import NBINS


def nr_uks(
    modelclass,
    ni,
    mol,
    grids,
    dms,
    xc_code="b3lyp",
    hermi=1,
    max_memory=20,
    verbose=None,
):
    """
    Obtain the nelec, excsum, and vmat.
    Note the max_memory=20 use around 8GB gpu memory.
    """
    xctype = ni._xc_type(xc_code)
    ao_loc = mol.ao_loc_nr()
    cutoff = grids.cutoff * 1e2
    nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

    dma, dmb = _format_uks_dm(dms)
    nao = dma.shape[-1]
    make_rhoa, nset = ni._gen_rho_evaluator(mol, dma, hermi, False, grids)[:2]
    make_rhob = ni._gen_rho_evaluator(mol, dmb, hermi, False, grids)[0]

    nelec = np.zeros((2, nset))
    excsum = np.zeros(nset)
    vmat = np.zeros((2, nset, nao, nao))

    def block_loop(ao_deriv):
        for ao, mask, weights_, coords_ in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)
                energy_den, vxc = modelclass.eval_xc_eff(
                    mol, dms, rho, ni, grids, weights_, coords_
                )

                if xctype == "LDA":
                    den_a = rho_a * weights_
                    den_b = rho_b * weights_
                else:
                    den_a = rho_a[0] * weights_
                    den_b = rho_b[0] * weights_

                nelec[0, i] += den_a.sum()
                nelec[1, i] += den_b.sum()
                excsum[i] += np.dot(weights_, energy_den)
                wv = weights_ * vxc
                yield i, ao, mask, wv

    pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)
    aow = None
    # if xctype == "LDA":
    #     ao_deriv = 0
    #     for i, ao, mask, wv in block_loop(ao_deriv):
    #         _dot_ao_ao_sparse(
    #             ao, ao, wv[0, 0], nbins, mask, pair_mask, ao_loc, hermi, vmat[0, i]
    #         )
    #         _dot_ao_ao_sparse(
    #             ao, ao, wv[1, 0], nbins, mask, pair_mask, ao_loc, hermi, vmat[1, i]
    #         )

    if xctype == "GGA":
        ao_deriv = 1
        for i, ao, mask, wv in block_loop(ao_deriv):
            wv[:, 0] *= 0.5
            wva, wvb = wv
            aow = _scale_ao_sparse(ao, wva, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0],
                aow,
                None,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                hermi=0,
                out=vmat[0, i],
            )
            aow = _scale_ao_sparse(ao, wvb, mask, ao_loc, out=aow)
            _dot_ao_ao_sparse(
                ao[0],
                aow,
                None,
                nbins,
                mask,
                pair_mask,
                ao_loc,
                hermi=0,
                out=vmat[1, i],
            )
        vmat = lib.hermi_sum(vmat.reshape((-1, nao, nao)), axes=(0, 2, 1)).reshape(
            2, nset, nao, nao
        )

    # elif xctype == "MGGA":
    #     if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
    #         raise NotImplementedError("laplacian in meta-GGA method")
    #     assert not MGGA_DENSITY_LAPL
    #     ao_deriv = 1
    #     v1 = np.zeros_like(vmat)
    #     for i, ao, mask, wv in block_loop(ao_deriv):
    #         wv[:, 0] *= 0.5
    #         wv[:, 4] *= 0.5
    #         wva, wvb = wv
    #         aow = _scale_ao_sparse(ao[:4], wva[:4], mask, ao_loc, out=aow)
    #         _dot_ao_ao_sparse(
    #             ao[0],
    #             aow,
    #             None,
    #             nbins,
    #             mask,
    #             pair_mask,
    #             ao_loc,
    #             hermi=0,
    #             out=vmat[0, i],
    #         )
    #         _tau_dot_sparse(
    #             ao, ao, wva[4], nbins, mask, pair_mask, ao_loc, out=v1[0, i]
    #         )
    #         aow = _scale_ao_sparse(ao[:4], wvb[:4], mask, ao_loc, out=aow)
    #         _dot_ao_ao_sparse(
    #             ao[0],
    #             aow,
    #             None,
    #             nbins,
    #             mask,
    #             pair_mask,
    #             ao_loc,
    #             hermi=0,
    #             out=vmat[1, i],
    #         )
    #         _tau_dot_sparse(
    #             ao, ao, wvb[4], nbins, mask, pair_mask, ao_loc, out=v1[1, i]
    #         )
    #     vmat = lib.hermi_sum(vmat.reshape((-1, nao, nao)), axes=(0, 2, 1)).reshape(
    #         2, nset, nao, nao
    #     )
    #     vmat += v1
    # elif xctype == "HF":
    #     pass
    else:
        raise NotImplementedError(f"numint.nr_uks for functional {xc_code}")

    if isinstance(dma, np.ndarray) and dma.ndim == 2:
        vmat = vmat[:, 0]
        nelec = nelec.reshape(2)
        excsum = excsum[0]

    dtype = np.result_type(dma, dmb)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


def get_veff_modified(ks, modeldict, lambda_rho=None, dm_tar=None):
    """
    Get the method of "Get the effective potential for the UKS method".
    """

    def get_veff(
        ks_,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
        lambda_rho=lambda_rho,
        dm_tar=dm_tar,
    ):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

        ground_state = dm.ndim == 3 and dm.shape[0] == 2
        ni = ks_._numint

        nelec, exc, vxc = nr_uks(modeldict, ni, mol, ks_.grids, dm, ks_.xc)

        if not ni.libxc.is_hybrid_xc(ks_.xc):
            vk = None
            if (
                ks_._eri is None
                and ks_.direct_scf
                and getattr(vhf_last, "vj", None) is not None
            ):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj = ks_.get_j(mol, ddm[0] + ddm[1], hermi)
                vj += vhf_last.vj
            else:
                vj = ks_.get_j(mol, dm[0] + dm[1], hermi)
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks_.xc, spin=mol.spin)
            if (
                ks_._eri is None
                and ks_.direct_scf
                and getattr(vhf_last, "vk", None) is not None
            ):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj, vk = ks_.get_jk(mol, ddm, hermi)
                vk *= hyb
                if omega != 0:
                    vklr = ks_.get_k(mol, ddm, hermi, omega)
                    vklr *= alpha - hyb
                    vk += vklr
                vj = vj[0] + vj[1] + vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks_.get_jk(mol, dm, hermi)
                vj = vj[0] + vj[1]
                vk *= hyb
                if omega != 0:
                    vklr = ks_.get_k(mol, dm, hermi, omega)
                    vklr *= alpha - hyb
                    vk += vklr
            vxc += vj - vk

            if ground_state:
                exc -= (
                    np.einsum("ij,ji", dm[0], vk[0]).real
                    + np.einsum("ij,ji", dm[1], vk[1]).real
                ) * 0.5
        if ground_state:
            ecoul = np.einsum("ij,ji", dm[0] + dm[1], vj).real * 0.5
        else:
            ecoul = None

        if lambda_rho is not None and dm_tar is not None:
            delta_j = ks_.get_j(mol, dm - dm_tar, hermi=hermi)
            vxc = vxc + lambda_rho * delta_j

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    ks.get_veff = types.MethodType(get_veff, ks)
