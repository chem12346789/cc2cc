# pylint: disable=W0212

import types

import numpy as np
import torch

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.numint import (
    _dot_ao_ao,
    _scale_ao_sparse,
    _dot_ao_ao_sparse,
    _tau_dot_sparse,
    _format_uks_dm,
    MGGA_DENSITY_LAPL,
)
from pyscf.dft.gen_grid import NBINS
from pyscf.grad.rks import _d1_dot_, _gga_grad_sum_


def get_veff_modified(
    ks,
    modeldict,
    lambda_rho=None,
    dm_tar=None,
    max_memory=800,
):
    """
    Get the method of "Get the effective potential for the UKS method".
    Note the max_memory=800 use around 8GB gpu memory.
    """

    def nr_uks(
        modeldict,
        ni,
        mol,
        grids,
        xc_code,
        dms,
        max_memory,
        hermi=1,
    ):
        """
        Obtain the nelec, excsum, and vmat.
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
        excsum_b3lyp = np.zeros(nset)
        vmat = np.zeros((2, nset, nao, nao))

        def block_loop(ao_deriv):
            for ao, mask, weights_, coords_ in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory=max_memory, non0tab=grids.non0tab
            ):
                for i in range(nset):
                    rho_a = make_rhoa(i, ao, mask, xctype)
                    rho_b = make_rhob(i, ao, mask, xctype)
                    rho = (rho_a, rho_b)
                    exc_b3lyp, energy_den, vxc = modeldict.eval_xc_eff(
                        rho, ni, dms, grids, coords_, mask
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
                    excsum_b3lyp[i] += np.dot(weights_, exc_b3lyp)
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
            excsum_b3lyp = excsum_b3lyp[0]

        dtype = np.result_type(dma, dmb)
        if vmat.dtype != dtype:
            vmat = np.asarray(vmat, dtype=dtype)

        if hasattr(modeldict.model, "normal_factor"):
            modeldict.model.normal_factor = np.abs(excsum_b3lyp)

        return nelec, excsum, vmat

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

        nelec, exc, vxc = nr_uks(
            modeldict,
            ni,
            mol,
            ks_.grids,
            ks_.xc,
            dm,
            max_memory=max_memory,
            hermi=hermi,
        )

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


def get_veff_grad_modified(
    ks_grad,
    modeldict,
    max_memory=800,
    dm_ks=None,
):
    """
    Get the method of "Get the effective potential for the UKS Gradients method".
    """

    def get_vxc(
        ni,
        mol,
        grids,
        xc_code,
        dms,
        relativity=0,
        hermi=1,
        max_memory=max_memory,
        verbose=None,
    ):
        xctype = ni._xc_type(xc_code)
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
        ao_loc = mol.ao_loc_nr()

        vmat = np.zeros((2, 3, nao, nao))
        # if xctype == "LDA":
        #     ao_deriv = 1
        #     for ao, mask, weight, coords in ni.block_loop(
        #         mol, grids, nao, ao_deriv, max_memory
        #     ):
        #         rho_a = make_rho(0, ao[0], mask, xctype)
        #         rho_b = make_rho(1, ao[0], mask, xctype)
        #         vxc = ni.eval_xc_eff(xc_code, (rho_a, rho_b), 1, xctype=xctype)[1]
        #         wv = weight * vxc[:, 0]
        #         aow = numint._scale_ao(ao[0], wv[0])
        #         _d1_dot_(vmat[0], mol, ao[1:4], aow, mask, ao_loc, True)
        #         aow = numint._scale_ao(ao[0], wv[1])
        #         _d1_dot_(vmat[1], mol, ao[1:4], aow, mask, ao_loc, True)

        if xctype == "GGA":
            ao_deriv = 2
            for ao, mask, weight, coords_ in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory
            ):
                rho_a = make_rho(0, ao[:4], mask, xctype)
                rho_b = make_rho(1, ao[:4], mask, xctype)
                rho = (rho_a, rho_b)
                _, _, vxc = modeldict.eval_xc_eff(rho, ni, dms, grids, coords_, mask)
                wv = weight * vxc
                wv[:, 0] *= 0.5
                _gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
                _gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)

                # # aow = _scale_ao(ao[:4], wv[:4])
                # # _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)
                # # # ##### in np.einsum #####
                # vmat[:, idm] += np.einsum(
                #     "snp,p,xpi,npj->sxij",
                #     vxc,
                #     weight,
                #     ao[1:4],
                #     ao_array,
                #     optimize=True,
                # )

                # # aow = _make_dR_dao_w(ao, wv[:4])
                # # _d1_dot_(vmat[idm], mol, aow, ao[0], mask, ao_loc, True)
                # # # ##### in np.einsum #####
                # vmat[:, idm] += np.einsum(
                #     "snp,p,nxpi,pj->sxij",
                #     vxc,
                #     weight,
                #     ao_mat,
                #     ao[0],
                #     optimize=True,
                # )
                # de = numpy.einsum("sxij,sij->x", vhf[:, :, p0:p1], dm0[:, p0:p1]) * 2

        # elif xctype == "NLC":
        #     raise NotImplementedError("NLC")

        # elif xctype == "MGGA":
        #     ao_deriv = 2
        #     for ao, mask, weight, coords in ni.block_loop(
        #         mol, grids, nao, ao_deriv, max_memory
        #     ):
        #         rho_a = make_rho(0, ao[:10], mask, xctype)
        #         rho_b = make_rho(1, ao[:10], mask, xctype)
        #         vxc = ni.eval_xc_eff(xc_code, (rho_a, rho_b), 1, xctype=xctype)[1]
        #         wv = weight * vxc
        #         wv[:, 0] *= 0.5
        #         wv[:, 4] *= 0.5
        #         rks_grad._gga_grad_sum_(vmat[0], mol, ao, wv[0], mask, ao_loc)
        #         rks_grad._gga_grad_sum_(vmat[1], mol, ao, wv[1], mask, ao_loc)
        #         rks_grad._tau_grad_dot_(vmat[0], mol, ao, wv[0, 4], mask, ao_loc, True)
        #         rks_grad._tau_grad_dot_(vmat[1], mol, ao, wv[1, 4], mask, ao_loc, True)

        exc = np.zeros((mol.natm, 3))
        # - sign because nabla_X = -nabla_x
        return exc, -vmat

    def get_veff(ks_grad_, mol=None, dm=None):
        """
        First order derivative of DFT effective potential matrix (wrt electron coordinates)

        Args:
            ks_grad_ : grad.uhf.Gradients or grad.uks.Gradients object
        """
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad_.base
        ni = mf._numint
        # grids, nlcgrids = rks_grad._initialize_grids(ks_grad_)

        ni = mf._numint
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, ks_grad_.max_memory * 0.9 - mem_now)
        exc, vxc = get_vxc(
            ni,
            mol,
            ks_grad_.grids,
            mf.xc,
            dm,
            max_memory=max_memory,
            verbose=ks_grad_.verbose,
        )
        # if ks_grad_.grid_response:
        #     exc, vxc = get_vxc_full_response(
        #         ni,
        #         mol,
        #         grids,
        #         mf.xc,
        #         dm,
        #         max_memory=max_memory,
        #         verbose=ks_grad_.verbose,
        #     )
        #     if mf.do_nlc():
        #         if ni.libxc.is_nlc(mf.xc):
        #             xc = mf.xc
        #         else:
        #             xc = mf.nlc
        #         enlc, vnlc = rks_grad.get_nlc_vxc_full_response(
        #             ni,
        #             mol,
        #             nlcgrids,
        #             xc,
        #             dm[0] + dm[1],
        #             max_memory=max_memory,
        #             verbose=ks_grad_.verbose,
        #         )
        #         exc += enlc
        #         vxc += vnlc
        #     logger.debug1(ks_grad_, "sum(grids response) %s", exc.sum(axis=0))
        # else:
        #     exc, vxc = get_vxc(
        #         ni,
        #         mol,
        #         grids,
        #         mf.xc,
        #         dm,
        #         max_memory=max_memory,
        #         verbose=ks_grad_.verbose,
        #     )
        #     if mf.do_nlc():
        #         if ni.libxc.is_nlc(mf.xc):
        #             xc = mf.xc
        #         else:
        #             xc = mf.nlc
        #         enlc, vnlc = rks_grad.get_nlc_vxc(
        #             ni,
        #             mol,
        #             nlcgrids,
        #             xc,
        #             dm[0] + dm[1],
        #             max_memory=max_memory,
        #             verbose=ks_grad_.verbose,
        #         )
        #         vxc += vnlc
        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc += vj[0] + vj[1]
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            vk *= hyb
            if omega != 0:
                vk += ks_grad_.get_k(mol, dm, omega=omega) * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk

        return lib.tag_array(vxc, exc1_grid=exc)

    def extra_force(ks_grad_, atom_id, envs):
        """
        First order derivative of DFT effective potential matrix (wrt electron coordinates)

        Args:
            ks_grad_ : grad.uhf.Gradients or grad.uks.Gradients object
        """
        mol = ks_grad_.base.mol

        if dm_ks is None:
            dm = ks_grad_.base.make_rdm1()
        else:
            dm = dm_ks

        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad_.base
        ni = mf._numint

        mem_now = lib.current_memory()[0]
        max_memory = max(2000, ks_grad_.max_memory * 0.9 - mem_now)

        xctype = ni._xc_type(mf.xc)

        ao_loc = mol.ao_loc_nr()

        force = np.zeros((3))
        aoslices = mol.aoslice_by_atom()
        p0, p1 = aoslices[atom_id, 2:]

        ao_deriv = 2

        weight = ks_grad_.grids.weights
        coords_ = ks_grad_.grids.coords
        mask = ks_grad_.grids.non0tab

        ao = pyscf.dft.numint.eval_ao(mol, coords_, deriv=ao_deriv)
        rho_a = pyscf.dft.numint.eval_rho(mol, ao[:4], dm[0], xctype=xctype)
        rho_b = pyscf.dft.numint.eval_rho(mol, ao[:4], dm[1], xctype=xctype)
        rho = (rho_a, rho_b)

        rho_cube, _, vxc_b3lyp = ks_grad_.grids.gen_cube_rho_uks(
            rho, ni, dm, coords=coords_, mask=mask, require_vxc=True
        )
        input_mat = torch.tensor(
            rho_cube,
            dtype=modeldict.dtype,
            device=modeldict.device,
        )
        input_mat.requires_grad = True
        output_mat = modeldict.model(input_mat)[:, 0]
        middle_cube = torch.autograd.grad(torch.sum(output_mat), input_mat)[0]
        middle_mat = (
            ks_grad_.grids.get_center_density(middle_cube).detach().cpu().numpy()
        )
        grad_mat = np.array(
            [
                0.08 + middle_mat[:, 0],
                0.19 + middle_mat[:, 1],
                0.72 + middle_mat[:, 2],
                0.81 + middle_mat[:, 3],
            ]
        )

        wv = weight * vxc_b3lyp
        wv[:, :, 0, :] *= 0.5

        # # dX, dY, dZ = 1, 2, 3
        # # XX, XY, XZ = 4, 5, 6
        # # YX, YY, YZ = 5, 7, 8
        # # ZX, ZY, ZZ = 6, 8, 9
        ao_array = np.array([ao[0], ao[1], ao[2], ao[3]])
        ao_mat = np.array(
            [
                [ao[1], ao[2], ao[3]],
                [ao[4], ao[5], ao[6]],
                [ao[5], ao[7], ao[8]],
                [ao[6], ao[8], ao[9]],
            ]
        )

        # summation of above three parts
        grad2force = np.einsum(
            "msnp,xpi,npj,sij->mpx",
            wv,
            ao[1:4, :, p0:p1],
            ao_array,
            dm[:, p0:p1],
            optimize=True,
        ) + np.einsum(
            "msnp,nxpi,pj,sij->mpx",
            wv,
            ao_mat[:, :, :, p0:p1],
            ao[0],
            dm[:, p0:p1],
            optimize=True,
        )
        grad2force = -grad2force * 2
        force = np.einsum(
            "mp,mpx->x",
            grad_mat,
            grad2force,
            optimize=True,
        )

        return force

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)
    # ks_grad.extra_force = types.MethodType(extra_force, ks_grad)


def get_veff_grad_modified_zeros(ks_grad):
    """
    Get the method of "Get the effective potential for the UKS Gradients method".
    """

    def get_veff(ks_grad_, mol=None, dm=None):
        """
        First order derivative of DFT effective potential matrix (wrt electron coordinates)

        Args:
            ks_grad_ : grad.uhf.Gradients or grad.uks.Gradients object
        """
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad_.base
        ni = mf._numint

        ni = mf._numint
        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc = vj[0] + vj[1]
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            vk *= hyb
            if omega != 0:
                vk += ks_grad_.get_k(mol, dm, omega=omega) * (alpha - hyb)
            vxc = vj[0] + vj[1] - vk

        return lib.tag_array(vxc, exc1_grid=None)

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)
