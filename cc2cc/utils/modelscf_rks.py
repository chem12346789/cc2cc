"""
This module provides the modified RKS method for model-based DFT calculations.
The ``uncommment commment'' is the original docstring.
"""

# pylint: disable=W0212

import types

import numpy as np
import torch

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.numint import (
    _scale_ao,
    _scale_ao_sparse,
    _dot_ao_ao_sparse,
    # _tau_dot_sparse,
)
from pyscf.dft.gen_grid import NBINS
from pyscf.grad.rks import _d1_dot_, _gga_grad_sum_, _make_dR_dao_w


def get_veff_modified(
    ks,
    modeldict,
    lambda_rho=None,
    dm_tar=None,
    max_memory=800,
):
    """
    Get the method of "Get the effective potential for the RKS method".
    """

    def nr_rks(
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
        Get the effective potential for the RKS method.
        Note the max_memory=800 use around 8GB gpu memory.
        Modified from pyscf.dft.numint.nr_rks (https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/numint.py)
        """
        xctype = ni._xc_type(xc_code)
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
        ao_loc = mol.ao_loc_nr()
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * np.log(cutoff) / np.log(grids.cutoff))

        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        excsum_b3lyp = np.zeros(nset)
        vmat = np.zeros((nset, nao, nao))

        def block_loop(ao_deriv):
            for ao, mask, weights_, coords_ in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory=max_memory, non0tab=grids.non0tab
            ):
                for i in range(nset):
                    rho = make_rho(i, ao, mask, xctype)
                    exc_b3lyp, energy_den, vxc = modeldict.eval_xc_eff(
                        rho, ni, dms, grids, coords_, mask
                    )

                    if xctype == "LDA":
                        den = rho * weights_
                    else:
                        den = rho[0] * weights_
                    nelec[i] += den.sum()
                    excsum[i] += np.dot(weights_, energy_den)
                    excsum_b3lyp[i] += modeldict.calculate_normal(
                        exc_b3lyp, weights_
                    )
                    wv = weights_ * vxc
                    yield i, ao, mask, wv

        aow = None
        pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)

        # if xctype == "LDA":
        #     ao_deriv = 0
        #     for i, ao, mask, wv in block_loop(ao_deriv):
        #         _dot_ao_ao_sparse(
        #             ao, ao, wv, nbins, mask, pair_mask, ao_loc, hermi, vmat[i]
        #         )

        if xctype == "GGA":
            ao_deriv = 1
            for i, ao, mask, wv in block_loop(ao_deriv):
                wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end
                aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
                _dot_ao_ao_sparse(
                    ao[0],
                    aow,
                    None,
                    nbins,
                    mask,
                    pair_mask,
                    ao_loc,
                    hermi=0,
                    out=vmat[i],
                )
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))

        # elif xctype == "MGGA":
        #     if any(x in xc_code.upper() for x in ("CC06", "CS", "BR89", "MK00")):
        #         raise NotImplementedError("laplacian in meta-GGA method")
        #     ao_deriv = 1
        #     v1 = np.zeros_like(vmat)
        #     for i, ao, mask, wv in block_loop(ao_deriv):
        #         wv[0] *= 0.5  # *.5 for v+v.conj().T
        #         wv[4] *= 0.5  # *.5 for 1/2 in tau
        #         aow = _scale_ao_sparse(ao[:4], wv[:4], mask, ao_loc, out=aow)
        #         _dot_ao_ao_sparse(
        #             ao[0],
        #             aow,
        #             None,
        #             nbins,
        #             mask,
        #             pair_mask,
        #             ao_loc,
        #             hermi=0,
        #             out=vmat[i],
        #         )
        #         _tau_dot_sparse(
        #             ao, ao, wv[4], nbins, mask, pair_mask, ao_loc, out=v1[i]
        #         )
        #     vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
        #     vmat += v1

        # elif xctype == "HF":
        #     pass
        else:
            raise NotImplementedError(f"numint.nr_rks for functional {xc_code}")

        if nset == 1:
            vmat = vmat[0]
            nelec = nelec[0]
            excsum = excsum[0]
        excsum_b3lyp = np.sum(excsum_b3lyp)

        if isinstance(dms, np.ndarray):
            dtype = dms.dtype
        else:
            dtype = np.result_type(*dms)
        if vmat.dtype != dtype:
            vmat = np.asarray(vmat, dtype=dtype)

        if hasattr(modeldict, "normal_factor"):
            modeldict.normal_factor = modeldict.calculate_normal_final(excsum_b3lyp)

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
        """
        # Get the effective potential for the RKS method.
        # This function is used to get the effective potential for the RKS method.
        # Modified from pyscf.dft.rks.get_veff; See
        # https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/rks.py
        """
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

        # ks_.initialize_grids(mol, dm)
        t0 = (logger.process_clock(), logger.perf_counter())

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2

        ni = ks_._numint
        if hermi == 2:  # because rho = 0
            n, exc, vxc = 0, 0, 0
        else:
            n, exc, vxc = nr_rks(
                modeldict,
                ni,
                mol,
                ks_.grids,
                ks_.xc,
                dm,
                max_memory=max_memory,
                hermi=hermi,
            )
            logger.debug(ks_, "nelec by numeric integration = %s", n)
            t0 = logger.timer(ks_, "vxc", *t0)

        incremental_jk = (
            ks_._eri is None
            and ks_.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        )
        if incremental_jk:
            _dm = np.asarray(dm) - np.asarray(dm_last)
        else:
            _dm = dm

        if not ni.libxc.is_hybrid_xc(ks_.xc):
            vk = None
            vj = ks_.get_j(mol, _dm, hermi)
            if incremental_jk:
                vj += vhf_last.vj
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks_.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = ks_.get_jk(mol, _dm, hermi)
                vk *= hyb
            elif alpha == 0:  # LR=0, only SR exchange
                vj = ks_.get_j(mol, _dm, hermi)
                vk = ks_.get_k(mol, _dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:  # SR=0, only LR exchange
                vj = ks_.get_j(mol, _dm, hermi)
                vk = ks_.get_k(mol, _dm, hermi, omega=omega)
                vk *= alpha
            else:  # SR and LR exchange with different ratios
                vj, vk = ks_.get_jk(mol, _dm, hermi)
                vk *= hyb
                vklr = ks_.get_k(mol, _dm, hermi, omega=omega)
                vklr *= alpha - hyb
                vk += vklr
            if incremental_jk:
                vj += vhf_last.vj
                vk += vhf_last.vk
            vxc += vj - vk * 0.5

            if ground_state:
                exc -= np.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
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
    Get the method of "Get the effective potential for the RKS Gradients method".
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

        vmat = np.zeros((nset, 3, nao, nao))
        # if xctype == "LDA":
        #     ao_deriv = 1
        #     for ao, mask, weight, coords in ni.block_loop(
        #         mol, grids, nao, ao_deriv, max_memory
        #     ):
        #         for idm in range(nset):
        #             rho = make_rho(idm, ao[0], mask, xctype)
        #             vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
        #             wv = weight * vxc[0]
        #             aow = numint._scale_ao(ao[0], wv)
        #             _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)

        if xctype == "GGA":
            ao_deriv = 2
            for ao, mask, weight, coords_ in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory
            ):
                for idm in range(nset):
                    rho = make_rho(idm, ao[:4], mask, xctype)
                    _, _, vxc = modeldict.eval_xc_eff(
                        rho, ni, dms, grids, coords_, mask
                    )
                    wv = weight * vxc
                    wv[0] *= 0.5
                    _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)

                    # # aow = _scale_ao(ao[:4], wv[:4])
                    # # _d1_dot_(vmat[idm], mol, ao[1:4], aow, mask, ao_loc, True)
                    # # # ##### in np.einsum #####
                    # vmat[idm] += np.einsum(
                    #     "np,p,xpi,npj->xij",
                    #     vxc,
                    #     weight,
                    #     ao[1:4],
                    #     ao_array,
                    #     optimize=True,
                    # )

                    # # aow = _make_dR_dao_w(ao, wv[:4])
                    # # _d1_dot_(vmat[idm], mol, aow, ao[0], mask, ao_loc, True)
                    # # # ##### in np.einsum #####
                    # vmat[idm] += np.einsum(
                    #     "np,p,nxpi,pj->xij",
                    #     vxc,
                    #     weight,
                    #     ao_mat,
                    #     ao[0],
                    #     optimize=True,
                    # )

        # elif xctype == "MGGA":
        #     ao_deriv = 2
        #     for ao, mask, weight, coords in ni.block_loop(
        #         mol, grids, nao, ao_deriv, max_memory
        #     ):
        #         for idm in range(nset):
        #             rho = make_rho(idm, ao[:10], mask, xctype)
        #             vxc = ni.eval_xc_eff(xc_code, rho, 1, xctype=xctype)[1]
        #             wv = weight * vxc
        #             wv[0] *= 0.5
        #             wv[4] *= 0.5  # for the factor 1/2 in tau
        #             _gga_grad_sum_(vmat[idm], mol, ao, wv, mask, ao_loc)
        #             _tau_grad_dot_(vmat[idm], mol, ao, wv[4], mask, ao_loc, True)

        exc = None
        if nset == 1:
            vmat = vmat[0]
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
        # grids, nlcgrids = _initialize_grids(ks_grad_)

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
        #         enlc, vnlc = get_nlc_vxc_full_response(
        #             ni,
        #             mol,
        #             nlcgrids,
        #             xc,
        #             dm,
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
        #         enlc, vnlc = get_nlc_vxc(
        #             ni,
        #             mol,
        #             nlcgrids,
        #             xc,
        #             dm,
        #             max_memory=max_memory,
        #             verbose=ks_grad_.verbose,
        #         )
        #         vxc += vnlc
        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            vk *= hyb
            if omega != 0:
                vk += ks_grad_.get_k(mol, dm, omega=omega) * (alpha - hyb)
            vxc += vj - vk * 0.5

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

        mf = ks_grad_.base
        ni = mf._numint

        xctype = ni._xc_type(mf.xc)

        force = np.zeros((3))
        aoslices = mol.aoslice_by_atom()
        p0, p1 = aoslices[atom_id, 2:]

        ao_deriv = 2

        weight = ks_grad_.grids.weights
        coords_ = ks_grad_.grids.coords
        mask = ks_grad_.grids.non0tab

        ao = pyscf.dft.numint.eval_ao(mol, coords_, deriv=ao_deriv)
        rho = pyscf.dft.numint.eval_rho(mol, ao[:4], dm, xctype=xctype)

        rho_cube, _, vxc_b3lyp = ks_grad_.grids.gen_cube_rho_rks(
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
        wv[:, 0, :] *= 0.5

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
            "mnp,xpi,npj,ij->mpx",
            wv,
            ao[1:4, :, p0:p1],
            ao_array,
            dm[p0:p1],
            optimize=True,
        ) + np.einsum(
            "mnp,nxpi,pj,ij->mpx",
            wv,
            ao_mat[:, :, :, p0:p1],
            ao[0],
            dm[p0:p1],
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
    Get the method of "Get the effective potential for the RKS Gradients method".
    This will reurn force without contribution from the DFT functional.
    For debugging use only.
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

        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc = vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            vk *= hyb
            if omega != 0:
                vk += ks_grad_.get_k(mol, dm, omega=omega) * (alpha - hyb)
            vxc = vj - vk * 0.5

        return lib.tag_array(vxc, exc1_grid=None)

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)
    # ks_grad.extra_force = types.MethodType(extra_force, ks_grad)
