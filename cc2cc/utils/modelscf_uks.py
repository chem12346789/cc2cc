# pylint: disable=W0212

import types

import numpy as np
import torch

import pyscf
from pyscf import lib
from pyscf.lib import logger
import pyscf.dft.numint
from pyscf.dft.numint import (
    _scale_ao,
    _scale_ao_sparse,
    _dot_ao_ao,
    _dot_ao_ao_dense,
    _dot_ao_ao_sparse,
    _tau_dot_sparse,
    _format_uks_dm,
    MGGA_DENSITY_LAPL,
)
from pyscf.dft.gen_grid import NBINS
from pyscf.grad.rks import _d1_dot_, _gga_grad_sum_

from cc2cc.utils.env_var import CUBE_MIDDLE, EDGE_SIZE
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.Grids import Grid

lib.logger.TIMER_LEVEL = 4


def get_veff_modified(
    ks,
    modeldict: ModelClass,
    lambda_rho=None,
    dm_tar=None,
):
    """
    Get the method of "Get the effective potential for the UKS method".
    Note the max_memory=2000 use around 8GB gpu memory.
    """

    def nr_uks(
        modeldict: ModelClass,
        ni: pyscf.dft.numint.NumInt,
        mol: pyscf.gto.Mole,
        grids: Grid,
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
        vmat = np.zeros((2, nset, nao, nao))

        def block_loop(ao_deriv):
            for ao, mask, weights_, coords_ in ni.block_loop(
                mol,
                grids,
                nao,
                ao_deriv,
                max_memory=max_memory // (2 * EDGE_SIZE**3),
                non0tab=None,
            ):
                t0 = (logger.process_clock(), logger.perf_counter())
                for i in range(nset):
                    rho_a = make_rhoa(i, ao, mask, xctype)
                    rho_b = make_rhob(i, ao, mask, xctype)
                    den_a = rho_a[0] * weights_
                    den_b = rho_b[0] * weights_
                    nelec[0, i] += den_a.sum()
                    nelec[1, i] += den_b.sum()

                    gridcube = grids.gen_cube(mol, dms, coords_, mask)
                    t0 = logger.timer(mol, "    gen cube", *t0)
                    rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_uks(
                        ni, dms, require_vxc=True
                    )
                    t0 = logger.timer(mol, "    cube rho vxc", *t0)
                    energy_den, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                    t0 = logger.timer(mol, "    model eval", *t0)

                    excsum[i] += np.sum(energy_den)
                    wv = np.einsum(
                        "islpC,piC->slpC",
                        vxc_mat,
                        middle_cube,
                        optimize=True,
                    )
                    wv = wv.reshape(4, len(gridcube.coords))  # lpC -> lP

                    t0 = logger.timer(mol, "    post model eval", *t0)
                    yield i, ao_value, gridcube.non0tab, wv

        aow = None
        pair_mask = mol.get_overlap_cond() < -np.log(ni.cutoff)

        t0 = (logger.process_clock(), logger.perf_counter())
        if xctype == "GGA":
            ao_deriv = 1
            for i, ao, mask, wv in block_loop(ao_deriv):
                t0 = logger.timer(mol, "  vxc on grids", *t0)
                wv[:, 0] *= 0.5
                wva, wvb = wv
                aow = np.einsum("xgi,xg->gi", ao, wva, optimize=True)
                vmat[0, i] += np.einsum("gi,gj->ij", ao[0], aow, optimize=True)
                aow = np.einsum("xgi,xg->gi", ao, wvb, optimize=True)
                vmat[1, i] += np.einsum("gi,gj->ij", ao[0], aow, optimize=True)
                # aow = _scale_ao_sparse(ao, wva, mask, ao_loc, out=aow)
                # _dot_ao_ao_sparse(
                #     ao[0],
                #     aow,
                #     None,
                #     nbins,
                #     mask,
                #     pair_mask,
                #     ao_loc,
                #     hermi=0,
                #     out=vmat[0, i],
                # )
                # aow = _scale_ao_sparse(ao, wvb, mask, ao_loc, out=aow)
                # _dot_ao_ao_sparse(
                #     ao[0],
                #     aow,
                #     None,
                #     nbins,
                #     mask,
                #     pair_mask,
                #     ao_loc,
                #     hermi=0,
                #     out=vmat[1, i],
                # )
                t0 = logger.timer(mol, "  vxc mat", *t0)
            vmat = lib.hermi_sum(vmat.reshape((-1, nao, nao)), axes=(0, 2, 1)).reshape(
                2, nset, nao, nao
            )
        else:
            raise NotImplementedError(f"numint.nr_rks for functional {xc_code}")

        if isinstance(dma, np.ndarray) and dma.ndim == 2:
            vmat = vmat[:, 0]
            nelec = nelec.reshape(2)
            excsum = excsum[0]

        dtype = np.result_type(dma, dmb)
        if vmat.dtype != dtype:
            vmat = np.asarray(vmat, dtype=dtype)

        return nelec, excsum, vmat

    def get_veff(
        ks_,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
    ):
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()
        # ks_.initialize_grids(mol, dm)

        t0 = (logger.process_clock(), logger.perf_counter())

        ground_state = dm.ndim == 3 and dm.shape[0] == 2

        ni = ks_._numint
        if hermi == 2:  # because rho = 0
            n, exc, vxc = (0, 0), 0, 0
        else:
            max_memory = ks_.max_memory - lib.current_memory()[0]
            n, exc, vxc = nr_uks(
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
            t0 = logger.timer(ks_, "  vxc", *t0)

        incremental_jk = (
            ks_._eri is None
            and ks_.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        )
        if incremental_jk:
            _dm = np.asarray(dm) - np.asarray(dm_last)
        else:
            _dm = dm

        if not ni.libxc.is_hybrid_xc(ks.xc):
            vk = None
            vj = ks.get_j(mol, _dm[0] + _dm[1], hermi)
            if incremental_jk:
                vj += vhf_last.vj
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = ks.get_jk(mol, _dm, hermi)
                vk *= hyb
            elif alpha == 0:  # LR=0, only SR exchange
                vj = ks.get_j(mol, _dm, hermi)
                vk = ks.get_k(mol, _dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:  # SR=0, only LR exchange
                vj = ks.get_j(mol, _dm, hermi)
                vk = ks.get_k(mol, _dm, hermi, omega=omega)
                vk *= alpha
            else:  # SR and LR exchange with different ratios
                vj, vk = ks.get_jk(mol, _dm, hermi)
                vk *= hyb
                vklr = ks.get_k(mol, _dm, hermi, omega=omega)
                vklr *= alpha - hyb
                vk += vklr
            vj = vj[0] + vj[1]
            if incremental_jk:
                vj += vhf_last.vj
                vk += vhf_last.vk
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

        t0 = logger.timer(ks_, "  jk", *t0)

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    ks.get_veff = types.MethodType(get_veff, ks)


def get_veff_grad_modified(ks_grad, modeldict):
    """
    Get the method of "Get the effective potential for the UKS Gradients method".
    """

    def get_vxc(
        ni: pyscf.dft.numint.NumInt,
        mol,
        grids: Grid,
        xc_code,
        dms,
        max_memory,
        relativity=0,
        hermi=1,
        verbose=None,
    ):
        xctype = ni._xc_type(xc_code)
        make_rho, nset, nao = ni._gen_rho_evaluator(mol, dms, hermi, False, grids)
        ao_loc = mol.ao_loc_nr()

        vmat = np.zeros((2, 3, nao, nao))
        if xctype == "GGA":
            ao_deriv = 2
            for ao, mask, weights_, coords_ in ni.block_loop(
                mol, grids, nao, ao_deriv, max_memory
            ):
                gridcube = grids.gen_cube(mol, dms, coords_, mask)
                rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_uks(
                    ni, dms, ao_deriv=ao_deriv, require_vxc=True
                )
                _, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                wv = np.einsum(
                    "islpC,piC->slpC",
                    vxc_mat,
                    middle_cube,
                    optimize=True,
                )
                wv[:, 0] *= 0.5
                wv = wv.reshape(2, 4, len(gridcube.coords))  # slpC -> slP

                _gga_grad_sum_(vmat[0], mol, ao_value, wv[0], None, ao_loc)
                _gga_grad_sum_(vmat[1], mol, ao_value, wv[1], None, ao_loc)

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

        exc = np.zeros((mol.natm, 3))
        # - sign because nabla_X = -nabla_x
        return exc, -vmat

    def get_veff(ks_grad_, mol=None, dm=None):
        """Coulomb + XC functional"""
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad_.base
        ni = mf._numint

        max_memory = ks_grad_.max_memory * 0.9 - lib.current_memory()[0]
        exc, vxc = get_vxc(
            ni,
            mol,
            ks_grad_.grids,
            mf.xc,
            dm,
            max_memory,
            verbose=ks_grad_.verbose,
        )
        t0 = logger.timer(ks_grad, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad.get_j(mol, dm)
            vxc += vj[0] + vj[1]
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad.get_jk(mol, dm)
            if ks_grad.auxbasis_response:
                vk.aux = vk.aux * hyb
            vk[:] *= hyb  # inplace * for vk[:] to keep the .aux tag
            if omega != 0:  # For range separated Coulomb operator
                vk_lr = ks_grad.get_k(mol, dm, omega=omega)
                vk[:] += vk_lr * (alpha - hyb)
                if ks_grad.auxbasis_response:
                    vk.aux[:] += vk_lr.aux * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk
            if ks_grad.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
                e1_aux -= np.trace(vk.aux, axis1=0, axis2=1)

        if ks_grad.auxbasis_response:
            logger.debug1(ks_grad, "sum(auxbasis response) %s", e1_aux.sum(axis=0))
            vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = lib.tag_array(vxc, exc1_grid=exc)
        return vxc

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)


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
