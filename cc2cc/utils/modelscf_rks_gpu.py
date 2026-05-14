"""
GPU4PySCF version of the modified RKS hooks used by cc2cc.
"""

# pylint: disable=W0212

import types

import cupy as cp
import torch

from gpu4pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft.numint import NumInt, _scale_ao
from gpu4pyscf.grad.rks import _gga_grad_sum_
from gpu4pyscf.lib.cupy_helper import tag_array, asarray

from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.GridsGPU import GridGPU as Grid, iterate_grid_segments


def _hermi_sum(vmat):
    return vmat + vmat.transpose(0, 2, 1)


def get_veff_modified_rks_gpu(ks, modeldict):
    """
    Get the method of "Get the effective potential for the RKS method".
    """

    def nr_rks(
        modeldict: ModelClass,
        ni: NumInt,
        mol,
        grids: Grid,
        xc_code,
        dms,
        max_memory,
        hermi=1,
    ):
        """
        Get the effective potential for the RKS method.
        Modified from pyscf.dft.numint.nr_rks.
        """
        xctype = ni._xc_type(xc_code)

        nset = 1
        nao = mol.nao

        def block_loop(ao_deriv):
            for mask, weights_, coords_ in iterate_grid_segments(
                mol,
                grids,
                nao,
                ao_deriv,
                max_memory=max_memory // (2 * modeldict.model.cube_size),
                non0tab=None,
            ):
                for i in range(nset):

                    gridcube = grids.gen_cube(mol, dms, coords_, mask)
                    t0 = (logger.process_clock(), logger.perf_counter())
                    rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_rks(ni, dms)
                    t0 = logger.timer(mol, "    cube rho vxc", *t0)
                    energy_den, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                    energy_den = cp.asarray(energy_den)
                    middle_cube = cp.asarray(middle_cube)

                    excsum[i] += cp.sum(energy_den)
                    wv = cp.einsum("ixgC,giC->xgC", vxc_mat, middle_cube, optimize=True)
                    wv = wv.reshape(4, len(gridcube.coords))  # xgC -> xG

                    yield i, ao_value, gridcube.non0tab, wv

        nelec = cp.zeros(nset)
        excsum = cp.zeros(nset)
        vmat = cp.zeros((nset, nao, nao))

        t0 = (logger.process_clock(), logger.perf_counter())
        if xctype == "GGA":
            ao_deriv = 1
            for i, ao, _, wv in block_loop(ao_deriv):
                t0 = logger.timer(mol, "  vxc on grids", *t0)
                wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end

                aow = _scale_ao(ao, wv)
                vmat[i] += cp.dot(ao[0], aow.T)

                # aow = cp.einsum("xng,xg->ng", ao, wv, optimize=True)
                # vmat[i] += cp.einsum("ng,mg->nm", ao[0], aow, optimize=True)
                t0 = logger.timer(mol, "  vxc mat", *t0)
            vmat = _hermi_sum(vmat)
        else:
            raise NotImplementedError(f"numint.nr_rks for functional {xc_code}")

        if nset == 1:
            vmat = vmat[0]
            nelec = nelec[0]
            excsum = excsum[0]

        dtype = dms.dtype if isinstance(dms, cp.ndarray) else cp.result_type(*dms)
        if vmat.dtype != dtype:
            vmat = cp.asarray(vmat, dtype=dtype)

        return nelec, excsum, vmat

    def get_veff(
        ks_,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
    ):
        """
        Get the effective potential for the RKS method.
        Modified from pyscf.dft.rks.get_veff.
        """
        logger.debug(ks_, "Using modified get_veff (gpu4pyscf)")
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

        t0 = (logger.process_clock(), logger.perf_counter())

        ground_state = isinstance(dm, cp.ndarray) and dm.ndim == 2

        ni = ks_._numint
        if hermi == 2:  # because rho = 0
            n, exc, vxc = 0, 0, 0
        else:
            current_memory = getattr(lib, "current_memory", lambda: (0,))()[0]
            max_memory = ks_.max_memory - current_memory
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
            _dm = cp.asarray(dm) - cp.asarray(dm_last)
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
                exc -= cp.einsum("ij,ji", dm, vk).real * 0.5 * 0.5
        if ground_state:
            ecoul = cp.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None

        t0 = logger.timer(ks_, "jk", *t0)

        vxc = tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    ks.get_veff = types.MethodType(get_veff, ks)


def get_veff_grad_modified_rks_gpu(ks_grad, modeldict):
    """
    Get the method of "Get the effective potential for the RKS Gradients method".
    """

    raise NotImplementedError("get_veff_grad_modified_rks_gpu is not implemented yet")

    def get_vxc(
        ni: NumInt,
        mol,
        grids: Grid,
        xc_code,
        dms,
        max_memory,
        relativity=0,
        hermi=1,
        verbose=None,
    ):
        del relativity, hermi, verbose
        xctype = ni._xc_type(xc_code)
        ao_loc = mol.ao_loc_nr()

        nset = 1
        nao = mol.nao

        vmat = cp.zeros((nset, 3, nao, nao))

        if xctype == "GGA":
            ao_deriv = 2
            for mask, weights_, coords_ in iterate_grid_segments(
                mol,
                grids,
                nao,
                ao_deriv,
                max_memory=max_memory // (2 * modeldict.model.cube_size),
                non0tab=None,
            ):
                for idm in range(nset):
                    gridcube = grids.gen_cube(mol, dms, coords_, mask)
                    rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_rks(
                        ni, dms, ao_deriv=ao_deriv
                    )
                    _, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                    middle_cube = cp.asarray(middle_cube)

                    wv = cp.einsum(
                        "ilpC,piC->lpC",
                        vxc_mat,
                        middle_cube,
                        optimize=True,
                    )
                    wv[0] *= 0.5
                    wv = wv.reshape(4, len(gridcube.coords))  # lpC -> lP
                    _gga_grad_sum_(vmat[idm], mol, ao_value, wv, None, ao_loc)

        exc = None
        if nset == 1:
            vmat = vmat[0]
        # - sign because nabla_X = -nabla_x
        return exc, -vmat

    def get_veff(ks_grad_, mol=None, dm=None):
        """
        First order derivative of DFT effective potential matrix.
        """
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())

        mf = ks_grad_.base
        ni = mf._numint

        current_memory = getattr(lib, "current_memory", lambda: (0,))()[0]
        max_memory = ks_grad_.max_memory * 0.9 - current_memory
        exc, vxc = get_vxc(
            ni,
            mol,
            ks_grad_.grids,
            mf.xc,
            dm,
            max_memory,
            verbose=ks_grad_.verbose,
        )
        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc += vj
            if ks_grad_.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            if ks_grad_.auxbasis_response:
                vk.aux *= hyb
            vk[:] *= hyb  # Don't erase the .aux tags!
            if omega != 0:  # For range separated Coulomb operator
                vk_lr = ks_grad_.get_k(mol, dm, omega=omega)
                vk[:] += vk_lr * (alpha - hyb)
                if ks_grad_.auxbasis_response:
                    vk.aux[:] += vk_lr.aux * (alpha - hyb)
            vxc += vj - vk * 0.5
            if ks_grad_.auxbasis_response:
                e1_aux = (vj.aux - vk.aux * 0.5).sum((0, 1))

        if ks_grad_.auxbasis_response:
            logger.debug1(ks_grad_, "sum(auxbasis response) %s", e1_aux.sum(axis=0))
            vxc = lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = lib.tag_array(vxc, exc1_grid=exc)
        return lib.tag_array(vxc, exc1_grid=exc)

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)


__all__ = [
    "get_veff_modified_rks_gpu",
    "get_veff_grad_modified_rks_gpu",
]
