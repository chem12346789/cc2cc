"""
This module provides the modified RKS method for model-based DFT calculations.
The ``uncommment commment'' is the original docstring.
"""

# pylint: disable=W0212

import types

import numpy as np

import pyscf
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.numint import NumInt, _scale_ao, _dot_ao_ao_dense
from pyscf.grad.rks import _gga_grad_sum_

from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.Grids import Grid, iterate_grid_segments


def get_veff_modified_rks(ks, modeldict):
    """
    Get the method of "Get the effective potential for the RKS method".
    """

    def nr_rks(
        modeldict: ModelClass,
        ni: NumInt,
        mol: pyscf.gto.Mole,
        grids: Grid,
        xc_code,
        dms,
        max_memory,
        hermi=1,
    ):
        """
        Get the effective potential for the RKS method.
        Note the max_memory=2000 use around 8GB gpu memory.
        Modified from pyscf.dft.numint.nr_rks (https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/numint.py)
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
                    rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_rks(ni, dms)
                    energy_den, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)

                    excsum[i] += np.sum(energy_den)
                    wv = np.einsum("ixgC,giC->xgC", vxc_mat, middle_cube, optimize=True)


                    wv = wv.reshape(4, len(gridcube.coords))  # xgC -> xG
                    yield i, ao_value, gridcube.non0tab, wv

        aow = None
        nelec = np.zeros(nset)
        excsum = np.zeros(nset)
        vmat = np.zeros((nset, nao, nao))

        if xctype == "GGA":
            ao_deriv = 1
            for i, ao, mask, wv in block_loop(ao_deriv):
                wv[0] *= 0.5  # *.5 because vmat + vmat.T at the end

                aow = _scale_ao(ao, wv)
                _dot_ao_ao_dense(ao[0], aow, None, vmat[i])
            vmat = lib.hermi_sum(vmat, axes=(0, 2, 1))
        else:
            raise NotImplementedError(f"numint.nr_rks for functional {xc_code}")

        if nset == 1:
            vmat = vmat[0]
            nelec = nelec[0]
            excsum = excsum[0]

        if isinstance(dms, np.ndarray):
            dtype = dms.dtype
        else:
            dtype = np.result_type(*dms)
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
        """
        # Get the effective potential for the RKS method.
        # This function is used to get the effective potential for the RKS method.
        # Modified from pyscf.dft.rks.get_veff; See
        # https://github.com/pyscf/pyscf/blob/v2.9.0/pyscf/dft/rks.py
        """
        logger.debug(ks_, "Using modified get_veff")
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
            max_memory = ks_.max_memory - lib.current_memory()[0]
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

        t0 = logger.timer(ks_, "jk", *t0)

        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)
        return vxc

    ks.get_veff = types.MethodType(get_veff, ks)


def get_veff_grad_modified_rks(ks_grad, modeldict):
    """
    Get the method of "Get the effective potential for the RKS Gradients method".
    """

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
        xctype = ni._xc_type(xc_code)
        ao_loc = mol.ao_loc_nr()

        nset = 1
        nao = mol.nao

        vmat = np.zeros((nset, 3, nao, nao))

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

                    wv = np.einsum(
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
                # TODO: replaced with vk_sr which is numerically more stable for
                # inv(int2c2e)
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
