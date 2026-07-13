"""
GPU4PySCF version of the modified UKS hooks used by cc2cc.
"""

# pylint: disable=W0212

import types

import cupy as cp

from gpu4pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.dft.numint import NumInt, _scale_ao, _GDFTOpt
from gpu4pyscf.grad.rks import _gga_grad_sum_
from gpu4pyscf.lib.cupy_helper import tag_array

from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.GridsGPU import GridGPU as Grid, iterate_grid_segments


def _hermi_sum(vmat):
    return vmat + vmat.transpose(0, 2, 1)


def get_veff_modified_uks_gpu(ks, modeldict: ModelClass):
    """
    Get the method of "Get the effective potential for the UKS method".
    """

    def nr_uks(
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
        Obtain the nelec, excsum, and vmat.
        """
        del hermi
        xctype = ni._xc_type(xc_code)

        grids.opt = _GDFTOpt.from_mol(mol)
        _sorted_mol = grids.opt._sorted_mol
        mo_coeff = getattr(dms, "mo_coeff", None)
        mo_occ = getattr(dms, "mo_occ", None)
        dma, dmb = dms
        dma = cp.asarray(dma)
        dmb = cp.asarray(dmb)
        nao = mol.nao

        if mo_coeff is not None and mo_occ is not None:
            mo_coeff = grids.opt.sort_orbitals(mo_coeff, axis=[1])
            dms_sorted = (
                tag_array(dma, mo_coeff=mo_coeff[0], mo_occ=mo_occ[0]),
                tag_array(dmb, mo_coeff=mo_coeff[1], mo_occ=mo_occ[1]),
            )
        else:
            dma = grids.opt.sort_orbitals(dma, axis=[0, 1])
            dmb = grids.opt.sort_orbitals(dmb, axis=[0, 1])
            dms_sorted = (
                tag_array(dma, mo_coeff=None, mo_occ=None),
                tag_array(dmb, mo_coeff=None, mo_occ=None),
            )

        def block_loop(ao_deriv):
            for weights_, coords_ in iterate_grid_segments(
                _sorted_mol,
                grids,
                nao,
                ao_deriv,
                max_memory=max_memory // (2 * modeldict.model.cube_size),
            ):
                gridcube = grids.gen_cube(_sorted_mol, dms_sorted, coords_)
                rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_uks(ni, dms_sorted)
                energy_den, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                energy_den = cp.asarray(energy_den)
                middle_cube = cp.asarray(middle_cube)

                excsum[0] += cp.sum(energy_den)
                wv = cp.einsum(
                    "islpC,piC->slpC",
                    vxc_mat,
                    middle_cube,
                    optimize=True,
                )

                wv = wv.reshape(2, 4, len(gridcube.coords))  # slpC -> slP
                yield ao_value, wv

        nelec = cp.zeros((2, 1))
        excsum = cp.zeros(1)
        vmat = cp.zeros((2, 1, nao, nao))

        if xctype == "GGA":
            ao_deriv = 1
            for ao, wv in block_loop(ao_deriv):
                wv[:, 0] *= 0.5
                wva, wvb = wv

                # ao shape in GPU4PySCF (transpose=False): (comp, nao, ngrids)
                aowa = _scale_ao(ao, wva)
                aowb = _scale_ao(ao, wvb)
                vmat[0, 0] += cp.dot(ao[0], aowa.T)
                vmat[1, 0] += cp.dot(ao[0], aowb.T)

            vmat = _hermi_sum(vmat.reshape((-1, nao, nao))).reshape(2, 1, nao, nao)
        else:
            raise NotImplementedError(f"numint.nr_uks for functional {xc_code}")

        if isinstance(dma, cp.ndarray) and dma.ndim == 2:
            vmat = vmat[:, 0]
            nelec = nelec.reshape(2)
            excsum = excsum[0]

        if grids.opt is not None:
            vmat[0] = grids.opt.unsort_orbitals(vmat[0], axis=[0, 1])
            vmat[1] = grids.opt.unsort_orbitals(vmat[1], axis=[0, 1])
            grids.opt = None

        return nelec, excsum, vmat

    def get_veff(
        ks_,
        mol=None,
        dm=None,
        dm_last=0,
        vhf_last=0,
        hermi=1,
    ):
        logger.debug(ks_, "Using modified get_veff (gpu4pyscf)")
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

        t0 = (logger.process_clock(), logger.perf_counter())
        ground_state = dm.ndim == 3 and dm.shape[0] == 2

        ni = ks_._numint
        if hermi == 2:  # because rho = 0
            n, exc, vxc = (0, 0), 0, 0
        else:
            n, exc, vxc = nr_uks(
                modeldict,
                ni,
                mol,
                ks_.grids,
                ks_.xc,
                dm,
                max_memory=4000,
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
            _dm = cp.asarray(dm) - cp.asarray(dm_last)
        else:
            _dm = dm

        if not ni.libxc.is_hybrid_xc(ks_.xc):
            vk = None
            vj = ks_.get_j(mol, _dm[0] + _dm[1], hermi)
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
            vj = vj[0] + vj[1]
            if incremental_jk:
                vj += vhf_last.vj
                vk += vhf_last.vk
            vxc += vj - vk

            if ground_state:
                exc -= (
                    cp.einsum("ij,ji", dm[0], vk[0]).real
                    + cp.einsum("ij,ji", dm[1], vk[1]).real
                ) * 0.5
        if ground_state:
            ecoul = cp.einsum("ij,ji", dm[0] + dm[1], vj).real * 0.5
        else:
            ecoul = None

        t0 = logger.timer(ks_, "  jk", *t0)
        return tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

    ks.get_veff = types.MethodType(get_veff, ks)


def get_veff_grad_modified_uks_gpu(ks_grad, modeldict: ModelClass):
    """
    Get the method of "Get the effective potential for the UKS Gradients method".
    """
    raise NotImplementedError("get_veff_grad_modified_uks_gpu is not implemented yet")

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
        nao = mol.nao

        vmat = cp.zeros((2, 3, nao, nao))
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
                gridcube = grids.gen_cube(mol, dms, coords_, mask)
                rho_cube, vxc_mat, ao_value = gridcube.gen_cube_rho_uks(
                    ni, dms, ao_deriv=ao_deriv
                )
                _, middle_cube = modeldict.eval_xc_eff(rho_cube, weights_)
                middle_cube = cp.asarray(middle_cube)
                wv = cp.einsum(
                    "islpC,piC->slpC",
                    vxc_mat,
                    middle_cube,
                    optimize=True,
                )
                wv[:, 0] *= 0.5
                wv = wv.reshape(2, 4, len(gridcube.coords))  # slpC -> slP

                _gga_grad_sum_(vmat[0], mol, ao_value, wv[0], None, ao_loc)
                _gga_grad_sum_(vmat[1], mol, ao_value, wv[1], None, ao_loc)

        exc = cp.zeros((mol.natm, 3))
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

        exc, vxc = get_vxc(
            ni,
            mol,
            ks_grad_.grids,
            mf.xc,
            dm,
            ks_grad_.max_memory,
            verbose=ks_grad_.verbose,
        )
        t0 = logger.timer(ks_grad_, "vxc", *t0)

        if not ni.libxc.is_hybrid_xc(mf.xc):
            vj = ks_grad_.get_j(mol, dm)
            vxc += vj[0] + vj[1]
            if ks_grad_.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
            vj, vk = ks_grad_.get_jk(mol, dm)
            if ks_grad_.auxbasis_response:
                vk.aux = vk.aux * hyb
            vk[:] *= hyb  # inplace * for vk[:] to keep the .aux tag
            if omega != 0:  # For range separated Coulomb operator
                vk_lr = ks_grad_.get_k(mol, dm, omega=omega)
                vk[:] += vk_lr * (alpha - hyb)
                if ks_grad_.auxbasis_response:
                    vk.aux[:] += vk_lr.aux * (alpha - hyb)
            vxc += vj[0] + vj[1] - vk
            if ks_grad_.auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
                e1_aux -= cp.trace(vk.aux, axis1=0, axis2=1)

        if ks_grad_.auxbasis_response:
            logger.debug1(ks_grad_, "sum(auxbasis response) %s", e1_aux.sum(axis=0))
            vxc = tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        else:
            vxc = tag_array(vxc, exc1_grid=exc)
        return vxc

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)


__all__ = [
    "get_veff_modified_uks_gpu",
    "get_veff_grad_modified_uks_gpu",
]
