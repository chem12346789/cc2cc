"""CPU model-SCF potential and gradient hooks for either spin treatment."""

# pylint: disable=W0212

import types

import numpy as np
from pyscf import lib
from pyscf.dft.numint import _dot_ao_ao_dense, _scale_ao
from pyscf.grad import rks as rks_grad
from pyscf.grad.rks import _gga_grad_sum_
from pyscf.lib import logger

from cc2cc.utils.Grids import iterate_grid_response_segments, iterate_grid_segments
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.helper.chemistry.tensor_helper import parallel_einsum


def get_veff_modified(ks, modeldict: ModelClass, *, unrestricted: bool) -> None:
    """Install the CPU model potential with an explicitly selected spin mode."""
    nspin = 2 if unrestricted else 1

    def nr_model(ni, mol, grids, xc_code, dms, max_memory):
        xctype = ni._xc_type(xc_code)
        if unrestricted:
            dma, dmb = dms

        nset = 1
        nao = mol.nao
        nelec = np.zeros((nspin, nset))
        excsum = np.zeros(nset)
        vmat = np.zeros((nspin, nset, nao, nao))

        if xctype != "GGA":
            raise NotImplementedError(f"numint.nr_rks for functional {xc_code}")

        for mask, weights, coords in iterate_grid_segments(
            mol,
            grids,
            nao,
            1,
            max_memory=max_memory // (2 * modeldict.model.cube_size),
            non0tab=None,
        ):
            for i in range(nset):
                gridcube = grids.gen_cube(mol, dms, coords, mask)
                cube_rho = (
                    gridcube.gen_cube_rho_uks
                    if unrestricted
                    else gridcube.gen_cube_rho_rks
                )
                rho_cube, vxc_mat, ao = cube_rho(ni, dms)
                energy_den, middle_cube = modeldict.eval_xc_eff(rho_cube, weights)
                excsum[i] += np.sum(energy_den)
                if unrestricted:
                    wv = parallel_einsum("islpC,piC->slpC", vxc_mat, middle_cube)
                else:
                    wv = parallel_einsum("ixgC,giC->xgC", vxc_mat, middle_cube)[
                        None
                    ]
                wv = wv.reshape(nspin, 4, len(gridcube.coords))
                wv[:, 0] *= 0.5  # The density term is doubled by symmetrization.
                for spin in range(nspin):
                    aow = _scale_ao(ao, wv[spin])
                    _dot_ao_ao_dense(ao[0], aow, None, vmat[spin, i])

        vmat = lib.hermi_sum(vmat.reshape((-1, nao, nao)), axes=(0, 2, 1)).reshape(
            nspin, nset, nao, nao
        )
        if unrestricted:
            if isinstance(dma, np.ndarray) and dma.ndim == 2:
                vmat = vmat[:, 0]
                nelec = nelec.reshape(2)
                excsum = excsum[0]
            dtype = np.result_type(dma, dmb)
        else:
            vmat = vmat[0, 0]
            nelec = nelec[0, 0]
            excsum = excsum[0]
            dtype = dms.dtype if isinstance(dms, np.ndarray) else np.result_type(*dms)
        if vmat.dtype != dtype:
            vmat = np.asarray(vmat, dtype=dtype)

        return nelec, excsum, vmat

    def get_veff(ks_, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        logger.debug(ks_, "Using modified get_veff")
        if mol is None:
            mol = ks_.mol
        if dm is None:
            dm = ks_.make_rdm1()

        t0 = (logger.process_clock(), logger.perf_counter())
        ground_state = (
            dm.ndim == 3 and dm.shape[0] == 2
            if unrestricted else isinstance(dm, np.ndarray) and dm.ndim == 2
        )
        ni = ks_._numint
        if hermi == 2:
            n, exc, vxc = ((0, 0) if unrestricted else 0), 0, 0
        else:
            max_memory = ks_.max_memory - lib.current_memory()[0]
            n, exc, vxc = nr_model(
                ni, mol, ks_.grids, ks_.xc, dm, max_memory=max_memory
            )
            logger.debug(ks_, "nelec by numeric integration = %s", n)
            if ks_.do_nlc():
                ks_.initialize_grids(mol, dm)
                nlc_code = ks_.xc if ni.libxc.is_nlc(ks_.xc) else ks_.nlc
                nlc_dm = dm[0] + dm[1] if unrestricted else dm
                _, enlc, vnlc = ni.nr_nlc_vxc(
                    mol, ks_.nlcgrids, nlc_code, nlc_dm, max_memory=max_memory
                )
                exc += enlc
                vxc += vnlc
                logger.info(ks_, "NLC (%s) energy = %.12g Ha", nlc_code, enlc)
            t0 = logger.timer(ks_, "  vxc" if unrestricted else "vxc", *t0)

        incremental_jk = (
            ks_._eri is None
            and ks_.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        )
        _dm = np.asarray(dm) - np.asarray(dm_last) if incremental_jk else dm
        jk_ks = ks if unrestricted else ks_
        if not ni.libxc.is_hybrid_xc(jk_ks.xc):
            vk = None
            j_dm = _dm[0] + _dm[1] if unrestricted else _dm
            vj = jk_ks.get_j(mol, j_dm, hermi)
            if incremental_jk:
                vj += vhf_last.vj
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(jk_ks.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = jk_ks.get_jk(mol, _dm, hermi)
                vk *= hyb
            elif alpha == 0:
                vj = jk_ks.get_j(mol, _dm, hermi)
                vk = jk_ks.get_k(mol, _dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:
                vj = jk_ks.get_j(mol, _dm, hermi)
                vk = jk_ks.get_k(mol, _dm, hermi, omega=omega)
                vk *= alpha
            else:
                vj, vk = jk_ks.get_jk(mol, _dm, hermi)
                vk *= hyb
                vklr = jk_ks.get_k(mol, _dm, hermi, omega=omega)
                vklr *= alpha - hyb
                vk += vklr
            if unrestricted:
                vj = vj[0] + vj[1]
            if incremental_jk:
                vj += vhf_last.vj
                vk += vhf_last.vk
            if unrestricted:
                vxc += vj - vk
                if ground_state:
                    exc -= (
                        parallel_einsum("ij,ji", dm[0], vk[0]).real
                        + parallel_einsum("ij,ji", dm[1], vk[1]).real
                    ) * 0.5
            else:
                vxc += vj - vk * 0.5
                if ground_state:
                    exc -= parallel_einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            total_dm = dm[0] + dm[1] if unrestricted else dm
            ecoul = parallel_einsum("ij,ji", total_dm, vj).real * 0.5
        else:
            ecoul = None
        logger.timer(ks_, "  jk" if unrestricted else "jk", *t0)
        return lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

    ks.get_veff = types.MethodType(get_veff, ks)


def _gradient_jk(
    ks_grad, ni, xc, mol, dm, *, unrestricted: bool, auxbasis_response: bool
):
    e1_aux = None
    if not ni.libxc.is_hybrid_xc(xc):
        vj = ks_grad.get_j(mol, dm)
        vxc = vj[0] + vj[1] if unrestricted else vj
        if auxbasis_response:
            e1_aux = vj.aux.sum((0, 1))
    else:
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(xc, spin=mol.spin)
        vj, vk = ks_grad.get_jk(mol, dm)
        if auxbasis_response:
            if unrestricted:
                vk.aux = vk.aux * hyb
            else:
                vk.aux *= hyb
        vk[:] *= hyb  # Retain PySCF's auxiliary-response tag.
        if omega != 0:
            vk_lr = ks_grad.get_k(mol, dm, omega=omega)
            vk[:] += vk_lr * (alpha - hyb)
            if auxbasis_response:
                vk.aux[:] += vk_lr.aux * (alpha - hyb)
        if unrestricted:
            vxc = vj[0] + vj[1] - vk
            if auxbasis_response:
                e1_aux = vj.aux.sum((0, 1))
                e1_aux -= np.trace(vk.aux, axis1=0, axis2=1)
        else:
            vxc = vj - vk * 0.5
            if auxbasis_response:
                e1_aux = (vj.aux - vk.aux * 0.5).sum((0, 1))
    return vxc, e1_aux


def get_veff_grad_modified(
    ks_grad, modeldict: ModelClass, *, unrestricted: bool
) -> None:
    """Install the CPU model gradient with an explicitly selected spin mode."""
    nspin = 2 if unrestricted else 1

    def get_vxc(ni, mol, grids, xc_code, dms, max_memory):
        xctype = ni._xc_type(xc_code)
        ao_loc = mol.ao_loc_nr()
        nao = mol.nao
        vmat = np.zeros((nspin, 3, nao, nao))
        # Preserve the spin-dependent fixed-grid response tags.
        exc = (
            np.zeros((mol.natm, 3))
            if unrestricted or ks_grad.grid_response else None
        )

        if xctype == "GGA":
            ao_deriv = 2
            memory = max_memory // (2 * modeldict.model.cube_size)
            if ks_grad.grid_response:
                blocks = iterate_grid_response_segments(
                    mol, grids, nao, ao_deriv, memory
                )
            else:
                blocks = (
                    (None, mask, weights, coords, None)
                    for mask, weights, coords in iterate_grid_segments(
                        mol, grids, nao, ao_deriv, max_memory=memory, non0tab=None
                    )
                )
            for atom, mask, weights, coords, weight1 in blocks:
                gridcube = grids.gen_cube(mol, dms, coords, mask)
                cube_rho = (
                    gridcube.gen_cube_rho_uks
                    if unrestricted
                    else gridcube.gen_cube_rho_rks
                )
                rho_cube, vxc_mat, ao = cube_rho(ni, dms, ao_deriv=ao_deriv)
                if ks_grad.grid_response:
                    _, middle_cube, weight_derivative = modeldict.eval_xc_eff(
                        rho_cube, weights, weight_response=True
                    )
                    exc += parallel_einsum("p,axp->ax", weight_derivative, weight1)
                else:
                    _, middle_cube = modeldict.eval_xc_eff(rho_cube, weights)
                if unrestricted:
                    wv = parallel_einsum("islpC,piC->slpC", vxc_mat, middle_cube)
                else:
                    wv = parallel_einsum("ilpC,piC->lpC", vxc_mat, middle_cube)[
                        None
                    ]
                wv[:, 0] *= 0.5
                wv = wv.reshape(nspin, 4, len(gridcube.coords))
                vtmp = np.zeros_like(vmat)
                for spin in range(nspin):
                    _gga_grad_sum_(vtmp[spin], mol, ao, wv[spin], None, ao_loc)
                vmat += vtmp
                if ks_grad.grid_response:
                    if unrestricted:
                        exc[atom] += 2 * parallel_einsum("sxij,sji->x", vtmp, dms)
                    else:
                        exc[atom] += 2 * parallel_einsum("xij,ji->x", vtmp[0], dms)

        return exc, -(vmat if unrestricted else vmat[0])

    def get_veff(ks_grad_, mol=None, dm=None):
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        mf = ks_grad_.base
        ni = mf._numint
        max_memory = ks_grad_.max_memory * 0.9 - lib.current_memory()[0]
        exc, vxc = get_vxc(
            ni, mol, ks_grad_.grids, mf.xc, dm, max_memory=max_memory
        )
        if mf.do_nlc():
            mf.initialize_grids(mol, dm)
            nlc_code = mf.xc if ni.libxc.is_nlc(mf.xc) else mf.nlc
            nlc_vxc = (
                rks_grad.get_nlc_vxc_full_response
                if ks_grad_.grid_response else rks_grad.get_nlc_vxc
            )
            nlc_dm = dm[0] + dm[1] if unrestricted else dm
            enlc, vnlc = nlc_vxc(
                ni, mol, mf.nlcgrids, nlc_code, nlc_dm,
                max_memory=max_memory, verbose=ks_grad_.verbose,
            )
            vxc += vnlc
            if ks_grad_.grid_response:
                exc += enlc

        jk_grad = ks_grad if unrestricted else ks_grad_
        logger.timer(jk_grad, "vxc", *t0)
        auxbasis_response = jk_grad.auxbasis_response
        vjk, e1_aux = _gradient_jk(
            jk_grad, ni, mf.xc, mol, dm,
            unrestricted=unrestricted, auxbasis_response=auxbasis_response,
        )
        vxc += vjk
        if auxbasis_response:
            logger.debug1(jk_grad, "sum(auxbasis response) %s", e1_aux.sum(axis=0))
            return lib.tag_array(vxc, exc1_grid=exc, aux=e1_aux)
        return lib.tag_array(vxc, exc1_grid=exc)

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)


def get_veff_grad_modified_zeros(ks_grad, *, unrestricted: bool) -> None:
    """Install the Coulomb/exchange-only CPU gradient hook."""

    def get_veff(ks_grad_, mol=None, dm=None):
        if mol is None:
            mol = ks_grad_.mol
        if dm is None:
            dm = ks_grad_.base.make_rdm1()
        t0 = (logger.process_clock(), logger.perf_counter())
        mf = ks_grad_.base
        ni = mf._numint
        auxbasis_response = getattr(ks_grad_, "auxbasis_response", False)
        logger.timer(ks_grad_, "vxc", *t0)
        vxc, e1_aux = _gradient_jk(
            ks_grad_, ni, mf.xc, mol, dm,
            unrestricted=unrestricted, auxbasis_response=auxbasis_response,
        )
        if unrestricted and not ni.libxc.is_hybrid_xc(mf.xc):
            vxc = np.broadcast_to(vxc, (2, *vxc.shape)).copy()
        exc1_grid = np.zeros((mol.natm, 3)) if ks_grad_.grid_response else None
        if auxbasis_response:
            return lib.tag_array(vxc, exc1_grid=exc1_grid, aux=e1_aux)
        return lib.tag_array(vxc, exc1_grid=exc1_grid)

    ks_grad.get_veff = types.MethodType(get_veff, ks_grad)
