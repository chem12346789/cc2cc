"""GPU model-SCF hooks for explicitly selected restricted or unrestricted spin."""

# pylint: disable=W0212

import types

import cupy as cp
from gpu4pyscf.dft.numint import _scale_ao
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import tag_array, transpose_sum

from cc2cc.utils.GridsGPU import (
    eval_xc_eff_gpu,
    get_cached_gdftopt,
    iterate_grid_segments,
)
from cc2cc.utils.ModelClass import ModelClass
from cc2cc.utils.helper.runtime.env_var import MAX_GPU_JK_NAO
from cc2cc.utils.helper.runtime.xc_profile import XCStageProfiler


def get_veff_modified_gpu(
    ks, modeldict: ModelClass, max_memory_gpu=4000, *, unrestricted: bool
) -> None:
    """Install the GPU model potential with an explicitly selected spin mode."""
    nspin = 2 if unrestricted else 1

    def nr_model(ni, mol, grids, xc_code, dms, max_memory):
        profile = XCStageProfiler("UKS" if unrestricted else "RKS")
        xctype = ni._xc_type(xc_code)

        grids.opt = get_cached_gdftopt(grids, mol)
        sorted_mol = grids.opt._sorted_mol
        mo_coeff = getattr(dms, "mo_coeff", None)
        mo_occ = getattr(dms, "mo_occ", None)
        if unrestricted:
            dma, dmb = dms
            dma = cp.asarray(dma)
            dmb = cp.asarray(dmb)
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
        else:
            if mo_coeff is not None:
                mo_coeff = grids.opt.sort_orbitals(mo_coeff, axis=[0])
            else:
                assert dms.ndim == 2
                dms = cp.asarray(dms)
                dms = grids.opt.sort_orbitals(dms, axis=[0, 1])
            dms_sorted = tag_array(dms, mo_coeff=mo_coeff, mo_occ=mo_occ)

        nao = mol.nao

        def block_loop(ao_deriv):
            for weights, coords in iterate_grid_segments(
                sorted_mol,
                grids,
                nao,
                ao_deriv,
                max_memory=max_memory // (2 * modeldict.model.cube_size),
            ):
                gridcube = grids.gen_cube(sorted_mol, dms_sorted, coords)
                profile.record_block(len(coords), len(gridcube.coords))
                profile.mark("cube coordinates")
                cube_rho = (
                    gridcube.gen_cube_rho_uks
                    if unrestricted
                    else gridcube.gen_cube_rho_rks
                )
                rho_cube, vxc_mat, ao = cube_rho(ni, dms_sorted, profile=profile)
                energy_den, middle_cube = eval_xc_eff_gpu(modeldict, rho_cube, weights)
                profile.mark("model + DLPack")

                excsum[0] += cp.sum(energy_den)
                if unrestricted:
                    wv = cp.einsum(
                        "islpC,piC->slpC", vxc_mat, middle_cube, optimize=True
                    )
                else:
                    wv = cp.einsum(
                        "ixgC,giC->xgC", vxc_mat, middle_cube, optimize=True
                    )[None]
                wv = wv.reshape(nspin, 4, len(gridcube.coords))
                profile.mark("XC contraction")

                yield ao, wv
                del ao, wv, rho_cube, vxc_mat, energy_den, middle_cube, gridcube
                profile.mark("generator release")

        nelec = cp.zeros((nspin, 1), dtype=cp.float64)
        excsum = cp.zeros(1, dtype=cp.float64)
        vmat = cp.zeros((nspin, 1, nao, nao), dtype=cp.float64)
        profile.mark("setup")

        if xctype == "GGA":
            aow_buffer = None
            for ao, wv in block_loop(1):
                profile.mark("block handoff")
                if aow_buffer is None or aow_buffer.size < ao[0].size:
                    aow_buffer = cp.empty(ao[0].size, dtype=cp.float64)
                    profile.mark("AO scratch allocation")
                profile.mark("AO scratch check")
                wv[:, 0] *= 0.5  # The density term is doubled by symmetrization.
                profile.record_ao_layout(ao, wv[0], aow_buffer)
                profile.mark("AO weight preparation")
                for spin in range(nspin):
                    aow = _scale_ao(ao, wv[spin], out=aow_buffer)
                    profile.mark("AO weighting")
                    vmat[spin, 0] += cp.dot(ao[0], aow.T)
                    profile.mark("matrix GEMM")
                del ao, wv, aow
                profile.mark("caller release")

            vmat = vmat.reshape((-1, nao, nao))
            if unrestricted:
                vmat = vmat + vmat.transpose(0, 2, 1)
            else:
                vmat = transpose_sum(vmat)
            vmat = vmat.reshape(nspin, 1, nao, nao)
        else:
            mode = "uks" if unrestricted else "rks"
            raise NotImplementedError(f"numint.nr_{mode} for functional {xc_code}")

        if unrestricted:
            if isinstance(dma, cp.ndarray) and dma.ndim == 2:
                vmat = vmat[:, 0]
                nelec = nelec.reshape(2)
                excsum = excsum[0]
            if grids.opt is not None:
                vmat[0] = grids.opt.unsort_orbitals(vmat[0], axis=[0, 1])
                vmat[1] = grids.opt.unsort_orbitals(vmat[1], axis=[0, 1])
                grids.opt = None
        else:
            vmat = vmat[0, 0]
            nelec = nelec[0, 0]
            excsum = excsum[0]
            if grids.opt is not None:
                vmat = grids.opt.unsort_orbitals(vmat, axis=[0, 1])
                grids.opt = None

        profile.mark("symmetry + unsort")
        profile.report(logger, ks)
        return nelec, excsum, vmat

    def get_veff(ks_, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        logger.info(ks_, "Using modified get_veff (gpu4pyscf)")
        if mol is None:
            mol = ks_.mol
        if dm is None:
            dm = ks_.make_rdm1()

        t0 = (logger.process_clock(), logger.perf_counter())
        ground_state = (
            dm.ndim == 3 and dm.shape[0] == 2
            if unrestricted
            else isinstance(dm, cp.ndarray) and dm.ndim == 2
        )
        ni = ks_._numint
        if hermi == 2:
            n, exc, vxc = ((0, 0) if unrestricted else 0), 0, 0
        else:
            n, exc, vxc = nr_model(
                ni, mol, ks_.grids, ks_.xc, dm, max_memory=max_memory_gpu
            )
            logger.info(ks_, "nelec by numeric integration = %s", n)
            t0 = logger.timer(ks_, "  vxc" if unrestricted else "vxc", *t0)
            if ks_.do_nlc():
                ks_.initialize_grids(mol, dm)
                nlc_code = ks_.xc if ni.libxc.is_nlc(ks_.xc) else ks_.nlc
                _, enlc, vnlc = ni.nr_nlc_vxc(
                    mol, ks_.nlcgrids, nlc_code, dm, max_memory=max_memory_gpu
                )
                exc += enlc
                vxc += vnlc
                logger.info(ks_, "NLC (%s) energy = %.12g Ha", nlc_code, enlc)

        if mol.nao > MAX_GPU_JK_NAO:
            logger.info(
                ks_,
                "Large system detected: %d AO, using CPU for J/K calculations",
                mol.nao,
            )
            if getattr(ks_, "ks_jk", None) is None:
                ks_.ks_jk = ks_.to_cpu()

            def get_j(mol, dm, hermi):
                dm = cp.asnumpy(dm)
                return cp.asarray(ks_.ks_jk.get_j(mol, dm, hermi))

            def get_k(mol, dm, hermi, omega=0):
                dm = cp.asnumpy(dm)
                return cp.asarray(ks_.ks_jk.get_k(mol, dm, hermi, omega=omega))

            def get_jk(mol, dm, hermi):
                dm = cp.asnumpy(dm)
                return cp.asarray(ks_.ks_jk.get_jk(mol, dm, hermi))

        else:
            get_j = ks_.get_j
            get_k = ks_.get_k
            get_jk = ks_.get_jk

        incremental_jk = (
            ks_._eri is None
            and ks_.direct_scf
            and getattr(vhf_last, "vj", None) is not None
        )
        _dm = cp.asarray(dm) - cp.asarray(dm_last) if incremental_jk else dm
        if not ni.libxc.is_hybrid_xc(ks_.xc):
            vk = None
            j_dm = _dm[0] + _dm[1] if unrestricted else _dm
            vj = get_j(mol, j_dm, hermi)
            if incremental_jk:
                vj += vhf_last.vj
            vxc += vj
        else:
            omega, alpha, hyb = ni.rsh_and_hybrid_coeff(ks_.xc, spin=mol.spin)
            if omega == 0:
                vj, vk = get_jk(mol, _dm, hermi)
                vk *= hyb
            elif alpha == 0:
                vj = get_j(mol, _dm, hermi)
                vk = get_k(mol, _dm, hermi, omega=-omega)
                vk *= hyb
            elif hyb == 0:
                vj = get_j(mol, _dm, hermi)
                vk = get_k(mol, _dm, hermi, omega=omega)
                vk *= alpha
            else:
                vj, vk = get_jk(mol, _dm, hermi)
                vk *= hyb
                vklr = get_k(mol, _dm, hermi, omega=omega)
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
                        cp.einsum("ij,ji", dm[0], vk[0]).real
                        + cp.einsum("ij,ji", dm[1], vk[1]).real
                    ) * 0.5
            else:
                vxc += vj - vk * 0.5
                if ground_state:
                    exc -= cp.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            total_dm = dm[0] + dm[1] if unrestricted else dm
            ecoul = cp.einsum("ij,ji", total_dm, vj).real * 0.5
        else:
            ecoul = None
        logger.timer(ks_, "  jk" if unrestricted else "jk", *t0)
        return tag_array(vxc, ecoul=ecoul, exc=exc, vj=vj, vk=vk)

    ks.get_veff = types.MethodType(get_veff, ks)


def get_veff_grad_modified_gpu(
    ks_grad, modeldict: ModelClass, *, unrestricted: bool
) -> None:
    """GPU model gradients are not implemented for either spin treatment."""
    mode = "UKS" if unrestricted else "RKS"
    raise NotImplementedError(
        f"GPU model gradients are not implemented for {mode}"
    )


__all__ = [
    "get_veff_modified_gpu",
    "get_veff_grad_modified_gpu",
]
