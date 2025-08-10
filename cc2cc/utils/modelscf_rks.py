"""
This module provides the modified RKS method for model-based DFT calculations.
The ``uncommment commment'' is the original docstring.
"""

# pylint: disable=W0212

import types

import numpy as np

from pyscf import lib
from pyscf.lib import logger
from pyscf.dft.numint import (
    _scale_ao_sparse,
    _dot_ao_ao_sparse,
    # _tau_dot_sparse,
)
from pyscf.dft.gen_grid import NBINS


def nr_rks(
    modelclass,
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
    vmat = np.zeros((nset, nao, nao))

    def block_loop(ao_deriv):
        for ao, mask, weights_, coords_ in ni.block_loop(
            mol, grids, nao, ao_deriv, max_memory=max_memory, non0tab=grids.non0tab
        ):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)
                energy_den, vxc = modelclass.eval_xc_eff(
                    rho, ni, dms, grids, coords_, mask
                )

                if xctype == "LDA":
                    den = rho * weights_
                else:
                    den = rho[0] * weights_
                nelec[i] += den.sum()
                excsum[i] += np.dot(weights_, energy_den)
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
        nelec = nelec[0]
        excsum = excsum[0]
        vmat = vmat[0]

    if isinstance(dms, np.ndarray):
        dtype = dms.dtype
    else:
        dtype = np.result_type(*dms)
    if vmat.dtype != dtype:
        vmat = np.asarray(vmat, dtype=dtype)
    return nelec, excsum, vmat


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

        Coulomb + XC functional

        .. note::
            This function will modify the input ks object.

        Args:
            ks : an instance of :class:`RKS`
                XC functional are controlled by ks.xc attribute.  Attribute
                ks.grids might be initialized.
            dm : ndarray or list of ndarrays
                A density matrix or a list of density matrices

        Kwargs:
            dm_last : ndarray or a list of ndarrays or 0
                The density matrix baseline.  If not 0, this function computes the
                increment of HF potential w.r.t. the reference HF potential matrix.
            vhf_last : ndarray or a list of ndarrays or 0
                The reference Vxc potential matrix.
            hermi : int
                Whether J, K matrix is hermitian

                | 0 : no hermitian or symmetric
                | 1 : hermitian
                | 2 : anti-hermitian

        Returns:
            matrix Veff = J + Vxc.  Veff can be a list matrices, if the input
            dm is a list of density matrices.
        """
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

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
