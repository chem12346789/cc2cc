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


def nr_rks(
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
            mol, grids, nao, ao_deriv, max_memory=max_memory
        ):
            for i in range(nset):
                rho_cube = grids.gen_cube_rho_rks(
                    mol, dms, ni=ni, coords=coords_, weights=weights_
                )

                input_mat = torch.tensor(
                    rho_cube,
                    dtype=modelclass.dtype,
                    device=modelclass.device,
                )
                input_mat.requires_grad = True
                output_mat = modelclass.model(input_mat)[:, 0]

                middle_cube = torch.autograd.grad(
                    torch.sum(output_mat),
                    input_mat,
                    create_graph=True,
                )[0]

                middle_mat = (
                    grids.get_center_density(middle_cube).detach().cpu().numpy()
                )
                energy_den = output_mat.detach().cpu().numpy()

                rho = make_rho(i, ao, mask, xctype)
                exc_lda, vxc_lda = ni.eval_xc_eff(
                    "LDA,", rho[0], deriv=1, xctype=ni._xc_type("LDA,")
                )[:2]
                exc_vwn, vxc_vwn = ni.eval_xc_eff(
                    ",VWN3", rho[0], deriv=1, xctype=ni._xc_type(",VWN3")
                )[:2]
                exc_b88, vxc_b88 = ni.eval_xc_eff(
                    "B88,", rho, deriv=1, xctype=ni._xc_type("B88,")
                )[:2]
                exc_lyp, vxc_lyp = ni.eval_xc_eff(
                    ",LYP", rho, deriv=1, xctype=ni._xc_type(",LYP")
                )[:2]

                exc = 0.72 * exc_b88 + 0.81 * exc_lyp + 0.08 * exc_lda + 0.19 * exc_vwn
                vxc = (0.72 + middle_mat[:, 0]) * vxc_b88 + (
                    0.81 + middle_mat[:, 1]
                ) * vxc_lyp
                vxc[[0], :] += (0.08 + middle_mat[:, 2]) * vxc_lda + (
                    0.19 + middle_mat[:, 3]
                ) * vxc_vwn

                if xctype == "LDA":
                    den = rho * weights_
                else:
                    den = rho[0] * weights_
                nelec[i] += den.sum()
                excsum[i] += np.dot(den, exc) + np.dot(weights_, energy_den)
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
        raise NotImplementedError(f"numint.nr_uks for functional {xc_code}")

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


def get_veff_modified(ks, modeldict):
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
    ):
        """
        Get the effective potential for the RKS method.
        This function is used to get the effective potential for the RKS method.
        """
        # print("Using modified get_veff", flush=True)
        if mol is None:
            mol = ks_.mol

        if dm is None:
            dm = ks_.make_rdm1()

        ground_state = isinstance(dm, np.ndarray) and dm.ndim == 2
        ni = ks_._numint

        nelec, exc, vxc = nr_rks(modeldict, ni, mol, ks_.grids, dm, ks_.xc)

        if not ni.libxc.is_hybrid_xc(ks_.xc):
            vk = None
            if (
                ks_._eri is None
                and ks_.direct_scf
                and getattr(vhf_last, "vj", None) is not None
            ):
                ddm = np.asarray(dm) - np.asarray(dm_last)
                vj = ks_.get_j(mol, ddm, hermi)
                vj += vhf_last.vj
            else:
                vj = ks_.get_j(mol, dm, hermi)
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
                if omega != 0:  # For range separated Coulomb
                    vklr = ks_.get_k(mol, ddm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
                vj += vhf_last.vj
                vk += vhf_last.vk
            else:
                vj, vk = ks_.get_jk(mol, dm, hermi)
                vk *= hyb
                if omega != 0:
                    vklr = ks_.get_k(mol, dm, hermi, omega=omega)
                    vklr *= alpha - hyb
                    vk += vklr
            vxc += vj - vk * 0.5

            if ground_state:
                exc -= np.einsum("ij,ji", dm, vk).real * 0.5 * 0.5

        if ground_state:
            ecoul = np.einsum("ij,ji", dm, vj).real * 0.5
        else:
            ecoul = None

        vxc = lib.tag_array(
            vxc,
            ecoul=ecoul,
            exc=exc,
            vj=vj,
            vk=vk,
        )

        return vxc

    if not hasattr(ks.grids, "gen_cube_rho_rks"):
        raise ValueError("Grids does not have gen_cube_rho_rks.")

    ks.get_veff = types.MethodType(get_veff, ks)
