import numpy as np

import pyscf
import pyscf.dft

from cc2cc.utils.zmp import RZMP, UZMP
from cc2cc.utils.mol import AU2KCALMOL
from cc2cc.utils.TestDataDFT import diff_rho, diff_I_value

LEVEL_SHIFT = 3
MAX_CYCLE = 20000


def get_zmp_rks(
    mol, dm_tar, dm_dft, grids, max_l=4, verbose=0, start_l=0, max_cycle=MAX_CYCLE
):
    if dm_tar is None:
        return None, None

    mzmp = RZMP(mol, dm_tar, grids, dftxc=1, xc="b3lyp")
    mzmp.diis_space = 64
    mzmp.verbose = verbose
    mzmp.max_cycle = max_cycle

    ao = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
    # hatree part from dm1_tar
    rho_tar = pyscf.dft.numint.eval_rho(mol, ao, dm_tar, xctype="LDA")
    print("Start hatree part...")
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    hatree_cc_grids = 0.5 * np.einsum("pij,ij->p", int1e_grids, dm_tar) * rho_tar
    # hatree part from dm1_dft
    rho_dft = pyscf.dft.numint.eval_rho(mol, ao, dm_dft, xctype="LDA")
    hatree_dft_grids = 0.5 * np.einsum("pij,ij->p", int1e_grids, dm_dft) * rho_dft

    nuc_grids = np.zeros_like(grids.weights)
    for i, coord in enumerate(grids.coords):
        for i_atom in range(mol.natm):
            distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
            if distance > 1e-8:
                nuc_grids[i] -= mol.atom_charges()[i_atom] / distance
    nuc_tar_grids = rho_tar * nuc_grids
    nuc_dft_grids = rho_dft * nuc_grids

    diff_rho_best = None
    dm_zmp_old = None
    converged = True

    for l in np.linspace(start_l, max_l, max_l - start_l + 1):
        mzmp.level_shift = LEVEL_SHIFT**l
        print(f"Running ZMP with lambda_ZMP = {2**l}")
        mzmp.zscf(2**l)
        dm_zmp = mzmp.make_rdm1()
        zmp_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_zmp, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_dft, unit="A.U.")
        tar_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_tar, unit="A.U.")
        print(
            f"ZMP, rho diff = {diff_rho(mol, dm_zmp, dm_tar, grids):.3e} "
            f"I value = {diff_I_value(mol, dm_zmp, dm_tar, grids):.3e} "
            f"dipole diff = {np.linalg.norm(zmp_dipole - tar_dipole):.3e} ",
            flush=True,
        )
        print(
            f"DFT, rho diff = {diff_rho(mol, dm_dft, dm_tar, grids):.3e} "
            f"I value = {diff_I_value(mol, dm_dft, dm_tar, grids):.3e} "
            f"dipole diff = {np.linalg.norm(dft_dipole - tar_dipole):.3e} ",
            flush=True,
        )
        if (
            diff_rho_best is None
            or diff_rho(mol, dm_zmp, dm_tar, grids) < diff_rho_best
        ):
            diff_rho_best = diff_rho(mol, dm_zmp, dm_tar, grids)
        elif diff_rho(mol, dm_zmp, dm_tar, grids) > diff_rho_best * 10:
            print("Warning: rho difference increased significantly, may be diverged.")
            converged = False
            break

        # hatree part from dm1_zmp
        rho_zmp = pyscf.dft.numint.eval_rho(mol, ao, dm_zmp, xctype="LDA")
        hatree_zmp_grids = 0.5 * np.einsum("pij,ij->p", int1e_grids, dm_zmp) * rho_zmp
        print(
            f"Difference in Hartree grids between CCSD: DFT {np.sum(np.abs(hatree_cc_grids - hatree_dft_grids) * grids.weights) * AU2KCALMOL}, ZMP {np.sum(np.abs(hatree_cc_grids - hatree_zmp_grids) * grids.weights) * AU2KCALMOL}"
        )

        nuc_zmp_grids = rho_zmp * nuc_grids
        print(
            f"Difference in nuclear grids between CCSD: DFT {np.sum(np.abs(nuc_tar_grids - nuc_dft_grids) * grids.weights) * AU2KCALMOL}, ZMP {np.sum(np.abs(nuc_tar_grids - nuc_zmp_grids) * grids.weights) * AU2KCALMOL}"
        )
        dm_zmp_old = dm_zmp.copy()

    if converged:
        dm_zmp = mzmp.make_rdm1()
        return mzmp, dm_zmp

    print("Warning: rho difference increased significantly, may be diverged.")
    return mzmp, dm_zmp_old


def get_zmp_uks(
    mol, dm_tar, dm_dft, grids, max_l=4, verbose=0, start_l=0, max_cycle=MAX_CYCLE
):
    if dm_tar is None:
        return None, None

    mzmp = UZMP(mol, dm_tar, grids, dftxc=1, xc="b3lyp")
    mzmp.diis_space = 64
    mzmp.verbose = verbose
    mzmp.max_cycle = max_cycle

    ao = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=0)
    # hatree part from dm1_tar
    rho_tar = [
        pyscf.dft.numint.eval_rho(mol, ao, dm_tar[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao, dm_tar[1], xctype="LDA"),
    ]
    print("Start hatree part...")
    int1e_grids = mol.intor("int1e_grids", grids=grids.coords)
    hatree_cc_grids = (
        0.5
        * np.einsum("pij,ij->p", int1e_grids, dm_tar[0] + dm_tar[1])
        * (rho_tar[0] + rho_tar[1])
    )
    # hatree part from dm1_dft
    rho_dft = [
        pyscf.dft.numint.eval_rho(mol, ao, dm_dft[0], xctype="LDA"),
        pyscf.dft.numint.eval_rho(mol, ao, dm_dft[1], xctype="LDA"),
    ]
    hatree_dft_grids = (
        0.5
        * np.einsum("pij,ij->p", int1e_grids, dm_dft[0] + dm_dft[1])
        * (rho_dft[0] + rho_dft[1])
    )

    nuc_grids = np.zeros_like(grids.weights)
    for i, coord in enumerate(grids.coords):
        for i_atom in range(mol.natm):
            distance = np.linalg.norm(mol.atom_coords()[i_atom] - coord)
            if distance > 1e-8:
                nuc_grids[i] -= mol.atom_charges()[i_atom] / distance
    nuc_tar_grids = (rho_tar[0] + rho_tar[1]) * nuc_grids
    nuc_dft_grids = (rho_dft[0] + rho_dft[1]) * nuc_grids

    diff_rho_best = None
    dm_zmp_old = None
    converged = True

    for l in np.linspace(start_l, max_l, max_l - start_l + 1):
        mzmp.level_shift = LEVEL_SHIFT**l
        print(f"Running ZMP with lambda_ZMP = {2**l}")
        mzmp.zscf(2**l)
        dm_zmp = mzmp.make_rdm1()
        zmp_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_zmp, unit="A.U.")
        dft_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_dft, unit="A.U.")
        tar_dipole = pyscf.scf.hf.dip_moment(mol=mol, dm=dm_tar, unit="A.U.")
        print(
            f"ZMP, rho diff = {diff_rho(mol, dm_zmp, dm_tar, grids):.3e} "
            f"I value = {diff_I_value(mol, dm_zmp, dm_tar, grids):.3e} "
            f"dipole diff = {np.linalg.norm(zmp_dipole - tar_dipole):.3e} ",
            flush=True,
        )
        print(
            f"DFT, rho diff = {diff_rho(mol, dm_dft, dm_tar, grids):.3e} "
            f"I value = {diff_I_value(mol, dm_dft, dm_tar, grids):.3e} "
            f"dipole diff = {np.linalg.norm(dft_dipole - tar_dipole):.3e} ",
            flush=True,
        )

        if (
            diff_rho_best is None
            or diff_rho(mol, dm_zmp, dm_tar, grids) < diff_rho_best
        ):
            diff_rho_best = diff_rho(mol, dm_zmp, dm_tar, grids)
        elif diff_rho(mol, dm_zmp, dm_tar, grids) > diff_rho_best * 10:
            print("Warning: rho difference increased significantly, may be diverged.")
            converged = False
            break

        # hatree part from dm1_zmp
        rho_zmp = [
            pyscf.dft.numint.eval_rho(mol, ao, dm_zmp[0], xctype="LDA"),
            pyscf.dft.numint.eval_rho(mol, ao, dm_zmp[1], xctype="LDA"),
        ]
        hatree_zmp_grids = (
            0.5
            * np.einsum("pij,ij->p", int1e_grids, dm_zmp[0] + dm_zmp[1])
            * (rho_zmp[0] + rho_zmp[1])
        )
        print(
            f"Difference in Hartree grids between CCSD: DFT {np.sum(np.abs(hatree_cc_grids - hatree_dft_grids) * grids.weights) * AU2KCALMOL}, ZMP {np.sum(np.abs(hatree_cc_grids - hatree_zmp_grids) * grids.weights) * AU2KCALMOL}"
        )

        nuc_zmp_grids = (rho_zmp[0] + rho_zmp[1]) * nuc_grids
        print(
            f"Difference in nuclear grids between CCSD: DFT {np.sum(np.abs(nuc_tar_grids - nuc_dft_grids) * grids.weights) * AU2KCALMOL}, ZMP {np.sum(np.abs(nuc_tar_grids - nuc_zmp_grids) * grids.weights) * AU2KCALMOL}"
        )
        dm_zmp_old = [dm.copy() for dm in dm_zmp]

    if converged:
        dm_zmp = mzmp.make_rdm1()
        return mzmp, dm_zmp
    print("Warning: rho difference increased significantly, may be diverged.")
    return mzmp, dm_zmp_old
