"""
ZMP is a method to invert the density matrix from CCSD(T). See the original paper for details: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.50.2138 and https://arxiv.org/pdf/2603.22140v1.
It is based on the idea of minimizing the difference between the DFT and CC density matrices in a least-squares sense.
"""

import types

import numpy as np

import pyscf
from pyscf import lib
from pyscf.lib import logger


def zmp(mol, dm1_dft, dm1_cc):

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
            max_memory = ks_.max_memory - lib.current_memory()[0]
            n, exc, vxc = ni.nr_rks(mol, ks_.grids, ks_.xc, dm, max_memory=max_memory)
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

    ks = pyscf.scf.RKS(mol)
    ks.get_veff = types.MethodType(get_veff, ks)

    return


def uzmp(dm1_dft, dm1_cc):
    return
