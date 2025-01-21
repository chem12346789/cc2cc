"""@package docstring
Documentation for this module.

More details.
"""

import numpy as np

import opt_einsum as oe


class GenTau:
    """
    Documentation for a class.

    More details.
    """

    def __init__(self, mol, grids):
        """
        Documentation for a method.

        More details.
        """
        ao_value = pyscf.dft.numint.eval_ao(mol, grids.coords, deriv=2)
        ao_0 = ao_value[0]
        ao_1 = ao_value[1:4]
        ao_2_diag = ao_value[4] + ao_value[7] + ao_value[9]
        self.norb = mol.nao

        self.oe_taup_rho = oe.contract_expression(
            "pm,m,n,kpn->pk",
            ao_0,
            (norb,),
            (norb,),
            ao_1,
            constants=[0, 3],
            optimize="optimal",
        )

        self.oe_tau_rho = oe.contract_expression(
            "pm,m,n,pn->p",
            ao_0,
            (norb,),
            (norb,),
            ao_2_diag,
            constants=[0, 3],
            optimize="optimal",
        )

    def gen_taup_rho(
        self,
        dm1_r,
        eigs_v_dm1,
        eigs_e_dm1,
        oe_taup_rho,
        backend="numpy",
    ):
        """
        Documentation for a function.

        More details.
        """
        taup = np.zeros(len(dm1_r))
        norb = np.shape(eigs_v_dm1)[1]

        for i in range(norb):
            for j in range(i + 1):
                if i != j:
                    part = oe_taup_rho(
                        eigs_v_dm1[:, i], eigs_v_dm1[:, j], backend=backend
                    )
                    part -= oe_taup_rho(
                        eigs_v_dm1[:, j], eigs_v_dm1[:, i], backend=backend
                    )
                    part1 = np.sum(part**2, axis=1)
                    taup += part1 * eigs_e_dm1[i] * eigs_e_dm1[j]
        taup_rho = taup / dm1_r * 0.5
        return taup_rho

    def gen_tau_rho(
        self,
        dm1_r,
        eigs_v_dm1,
        eigs_e_dm1,
        oe_tau_rho,
        backend="numpy",
    ):
        """
        Documentation for a function.

        More details.
        """
        tau = np.zeros(len(dm1_r))
        norb = np.shape(eigs_v_dm1)[1]

        for i in range(norb):
            part = oe_tau_rho(eigs_v_dm1[:, i], eigs_v_dm1[:, i], backend=backend)
            tau += part * eigs_e_dm1[i]
        taup_rho = -tau / 2
        return taup_rho
