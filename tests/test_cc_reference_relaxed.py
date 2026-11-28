from unittest import TestCase

import numpy as np
from pyscf import dft, gto, scf

from cc2cc.utils.reference.reference_cc import (
    ccsd_t_target_density,
    ccsd_t_target_gradient,
    get_cc_energy,
    make_ccsd,
)


class TestRelaxedCCReference(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = []
        # Captured from the original pipeline at these geometries/settings.
        gradients = {
            False: [
                [0, 0, 0.01373262168533218],
                [0, 0.0031121889673704217, -0.006866310842664758],
                [0, -0.003112188967367757, -0.0068663108426627595],
            ],
            True: [
                [0, 0, 0.01247518732513786],
                [0, 0, -0.01247518732513786],
            ],
        }
        for unrestricted in (False, True):
            atom = (
                "O 0 0 0; H 0 0 0.97"
                if unrestricted else
                "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
            )
            mol = gto.M(
                atom=atom, basis="cc-pvdz", spin=int(unrestricted), verbose=0
            )
            mf = cls.scf_reference(mol, unrestricted)
            grids = dft.gen_grid.Grids(mol)
            grids.level = 3
            grids.build()
            mycc = make_ccsd(mf, unrestricted=unrestricted)
            mycc.kernel()
            eris = mycc.ao2mo()
            l1, l2, _, d1, dm_mo, dm_ao, d2, dm2 = ccsd_t_target_density(
                mycc, eris, unrestricted=unrestricted,
                reference=True, with_rdm2=True,
            )
            data = {
                "e_cc": mycc.e_tot + mycc.ccsd_t(eris=eris),
                "dm1_cc": dm_ao,
                "grad_cc": ccsd_t_target_gradient(
                    mycc, eris, l1, l2, d1, d2,
                    unrestricted=unrestricted,
                ),
            }
            data.update(get_cc_energy(
                mol, grids, mf, dm_ao, dm_mo, dm2,
                unrestricted=unrestricted,
            ))
            cls.cases.append((
                unrestricted, mol, mf, grids, data, dm_mo, dm2,
                np.asarray(gradients[unrestricted]),
            ))

    @staticmethod
    def scf_reference(mol, unrestricted, hcore=None):
        mf = scf.UHF(mol) if unrestricted else scf.RHF(mol)
        if hcore is not None:
            mf.get_hcore = lambda *args: hcore
        mf.set(conv_tol=1e-14, conv_tol_grad=1e-9, max_cycle=200).run()
        if not mf.converged:
            raise RuntimeError("Finite-field reference SCF did not converge.")
        return mf

    def test_energy_reconstruction(self):
        for u, mol, mf, _, data, _, dm2, _ in self.cases:
            with self.subTest(unrestricted=u):
                dm1 = data["dm1_cc"]
                if u:
                    self.assertIsInstance(dm2, tuple)
                    self.assertEqual(len(dm2), 3)
                    dm1 = dm1.sum(axis=0)
                    dm2 = dm2[0] + dm2[1] + dm2[1].transpose(2, 3, 0, 1) + dm2[2]
                energy = (
                    np.einsum("pq,qp", mf.get_hcore(), dm1)
                    + 0.5 * np.einsum("pqrs,pqrs", mol.intor("int2e"), dm2)
                    + mol.energy_nuc()
                )
                error = abs(energy - data["e_cc"])
                print(f"RDM energy error ({u=}): {error:.3e}")
                self.assertLess(error, 1e-10)

    def test_finite_field_dipole(self):
        step = 1e-4
        for u, mol, mf, _, data, _, _, _ in self.cases:
            with self.subTest(unrestricted=u):
                dm1 = data["dm1_cc"]
                total = dm1.sum(axis=0) if u else dm1
                dipole = scf.hf.dip_moment(mol, total, unit="AU", verbose=0)
                position = mol.intor("int1e_r", comp=3)
                nuclear = mol.atom_charges() @ mol.atom_coords()
                numerical = np.zeros(3)
                for axis in range(3):
                    energies = []
                    for field in (-step, step):
                        perturbed = self.scf_reference(
                            mol, u, mf.get_hcore() + field * position[axis]
                        )
                        mycc = make_ccsd(perturbed, unrestricted=u)
                        mycc.kernel()
                        self.assertTrue(mycc.converged)
                        energies.append(
                            mycc.e_tot + mycc.ccsd_t() - field * nuclear[axis]
                        )
                    numerical[axis] = -(energies[1] - energies[0]) / (2 * step)
                error = np.max(np.abs(dipole - numerical))
                print(f"Dipole error ({u=}): {error:.3e}")
                self.assertLess(error, 1e-6)

    def test_trace_and_symmetry(self):
        for u, mol, mf, _, data, dm_mo, dm2, _ in self.cases:
            with self.subTest(unrestricted=u):
                overlap = mf.get_ovlp()
                dm1 = data["dm1_cc"]
                np.testing.assert_allclose(
                    dm1, dm1.swapaxes(-1, -2), rtol=0, atol=1e-10
                )
                np.testing.assert_allclose(
                    dm_mo, dm_mo.swapaxes(-1, -2), rtol=0, atol=1e-10
                )
                coeff = np.asarray(mf.mo_coeff)
                np.testing.assert_allclose(
                    dm1, coeff @ dm_mo @ coeff.swapaxes(-1, -2),
                    rtol=0, atol=1e-10,
                )
                counts = mol.nelec if u else mol.nelectron
                np.testing.assert_allclose(
                    np.einsum("...pq,pq->...", dm1, overlap),
                    counts, rtol=0, atol=1e-10,
                )
                for pair in dm2 if u else (dm2,):
                    np.testing.assert_allclose(
                        pair, pair.transpose(1, 0, 3, 2), rtol=0, atol=1e-9
                    )
                for pair in (dm2[0], dm2[2]) if u else (dm2,):
                    np.testing.assert_allclose(
                        pair, pair.transpose(2, 3, 0, 1), rtol=0, atol=1e-9
                    )
                total = (
                    dm2[0] + dm2[1] + dm2[1].transpose(2, 3, 0, 1) + dm2[2]
                    if u else dm2
                )
                pair_trace = np.einsum("pqrs,pq,rs", total, overlap, overlap)
                error = abs(pair_trace - mol.nelectron * (mol.nelectron - 1))
                print(f"Pair-trace error ({u=}): {error:.3e}")
                # Deliberately retain the requested normalization assertion:
                # an effective CCSD(T) derivative density need not satisfy it.
                self.assertLess(error, 1e-9)

    def test_grid_energy(self):
        for u, mol, _, grids, data, _, _, _ in self.cases:
            with self.subTest(unrestricted=u):
                energy = data["tol_cc_grids"] @ grids.weights + mol.energy_nuc()
                error = abs(energy - data["e_cc"])
                print(f"Grid energy error ({u=}): {error:.3e}")
                self.assertLess(error, 1e-4)

    def test_gradient_regression(self):
        for u, _, _, _, data, _, _, original in self.cases:
            with self.subTest(unrestricted=u):
                error = np.max(np.abs(data["grad_cc"] - original))
                print(f"Gradient error ({u=}): {error:.3e}")
                self.assertLess(error, 1e-10)
