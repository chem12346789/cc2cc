import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyscf

from cc2cc.utils.reference.reference_b3lyp import converged_b3lyp
from cc2cc.utils.Grids import Grid
import cc2cc.utils.reference.reference_vv10 as vv10
from cc2cc.utils.reference.reference_vv10 import post_dft_vv10_gradient


class TestPostDftVv10Gradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = pyscf.lib.num_threads()
        pyscf.lib.num_threads(1)

    @classmethod
    def tearDownClass(cls):
        pyscf.lib.num_threads(cls.original_threads)

    def test_gradient_with_grid_response(self):
        cases = (
            (0, "O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2"),
            (1, "He 0 0 0; H 0.3 0 1.5"),
        )
        for spin, atoms in cases:
            with self.subTest(spin=spin):
                mol = pyscf.M(
                    atom=atoms, basis="sto-3g", unit="Bohr", spin=spin, verbose=0
                )
                mf = converged_b3lyp(mol)
                coords = mol.atom_coords().copy()
                dm = mf.make_rdm1().copy()
                scf_grad = mf.nuc_grad_method()
                with patch.object(mf, "nuc_grad_method", return_value=scf_grad):
                    analytical = post_dft_vv10_gradient(mf, Grid(mol, 0, 7))
                self.assertEqual(analytical.shape, (mol.natm, 3))
                self.assertEqual(analytical.dtype, np.float64)
                self.assertTrue(scf_grad.grid_response)
                np.testing.assert_allclose(analytical.sum(axis=0), 0, atol=5e-5)
                np.testing.assert_array_equal(mol.atom_coords(), coords)
                np.testing.assert_array_equal(mf.make_rdm1(), dm)

    def test_occupied_overlap_response_is_included(self):
        mol = pyscf.M(
            atom="O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2",
            basis="sto-3g", unit="Bohr", verbose=0,
        )
        mf = converged_b3lyp(mol)
        grids = Grid(mol, 0, 7)
        reference = post_dft_vv10_gradient(mf, grids)
        hessian = mf.Hessian()
        solve_mo1 = hessian.solve_mo1
        occupied_coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        occupied_shift = np.einsum(
            "pi,xij->xpj",
            occupied_coeff,
            np.broadcast_to(
                np.eye(occupied_coeff.shape[1]),
                (3, occupied_coeff.shape[1], occupied_coeff.shape[1]),
            ),
        )

        def shifted_response(*args, **kwargs):
            mo1, energy1 = solve_mo1(*args, **kwargs)
            return [response + occupied_shift for response in mo1], energy1

        with patch.object(mf, "Hessian", return_value=hessian):
            with patch.object(
                hessian, "solve_mo1", side_effect=shifted_response,
            ):
                shifted = post_dft_vv10_gradient(mf, grids)
        dm = mf.make_rdm1()
        _, _, vnlc = mf._numint.nr_nlc_vxc(mol, grids, "vv10", dm)
        # Scaling an occupied orbital changes both factors of its density matrix.
        expected_change = 2 * np.trace(vnlc @ dm)
        self.assertGreater(abs(expected_change), 1e-5)
        np.testing.assert_allclose(
            shifted - reference, expected_change, rtol=0, atol=1e-10,
        )

    def test_orbital_response_is_required(self):
        mol = pyscf.M(
            atom="O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2",
            basis="sto-3g", unit="Bohr", verbose=0,
        )
        mf = converged_b3lyp(mol)
        grids = Grid(mol, 0, 7)
        analytical = post_dft_vv10_gradient(mf, grids)
        hessian = mf.Hessian()
        h1 = hessian.make_h1(mf.mo_coeff, mf.mo_occ)
        mo1, energy1 = hessian.solve_mo1(
            mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1
        )
        with patch.object(mf, "Hessian", return_value=hessian):
            with patch.object(
                hessian, "solve_mo1",
                return_value=([np.zeros_like(x) for x in mo1], energy1),
            ):
                unrelaxed = post_dft_vv10_gradient(mf, grids)
        self.assertGreater(np.max(np.abs(analytical - unrelaxed)), 1e-5)

    def test_rejects_nonconverged_or_self_consistent_nlc(self):
        mol = pyscf.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = converged_b3lyp(mol)
        grids = Grid(mol, 0, 7)
        dm = mf.make_rdm1()
        mf.converged = False
        with (
            patch.object(vv10, "converged_b3lyp", return_value=mf),
            patch.object(vv10, "post_dft_vv10_gradient") as gradient,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(RuntimeError, "reference did not converge"):
                vv10.get_post_vv10_data(
                    mol, grids, dm, mf.e_tot, dm, None, evaluate=True,
                )
            gradient.assert_not_called()
        mf.converged = True
        mf.nlc = "vv10"
        with self.assertRaisesRegex(ValueError, "without NLC"):
            post_dft_vv10_gradient(mf, grids)

    def test_optional_finite_difference_check_reports_difference(self):
        mol = SimpleNamespace(natm=2)
        grids = SimpleNamespace(weights=np.ones(1, dtype=np.float64))
        density = np.eye(2, dtype=np.float64)
        mf = Mock(converged=True)
        mf.make_rdm1.return_value = density
        mf.nuc_grad_method.return_value.kernel.return_value = np.zeros((2, 3))
        for enabled, error in ((False, 1), (True, 4e-5), (True, 6e-5)):
            with self.subTest(enabled=enabled, error=error):
                numerical = np.full((2, 3), error, dtype=np.float64)
                output = io.StringIO()
                with (
                    patch.object(vv10, "eval_nlc_exc_density", return_value=np.zeros(1)),
                    patch.object(
                        pyscf.dft.numint.NumInt, "nr_nlc_vxc",
                        return_value=(2, 0.0, None),
                    ),
                    patch.object(vv10, "converged_b3lyp", return_value=mf),
                    patch.object(
                        vv10, "post_dft_vv10_gradient", return_value=np.zeros((2, 3)),
                    ),
                    patch.object(
                        vv10, "post_vv10_finite_difference_gradient", return_value=numerical,
                    ) as finite_difference,
                    patch.object(
                        vv10, "b3lyp_finite_difference_gradient",
                        return_value=np.zeros((2, 3)),
                    ) as b3lyp_difference,
                    patch.object(vv10, "diff_rho", return_value=0.0),
                    contextlib.redirect_stdout(output),
                ):
                    data = vv10.get_post_vv10_data(
                        mol, grids, density, -1.0, density, SimpleNamespace(grid_level=0),
                        check_post_vv10_gradient=enabled,
                    )
                if enabled:
                    finite_difference.assert_called_once_with(
                        mol, 0, unrestricted=False, dm0=density,
                    )
                    b3lyp_difference.assert_called_once_with(
                        mol, unrestricted=False, dm0=density,
                    )
                    np.testing.assert_array_equal(data["grad_post_vv10_fd"], numerical)
                    self.assertIn(
                        "Post-DFT VV10 analytical/FD gradient max difference "
                        f"(Hartree/Bohr): {error:.12e}",
                        output.getvalue(),
                    )
                else:
                    finite_difference.assert_not_called()
                    b3lyp_difference.assert_not_called()
                    self.assertNotIn("grad_post_vv10_fd", data)
                    self.assertNotIn("analytical/FD", output.getvalue())


if __name__ == "__main__":
    unittest.main()
