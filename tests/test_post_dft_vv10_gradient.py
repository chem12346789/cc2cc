import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyscf

import gen_data
from cc2cc.utils import Grid
from cc2cc.utils.nlc import post_dft_vv10_gradient


class TestPostDftVv10Gradient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = pyscf.lib.num_threads()
        pyscf.lib.num_threads(1)

    @classmethod
    def tearDownClass(cls):
        pyscf.lib.num_threads(cls.original_threads)

    def test_gradient_without_grid_response(self):
        cases = (
            (0, "O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2"),
            (1, "He 0 0 0; H 0.3 0 1.5"),
        )
        for spin, atoms in cases:
            with self.subTest(spin=spin):
                mol = pyscf.M(
                    atom=atoms, basis="sto-3g", unit="Bohr", spin=spin, verbose=0
                )
                mf = gen_data._converged_b3lyp(mol)
                coords = mol.atom_coords().copy()
                dm = mf.make_rdm1().copy()
                scf_grad = mf.nuc_grad_method()
                with patch.object(mf, "nuc_grad_method", return_value=scf_grad):
                    analytical = post_dft_vv10_gradient(mf, Grid(mol, 0, 7))
                self.assertEqual(analytical.shape, (mol.natm, 3))
                self.assertEqual(analytical.dtype, np.float64)
                self.assertFalse(scf_grad.grid_response)
                np.testing.assert_allclose(analytical.sum(axis=0), 0, atol=5e-5)
                np.testing.assert_array_equal(mol.atom_coords(), coords)
                np.testing.assert_array_equal(mf.make_rdm1(), dm)

    def test_occupied_overlap_response_is_omitted(self):
        mol = pyscf.M(
            atom="O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2",
            basis="sto-3g", unit="Bohr", verbose=0,
        )
        mf = gen_data._converged_b3lyp(mol)
        grids = Grid(mol, 0, 7)
        reference = post_dft_vv10_gradient(mf, grids)
        hessian = mf.Hessian()
        h1 = hessian.make_h1(mf.mo_coeff, mf.mo_occ)
        mo1, energy1 = hessian.solve_mo1(
            mf.mo_energy, mf.mo_coeff, mf.mo_occ, h1
        )
        occupied_coeff = mf.mo_coeff[:, mf.mo_occ > 0]
        occupied_shift = np.einsum(
            "pi,xij->xpj",
            occupied_coeff,
            np.broadcast_to(
                np.eye(occupied_coeff.shape[1]),
                (3, occupied_coeff.shape[1], occupied_coeff.shape[1]),
            ),
        )
        shifted_mo1 = [
            response + occupied_shift
            for response in mo1
        ]
        with patch.object(mf, "Hessian", return_value=hessian):
            with patch.object(
                hessian, "solve_mo1", return_value=(shifted_mo1, energy1)
            ):
                shifted = post_dft_vv10_gradient(mf, grids)
        np.testing.assert_allclose(shifted, reference, rtol=0, atol=1e-10)

    def test_orbital_response_is_required(self):
        mol = pyscf.M(
            atom="O 0 0 0; H 1.4 0 1.1; H -1.3 0 1.2",
            basis="sto-3g", unit="Bohr", verbose=0,
        )
        mf = gen_data._converged_b3lyp(mol)
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
        mol = pyscf.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        mf = gen_data._converged_b3lyp(mol)
        grids = Grid(mol, 0, 7)
        mf.converged = False
        with self.assertRaisesRegex(RuntimeError, "converged B3LYP"):
            post_dft_vv10_gradient(mf, grids)
        mf.converged = True
        mf.nlc = "vv10"
        with self.assertRaisesRegex(ValueError, "without NLC"):
            post_dft_vv10_gradient(mf, grids)

    def test_optional_finite_difference_check_threshold(self):
        source = Path(gen_data.__file__)
        tree = ast.parse(source.read_text())
        comparison = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and ast.unparse(node.test).startswith("args.if_continue")
        )
        start = next(
            i for i, node in enumerate(comparison.body)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "mf_dft"
        )
        stop = next(
            i for i, node in enumerate(comparison.body)
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == "addon_path"
        )
        block = ast.Module(body=comparison.body[start:stop], type_ignores=[])
        for enabled, error in ((False, 1), (True, 4e-5), (True, 6e-5)):
            with self.subTest(enabled=enabled, error=error):
                fd_calls = []
                numerical = np.full((2, 3), error)

                def finite_difference(*args):
                    fd_calls.append(args)
                    return numerical

                namespace = dict(
                    np=np, mol=object(), dm_dft=object(), nlcgrids=object(),
                    name="test", args=SimpleNamespace(
                        check_post_vv10_gradient=enabled, grid_level=0,
                    ),
                    _converged_b3lyp=lambda *args, **kwargs: object(),
                    post_dft_vv10_gradient=lambda *args: np.zeros((2, 3)),
                    _post_vv10_finite_difference_gradient=finite_difference,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    if enabled and error > 5e-5:
                        with self.assertRaisesRegex(RuntimeError, "check failed"):
                            exec(compile(block, str(source), "exec"), namespace)
                    else:
                        exec(compile(block, str(source), "exec"), namespace)
                self.assertEqual(len(fd_calls), int(enabled))
                if not enabled:
                    self.assertIsNone(namespace["grad_post_vv10_fd"])


if __name__ == "__main__":
    unittest.main()
