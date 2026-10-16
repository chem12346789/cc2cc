import ast
import contextlib
import io
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
import pyscf

import gen_data


class TestB3lypGradient(TestCase):
    def test_finite_difference_matches_analytical(self):
        original_threads = pyscf.lib.num_threads()
        pyscf.lib.num_threads(1)
        self.addCleanup(pyscf.lib.num_threads, original_threads)
        for spin, atoms in (
            (0, "H 0 0 0; H 0.2 0 1.4"),
            (1, "He 0 0 0; H 0.3 0 1.5"),
        ):
            with self.subTest(spin=spin):
                mol = pyscf.M(
                    atom=atoms, basis="sto-3g", unit="Bohr", spin=spin, verbose=0
                )
                coords = mol.atom_coords().copy()
                mf = gen_data._converged_b3lyp(mol)
                analytical = mf.nuc_grad_method().kernel()
                numerical = gen_data._b3lyp_finite_difference_gradient(mol)
                self.assertEqual(numerical.shape, (mol.natm, 3))
                self.assertEqual(numerical.dtype, np.float64)
                np.testing.assert_allclose(analytical, numerical, rtol=0, atol=5e-5)
                np.testing.assert_array_equal(mol.atom_coords(), coords)

    def test_reports_maximum_absolute_component_difference(self):
        source = Path(gen_data.__file__)
        tree = ast.parse(source.read_text())
        block = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "mol.natm == 1"
        )
        analytical = np.array([[0.1, -0.2, 0.3], [-0.1, 0.2, -0.3]])
        numerical = analytical.copy()
        numerical[0, 1] -= 0.002
        mf = Mock()
        mf.nuc_grad_method.return_value.kernel.return_value = analytical
        namespace = dict(
            np=np, mol=Mock(), dm_dft=Mock(), nlcgrids=Mock(),
            args=Mock(grid_level=0),
            _converged_b3lyp=Mock(return_value=mf),
            post_dft_vv10_gradient=Mock(return_value=analytical),
            _post_vv10_finite_difference_gradient=Mock(return_value=analytical),
            _b3lyp_finite_difference_gradient=Mock(return_value=numerical),
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(ast.Module(body=block.orelse, type_ignores=[]),
                         str(source), "exec"), namespace)
        self.assertIn(
            "DFT (B3LYP) analytical/FD gradient max difference "
            "(Hartree/Bohr): 2.000000000000e-03", output.getvalue()
        )
        namespace["_b3lyp_finite_difference_gradient"].assert_called_once_with(
            namespace["mol"]
        )

    def test_post_vv10_reference_preserves_grid_level_and_step(self):
        mol = pyscf.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        with patch.object(gen_data, "_post_vv10_total_energy",
                          side_effect=lambda displaced, level:
                          np.sum(displaced.atom_coords() ** 2) + level) as energy:
            result = gen_data._post_vv10_finite_difference_gradient(mol, 2, 0.002)
        np.testing.assert_array_equal(result, np.zeros((1, 3)))
        self.assertEqual(energy.call_count, 6)
        self.assertTrue(all(call.args[1] == 2 for call in energy.call_args_list))
