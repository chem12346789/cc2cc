import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, sentinel

import numpy as np


class TestGenDataDensityComparison(unittest.TestCase):
    def test_cached_cc_density_comparisons(self):
        source = Path(__file__).resolve().parents[1] / "gen_data.py"
        tree = ast.parse(source.read_text())
        comparison = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and ast.unparse(node.test).startswith("args.if_continue")
        )
        start = next(
            index
            for index, node in enumerate(comparison.body)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "drho"
        )
        block = ast.Module(body=comparison.body[start:], type_ignores=[])
        diff_rho = Mock(side_effect=[0.01, 0.02, 0.03])
        namespace = dict(
            dm_cc=sentinel.dm_cc,
            diff_rho=diff_rho,
            mol=sentinel.mol,
            grids=sentinel.grids,
            dm_dft=sentinel.dm_dft,
            dm_scf_vv10=sentinel.dm_scf_vv10,
            name="test",
            e_post_vv10=1.0,
            e_dft=0.0,
            e_scf_vv10=0.9,
            AU2KCALMOL=1.0,
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(block, str(source), "exec"), namespace)
        self.assertEqual(
            diff_rho.call_args_list,
            [
                unittest.mock.call(
                    sentinel.mol, sentinel.dm_dft, sentinel.dm_scf_vv10, sentinel.grids
                ),
                unittest.mock.call(
                    sentinel.mol, sentinel.dm_dft, sentinel.dm_cc, sentinel.grids
                ),
                unittest.mock.call(
                    sentinel.mol, sentinel.dm_scf_vv10, sentinel.dm_cc, sentinel.grids
                ),
            ],
        )
        self.assertIn(
            "Integrated absolute density difference (electrons): 1.0000000000e-02",
            output.getvalue(),
        )
        self.assertIn(
            "(B3LYP/post-DFT VV10 vs CCSD, electrons): 2.0000000000e-02",
            output.getvalue(),
        )
        self.assertIn(
            "(SCF-VV10 vs CCSD, electrons): 3.0000000000e-02",
            output.getvalue(),
        )

    def test_missing_cached_fields_skip_comparison(self):
        source = Path(__file__).resolve().parents[1] / "gen_data.py"
        tree = ast.parse(source.read_text())
        cache_load = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and "np.load" in ast.unparse(node.items[0].context_expr)
        )
        loop = ast.For(
            target=ast.Name(id="_", ctx=ast.Store()),
            iter=ast.List(elts=[ast.Constant(value=0)], ctx=ast.Load()),
            body=[cache_load, ast.parse("completed.append(True)").body[0]],
            orelse=[],
        )
        block = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        weights = np.array([0.2, 0.3])
        fields = {
            "dm1_dft": sentinel.dm_dft, "e_dft": 1.0,
            "dm1_cc": sentinel.dm_cc, "weights": weights,
        }
        for missing in ((), ("dm1_dft",), ("e_dft",), ("dm1_cc",), ("weights",), tuple(fields)):
            with self.subTest(missing=missing):
                archive = MagicMock()
                archive.__enter__.return_value = {
                    key: value for key, value in fields.items() if key not in missing
                }
                mock_np = Mock()
                mock_np.load.return_value = archive
                grid = Mock(weights=weights.copy())
                namespace = dict(
                    np=mock_np, DATA_PATH=Path("."), name="test", completed=[], grids=grid
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(block, str(source), "exec"), namespace)
                self.assertEqual(namespace["completed"], [] if missing else [True])
                archive.__exit__.assert_called_once()
                if missing:
                    self.assertIn(f"missing cached fields: {', '.join(missing)}", output.getvalue())
                else:
                    self.assertEqual(namespace["dm_dft"], sentinel.dm_dft)
                    self.assertEqual(namespace["e_dft"], 1.0)
                    self.assertEqual(namespace["dm_cc"], sentinel.dm_cc)
                    self.assertIs(grid.weights, weights)

        archive.__enter__.return_value = fields
        grid.weights = np.ones(3)
        with self.assertRaisesRegex(ValueError, "Cached weights shape"):
            exec(compile(block, str(source), "exec"), namespace)
        self.assertEqual(grid.weights.shape, (3,))
