import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyscf
import pyscf.dft

from cc2cc.utils.Grids import Grid
from cc2cc.utils.helper.units import AU2KCALMOL
import cc2cc.utils.reference.reference_vv10 as vv10
from cc2cc.utils.reference.reference_vv10 import eval_nlc_exc_density


class TestNlcExcDensity(unittest.TestCase):
    def test_post_dft_grid_saved_to_base_preserving_fields(self):
        source = Path(__file__).resolve().parents[1] / "cc2cc" / "generate_cc_data.py"
        tree = ast.parse(source.read_text())
        generation = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "cc"
        )
        start = next(
            i for i, node in enumerate(generation.body)
            if isinstance(node, ast.Expr)
            and ast.unparse(node.value).startswith("data_dict.update(get_vv10_data(")
        )
        stop = next(
            i + 1 for i, node in enumerate(generation.body[start:], start)
            if isinstance(node, ast.Expr)
            and ast.unparse(node.value).startswith("np.savez(")
        )
        # Exercise the save path without running coupled-cluster data generation.
        block = ast.Module(body=generation.body[start:stop], type_ignores=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_AHB21-1A_def2-QZVPPD.npz"
            for check_fd in (False, True):
                with self.subTest(check_fd=check_fd):
                    exc_post_vv10_grid = np.array([0.1, 0.2], dtype=np.float64)
                    vv10_data = dict(
                        exc_post_vv10_grid=exc_post_vv10_grid, e_post_vv10=-0.3,
                        grad_post_vv10=np.zeros((2, 3), dtype=np.float64),
                    )
                    if check_fd:
                        vv10_data["grad_post_vv10_fd"] = np.full((2, 3), 1e-7)
                    zmp_data = {
                        f"{key}_zmp": value for key, value in vv10_data.items()
                        if key != "grad_post_vv10_fd"
                    }
                    namespace = dict(
                        np=np, DATA_PATH=Path(directory),
                        name="AHB21-1A_def2-QZVPPD",
                        data_dict={"e_dft_d3bj": -1.5},
                        get_vv10_data=Mock(side_effect=[vv10_data, zmp_data]),
                        get_zmp_constraint_data=Mock(return_value={}),
                        mol=object(), grids=object(), dm1_dft=object(),
                        e_dft=-1.0, dm1_cc=object(), args=object(),
                        dm1_zmp=object(), e_zmp=-0.9, mdft=object(),
                    )
                    exec(compile(block, str(source), "exec"), namespace)
                    with np.load(path) as data:
                        self.assertEqual(data["e_dft_d3bj"], -1.5)
                        self.assertEqual(data["exc_post_vv10_grid"].dtype, np.float64)
                        self.assertEqual("grad_post_vv10_fd" in data, check_fd)
                        for key, value in {**vv10_data, **zmp_data}.items():
                            np.testing.assert_array_equal(data[key], value)

    def test_post_dft_nlc_check_reports_and_rejects_mismatch(self):
        mol = Mock(natm=1)
        grids = Mock(weights=np.array([1.0, 2.0]))
        density = np.eye(1, dtype=np.float64)
        exc_density = np.array([0.1, 0.1], dtype=np.float64)
        for reference, fails in ((0.3, False), (0.4, True), (np.nan, True)):
            with self.subTest(reference=reference):
                output = io.StringIO()
                with (
                    patch.object(vv10, "eval_nlc_exc_density", return_value=exc_density),
                    patch.object(
                        pyscf.dft.numint.NumInt, "nr_nlc_vxc",
                        return_value=(2, reference, None),
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "disagrees"):
                            vv10.get_post_vv10_data(
                                mol, grids, density, -1.0, density, None, evaluate=True,
                            )
                    else:
                        data = vv10.get_post_vv10_data(
                            mol, grids, density, -1.0, density, None, evaluate=True,
                        )
                        self.assertAlmostEqual(data["e_post_vv10"], -0.7)
                difference = (np.dot(exc_density, grids.weights) - reference) * AU2KCALMOL
                self.assertIn(
                    f"NLC energy check (kcal/mol): difference = {difference:.12e}",
                    output.getvalue(),
                )

    def test_post_dft_nlc_reuses_existing_grid(self):
        mol = Mock(natm=1)
        grids = Mock(weights=np.array([1.0, 2.0]))
        density = np.eye(1, dtype=np.float64)
        with (
            patch.object(
                vv10, "eval_nlc_exc_density", return_value=np.array([0.1, 0.1]),
            ) as evaluate,
            patch.object(
                pyscf.dft.numint.NumInt, "nr_nlc_vxc", return_value=(2, 0.3, None),
            ) as integrate,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            vv10.get_post_vv10_data(
                mol, grids, density, -1.0, density, None, evaluate=True,
            )
        evaluate.assert_called_once_with(mol, grids, density, "vv10")
        integrate.assert_called_once_with(mol, grids, "vv10", density)

    def test_grid_energy_density_integrates_to_pyscf_nlc_energy(self):
        mol = pyscf.M(
            atom="H 0 0 0; H 0 0 0.74",
            basis="sto-3g",
            verbose=0,
        )
        grids = Grid(mol, 0, 7)
        dm = pyscf.dft.RKS(mol, xc="b3lyp").get_init_guess()

        exc_density = eval_nlc_exc_density(mol, grids, dm)
        _, excsum, _ = pyscf.dft.numint.NumInt().nr_nlc_vxc(
            mol, grids, "vv10", dm
        )

        self.assertEqual(exc_density.shape, grids.weights.shape)
        self.assertTrue(np.all(np.isfinite(exc_density)))
        self.assertAlmostEqual(
            float(np.dot(exc_density, grids.weights)), excsum, places=10
        )


if __name__ == "__main__":
    unittest.main()
