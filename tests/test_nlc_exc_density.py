import ast
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import Mock

import numpy as np
import pyscf
import pyscf.dft

from cc2cc.utils import AU2KCALMOL, Grid, eval_nlc_exc_density


class TestNlcExcDensity(unittest.TestCase):
    def test_post_dft_nlc_check_reports_and_rejects_mismatch(self):
        source = Path(__file__).resolve().parents[1] / "gen_data.py"
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
            and ast.unparse(node.targets[0]) == "enlc_post"
        )
        block = ast.Module(body=comparison.body[start:start + 5], type_ignores=[])
        for reference, fails in ((0.3, False), (0.4, True), (np.nan, True)):
            with self.subTest(reference=reference):
                mock_pyscf = Mock()
                mock_pyscf.dft.numint.NumInt.return_value.nr_nlc_vxc.return_value = (
                    2, reference, None
                )
                namespace = dict(
                    np=np, pyscf=mock_pyscf, mol=object(), name="test",
                    nlcgrids=Mock(weights=np.array([1.0, 2.0])),
                    dm_dft_tot=object(), exc_post_grid=np.array([0.1, 0.1]),
                    e_dft=-1.0, AU2KCALMOL=AU2KCALMOL,
                )
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    if fails:
                        with self.assertRaisesRegex(RuntimeError, "disagrees"):
                            exec(compile(block, str(source), "exec"), namespace)
                    else:
                        exec(compile(block, str(source), "exec"), namespace)
                        self.assertAlmostEqual(namespace["e_post_vv10"], -0.7)
                difference = (namespace["enlc_post"] - reference) * AU2KCALMOL
                self.assertIn(
                    f"NLC energy check (kcal/mol): difference = {difference:.12e}",
                    output.getvalue(),
                )

    def test_post_dft_nlc_reuses_existing_grid(self):
        source = Path(__file__).resolve().parents[1] / "gen_data.py"
        tree = ast.parse(source.read_text())
        assignment = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "nlcgrids"
        )
        grids = object()
        namespace = {"grids": grids}
        block = ast.Module(body=[assignment], type_ignores=[])
        exec(compile(block, str(source), "exec"), namespace)
        self.assertIs(namespace["nlcgrids"], grids)

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
