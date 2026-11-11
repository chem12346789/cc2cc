import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
from pyscf.lib import param

from scripts.force_validation import (
    STEPS_BOHR, build_dataset, gradient_report,
)
from cc2cc.utils.parser import str2bool


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "cc2cc/utils/mol_dataset/force-def2.json"


class TestForceValidation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET.read_text())

    def test_generated_dataset_and_steps(self):
        source = json.loads(
            (ROOT / "cc2cc/utils/mol_dataset/gmtkn-def2.json").read_text()
        )
        self.assertEqual(self.dataset, build_dataset(source))
        self.assertEqual(len(self.dataset["molecule"]), 188)
        self.assertEqual(len(set(self.dataset["molecule"])), 188)
        for name in self.dataset["molecule_Force_legacy"]:
            self.assertEqual(self.dataset[name], source[name])
        for label, case in self.dataset["force_validation"]["cases"].items():
            center = case["center"]
            self.assertEqual(self.dataset["spin"][center], 1 if label == "oh" else 0)
            coords = np.array([a[1:] for a in self.dataset[center]]) / param.BOHR
            for step in STEPS_BOHR:
                pairs = case["pairs"][f"{step:.3f}"]
                self.assertEqual(len(pairs), coords.size)
                self.assertEqual(len({(p["atom"], p["axis"]) for p in pairs}), coords.size)
                for pair in pairs:
                    for sign, key in ((-1, "minus"), (1, "plus")):
                        name = pair[key]
                        displaced = np.array([a[1:] for a in self.dataset[name]]) / param.BOHR
                        expected = np.zeros_like(coords)
                        expected[pair["atom"], pair["axis"]] = sign * step
                        np.testing.assert_allclose(displaced - coords, expected, atol=1e-14)
                        self.assertEqual(self.dataset["spin"][name], self.dataset["spin"][center])
                        self.assertEqual(self.dataset["charge"][name], self.dataset["charge"][center])
        water = self.dataset[self.dataset["force_validation"]["cases"]["h2o"]["center"]]
        bonds = [np.linalg.norm(np.array(h[1:]) - water[0][1:]) for h in water[1:]]
        self.assertGreater(abs(bonds[0] - bonds[1]), 0.01)

    def synthetic_records(self, quartic=False):
        records = {}
        for name in self.dataset["molecule"]:
            coords = np.array([a[1:] for a in self.dataset[name]]) / param.BOHR
            relative = coords - coords.mean(axis=0)
            if quartic:
                energy = np.sum(relative ** 4)
                gradient = 4 * (relative ** 3 - (relative ** 3).mean(axis=0))
            else:
                energy = np.sum(relative ** 2)
                gradient = 2 * relative
            records[name] = dict(energy=energy, gradient=gradient, settings={"test": True})
        return records

    def test_full_cartesian_report_and_invariance(self):
        records = self.synthetic_records()
        centers = self.dataset["molecule_Force_centers"]
        on = {name: records[name] for name in centers}
        off = {name: dict(records[name], gradient=records[name]["gradient"] + 0.01)
               for name in centers}
        reports = gradient_report(self.dataset, records, on, off)
        self.assertEqual(len(reports), 18)
        for row in reports:
            if row["grid_response"] == "on":
                self.assertLess(row["max_error"], 1e-10)
                np.testing.assert_allclose(row["net_gradient"], 0, atol=1e-14)
                np.testing.assert_allclose(row["torque"], 0, atol=1e-14)
            else:
                self.assertAlmostEqual(row["max_error"], 0.01, places=9)
                self.assertAlmostEqual(row["rms_error"], 0.01, places=9)

    def test_second_order_convergence(self):
        records = self.synthetic_records(quartic=True)
        for row in gradient_report(self.dataset, records, records, records):
            if row["observed_order"] is not None:
                self.assertAlmostEqual(row["observed_order"], 2, places=3)

    def test_on_only_report(self):
        records = self.synthetic_records()
        reports = gradient_report(self.dataset, records, records)
        self.assertEqual(len(reports), 9)
        self.assertTrue(all(row["grid_response"] == "on" for row in reports))
        self.assertTrue(all(row["max_error"] < 1e-10 for row in reports))

    def test_missing_results_and_mismatched_settings_are_rejected(self):
        records = self.synthetic_records()
        off = dict(records)
        center = self.dataset["molecule_Force_centers"][0]
        off[center] = dict(off[center], settings={"test": False})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            gradient_report(self.dataset, records, records, off)
        del records[center]
        with self.assertRaises(KeyError):
            gradient_report(self.dataset, records, records, records)

    def test_runner_convergence_gate(self):
        for spin in ("rks", "uks"):
            path = ROOT / "cc2cc/test_model.py"
            tree = ast.parse(path.read_text())
            block = next(node for node in ast.walk(tree)
                         if isinstance(node, ast.If)
                         and ast.unparse(node.test) == "mdft.converged is False")
            code = compile(ast.Module(body=[block], type_ignores=[]), str(path), "exec")
            namespace = dict(mdft=SimpleNamespace(converged=False), name="test",
                             method=spin.upper(),
                             args=SimpleNamespace(require_convergence=True))
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                exec(code, namespace)
            namespace["args"].require_convergence = False
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(code, namespace)
            self.assertIn("unconvergence output", output.getvalue())

    def test_scf_control_defaults_and_overrides(self):
        tree = ast.parse((ROOT / "test.py").read_text())
        options = {"--conv_tol", "--conv_tol_grad", "--require_convergence"}
        parser = argparse.ArgumentParser()
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in options):
                exec(compile(ast.Expression(node), "test.py", "eval"),
                     dict(parser=parser, str2bool=str2bool))
        defaults = parser.parse_args([])
        self.assertEqual((defaults.conv_tol, defaults.conv_tol_grad), (1e-7, 1e-3))
        self.assertFalse(defaults.require_convergence)
        tight = parser.parse_args([
            "--conv_tol", "1e-11", "--conv_tol_grad", "1e-7",
            "--require_convergence", "1",
        ])
        self.assertEqual((tight.conv_tol, tight.conv_tol_grad), (1e-11, 1e-7))
        self.assertTrue(tight.require_convergence)
