import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import mock_open, patch

import pandas as pd

from collect_info import AU2KCALMOL, Collect_info


class TestCollectInfoEnergyColumn(TestCase):
    def make_collector(self, **kwargs):
        with patch.object(Collect_info, "_get_done_cache_entry", return_value={"done": True}):
            return Collect_info(
                "model", 1, "basis", data_set="gmtkn-diet30-def2", verbose=0, **kwargs
            )

    def test_default_and_custom_cache_identity(self):
        default = self.make_collector()
        custom = self.make_collector(energy_column="post_scf_vv10_ene")
        self.assertEqual(default.energy_column, "scf_d3bj_ene")
        self.assertEqual(default.stamp, "basis_model_1_gmtkn-diet30-def2")
        self.assertEqual(custom.data_path, default.data_path)
        self.assertEqual(custom.stamp, default.stamp + "_post_scf_vv10_ene")

    def test_cli_option(self):
        path = Path(__file__).resolve().parents[1] / "collect_info.py"
        tree = ast.parse(path.read_text())
        option = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--energy_column"
        )
        parser = argparse.ArgumentParser()
        exec(compile(ast.Expression(option), str(path), "eval"), {"parser": parser})
        self.assertEqual(parser.parse_args([]).energy_column, "scf_d3bj_ene")
        self.assertEqual(
            parser.parse_args(["--energy_column", "post_scf_vv10_ene"]).energy_column,
            "post_scf_vv10_ene",
        )

    def test_default_d3bj_correction(self):
        collector = self.make_collector()
        collector.data = pd.DataFrame({"name": ["A_basis"], "scf_ene": [-1.0]})
        reference = pd.DataFrame({
            "name": ["A_def2-QZVP(D)"],
            "b3lyp_ene": [-2.0],
            "b3lyp-d3bj_ene": [-2.25],
        })
        with patch("collect_info.pd.read_csv", return_value=reference) as read_csv:
            with contextlib.redirect_stdout(io.StringIO()):
                collector.add_d3bj_correction()
                collector.add_d3bj_correction()
        self.assertEqual(collector.data["scf_d3bj_ene"].tolist(), [-1.25])
        read_csv.assert_called_once()

    def test_custom_column_is_not_filled_with_d3bj(self):
        collector = self.make_collector(energy_column="post_scf_vv10_ene")
        for values in (None, [-1.2], [float("nan")]):
            with self.subTest(values=values):
                collector.data = pd.DataFrame({"name": ["A_basis"], "scf_ene": [-1.0]})
                if values is not None:
                    collector.data["post_scf_vv10_ene"] = values
                original = collector.data.copy()
                with patch("collect_info.pd.read_csv") as read_csv:
                    collector.add_d3bj_correction()
                read_csv.assert_not_called()
                pd.testing.assert_frame_equal(collector.data, original)

    def test_summary_uses_selected_energy(self):
        dataset = {
            "reaction-S": {
                "r": {"systems": ["1"], "stoichiometry": [1], "reference": 1.0, "weight": 2.0}
            }
        }
        for column, label in (
            ("scf_d3bj_ene", "AI(BJ)"),
            ("post_scf_vv10_ene", "post_scf_vv10_ene"),
        ):
            with self.subTest(column=column):
                collector = self.make_collector(energy_column=column)
                collector.full_subset_dict = {"set": ["S"]}
                collector.name_subset_list = ["S"]
                collector.name_set_list = ["set"]
                collector.data = pd.DataFrame({
                    "name": ["S-1_basis"],
                    "scf_ene": [0.0],
                    "scf_d3bj_ene": [3.0 / AU2KCALMOL],
                    "post_scf_vv10_ene": [5.0 / AU2KCALMOL],
                })
                with (
                    patch("builtins.open", mock_open(read_data=json.dumps(dataset))),
                    patch("collect_info.Path.exists", return_value=False),
                    patch.object(pd.DataFrame, "to_csv"),
                    patch.object(pd.DataFrame, "to_excel"),
                    patch.object(collector, "save_done_cache"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertTrue(collector.get_wtmad_2())
                summary = collector.data_frame_dict["wtmad_2_df"]
                expected = 4.0 if column == "scf_d3bj_ene" else 8.0
                self.assertAlmostEqual(summary.loc["summary", label], expected)
                other = "post_scf_vv10_ene" if column == "scf_d3bj_ene" else "AI(BJ)"
                self.assertNotIn(other, summary.columns)
