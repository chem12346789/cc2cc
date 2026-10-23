import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


class TestGenDataEvaluation(unittest.TestCase):
    def test_gradient_generation_is_training_only(self):
        source = Path(__file__).resolve().parents[1] / "gen_data.py"
        tree = ast.parse(source.read_text())
        blocks = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "not evaluate"
            and "grad_post_vv10" in ast.unparse(node)
        ]
        self.assertEqual(len(blocks), 3)
        code = compile(ast.Module(body=blocks, type_ignores=[]), str(source), "exec")
        for evaluate in (True, False):
            for natm in (1, 2):
                with self.subTest(evaluate=evaluate, natm=natm):
                    np = Mock()
                    np.zeros.return_value = 0.0
                    np.max.return_value = 0.0
                    mf = Mock()
                    mf.nuc_grad_method.return_value.kernel.return_value = 0.0
                    namespace = dict(
                        evaluate=evaluate, mol=Mock(natm=natm), np=np,
                        dm_dft=Mock(), nlcgrids=Mock(), args=Mock(grid_level=0),
                        converged_b3lyp=Mock(return_value=mf),
                        post_dft_vv10_gradient=Mock(return_value=0.0),
                        post_vv10_finite_difference_gradient=Mock(return_value=0.0),
                        b3lyp_finite_difference_gradient=Mock(return_value=0.0),
                        data_dict_addon={}, print=Mock(),
                    )
                    exec(code, namespace)
                    expected_calls = int(not evaluate and natm > 1)
                    for name in (
                        "converged_b3lyp", "post_dft_vv10_gradient",
                        "post_vv10_finite_difference_gradient",
                        "b3lyp_finite_difference_gradient",
                    ):
                        self.assertEqual(namespace[name].call_count, expected_calls)
                    self.assertEqual(
                        mf.nuc_grad_method.call_count, expected_calls
                    )
                    self.assertEqual(
                        np.zeros.call_count, int(not evaluate and natm == 1)
                    )
                    if evaluate:
                        self.assertNotIn("grad_post_vv10", namespace)
                        self.assertEqual(namespace["data_dict_addon"], {})
                        namespace["print"].assert_not_called()
                    else:
                        self.assertEqual(
                            namespace["data_dict_addon"]["grad_post_vv10"], 0.0
                        )
                        self.assertIn(
                            "Post-DFT VV10 gradient",
                            namespace["print"].call_args.args[0],
                        )
