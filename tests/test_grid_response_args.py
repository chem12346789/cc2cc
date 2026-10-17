import argparse
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class TestGridResponseArgs(TestCase):
    def test_numeric_option(self):
        tree = ast.parse((ROOT / "test.py").read_text())
        option = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--grid_response"
        )
        parser = argparse.ArgumentParser()
        exec(compile(ast.Expression(option), "test.py", "eval"), {"parser": parser})
        self.assertEqual(parser.parse_args([]).grid_response, 0)
        for enabled in (0, 1):
            self.assertEqual(
                parser.parse_args(["--grid_response", str(enabled)]).grid_response,
                enabled,
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--grid_response", "2"])

    def test_runners_set_response_before_kernel(self):
        for spin in ("rks", "uks"):
            source = ROOT / "cc2cc" / f"test_model_{spin}.py"
            tree = ast.parse(source.read_text())
            block = next(
                node for node in ast.walk(tree)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "args.if_grad and args.max_cycle != -1"
            )
            code = compile(ast.Module(body=[block], type_ignores=[]), str(source), "exec")
            for enabled in (0, 1):
                with self.subTest(spin=spin, enabled=enabled):
                    gradient = Mock()

                    def kernel():
                        self.assertIs(gradient.grid_response, bool(enabled))
                        return np.zeros((1, 3), dtype=np.float64)

                    gradient.kernel.side_effect = kernel
                    mdft = Mock()
                    mdft.Gradients.return_value = gradient
                    namespace = {
                        "args": SimpleNamespace(
                            if_grad=True, max_cycle=1, grid_response=enabled
                        ),
                        "mdft": mdft,
                        "torch": SimpleNamespace(cuda=Mock(is_available=lambda: False)),
                        "utils": Mock(),
                        "modeldict": Mock(),
                        "np": np,
                    }
                    with contextlib.redirect_stdout(io.StringIO()):
                        exec(code, namespace)
                    gradient.kernel.assert_called_once_with()
