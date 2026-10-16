import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pyscf
import torch

from tests.test_modelscf_grid_response import ToyFunctional


def load_helper(name, namespace):
    # Isolate backend-independent logic when optional GPU imports are absent.
    path = Path(__file__).resolve().parents[1] / "cc2cc/utils/GridsGPU.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
         namespace)
    return namespace[name]


def load_cache_helper(device, factory):
    return load_helper("get_cached_gdftopt", dict(
        cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda: device)),
        _GDFTOpt=SimpleNamespace(from_mol=factory),
    ))


class TestDlpackProtocol(unittest.TestCase):
    def test_exchange_preserves_values_and_shares_output_storage(self):
        exchange = load_helper("eval_xc_eff_gpu", dict(
            torch=torch, cp=SimpleNamespace(from_dlpack=np.from_dlpack),
        ))
        functional = ToyFunctional(3)
        rho = np.linspace(0.1, 0.9, 24).reshape(2, 4, 3)
        weights = np.array([0.3, 0.7])
        expected = functional.eval_xc_eff(rho, weights)
        tensors = []
        evaluate = functional.eval_xc_eff

        def capture(*args, **kwargs):
            result = evaluate(*args, **kwargs)
            tensors.extend(result)
            return result

        with patch.object(functional, "eval_xc_eff", side_effect=capture):
            with patch.object(torch.Tensor, "cpu",
                              side_effect=AssertionError("host staging")):
                actual = exchange(functional, rho, weights)
        for array, tensor, reference in zip(actual, tensors, expected):
            self.assertEqual(array.ctypes.data, tensor.data_ptr())
            np.testing.assert_array_equal(array, reference)


class TestIntegrationOptimizerCache(unittest.TestCase):
    def test_reuse_and_invalidation(self):
        device = SimpleNamespace(id=0)
        factory = Mock(side_effect=lambda mol: object())
        get_optimizer = load_cache_helper(device, factory)
        mol = pyscf.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        grid = SimpleNamespace(opt=None)
        optimizer = get_optimizer(grid, mol)
        self.assertIs(get_optimizer(grid, mol), optimizer)
        self.assertIs(get_optimizer(grid, mol.copy()), optimizer)
        self.assertEqual(factory.call_count, 1)
        for change in (
            lambda: mol.set_geom_("He 0 0 1", unit="Bohr"),
            lambda: mol.build(basis="6-31g"),
            lambda: setattr(mol, "cart", True),
            lambda: setattr(device, "id", 1),
        ):
            change()
            previous_calls = factory.call_count
            replacement = get_optimizer(grid, mol)
            self.assertIsNot(replacement, optimizer)
            self.assertIs(get_optimizer(grid, mol), replacement)
            self.assertEqual(factory.call_count, previous_calls + 1)
            optimizer = replacement
        other_grid = SimpleNamespace(opt=None)
        self.assertIsNot(get_optimizer(other_grid, mol), optimizer)
        self.assertIsNone(grid.opt)

    def test_failed_build_does_not_validate_stale_cache(self):
        factory = Mock(side_effect=[object(), RuntimeError("build failed"), object()])
        get_optimizer = load_cache_helper(SimpleNamespace(id=0), factory)
        mol = pyscf.M(atom="He 0 0 0", basis="sto-3g", verbose=0)
        grid = SimpleNamespace()
        original = get_optimizer(grid, mol)
        mol.set_geom_("He 0 0 1", unit="Bohr")
        with self.assertRaisesRegex(RuntimeError, "build failed"):
            get_optimizer(grid, mol)
        self.assertIsNot(get_optimizer(grid, mol), original)
        self.assertEqual(factory.call_count, 3)


@unittest.skipUnless(
    importlib.util.find_spec("cupy") is not None
    and importlib.util.find_spec("gpu4pyscf") is not None,
    "requires CuPy and GPU4PySCF",
)
class TestGpuDlpackExchange(unittest.TestCase):
    def test_nondefault_streams_without_host_staging(self):
        import cupy as cp
        from cc2cc.utils.GridsGPU import eval_xc_eff_gpu

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = cp.cuda.Device().id
        functional = ToyFunctional(3)
        rho = np.linspace(0.1, 0.9, 24).reshape(2, 4, 3)
        weights = np.array([0.3, 0.7])
        expected = functional.eval_xc_eff(rho, weights)
        functional.device = f"cuda:{device}"
        cupy_stream = cp.cuda.Stream(non_blocking=True)
        torch_stream = torch.cuda.Stream(device=device)
        with cupy_stream, torch.cuda.stream(torch_stream):
            gpu_rho = cp.asarray(rho)
            gpu_weights = cp.asarray(weights)
            with patch.object(torch.Tensor, "cpu",
                              side_effect=AssertionError("host transfer")):
                actual = eval_xc_eff_gpu(functional, gpu_rho, gpu_weights)
                consumed = tuple(value + 0 for value in actual)
        cupy_stream.synchronize()
        for result, reference in zip(consumed, expected):
            self.assertIsInstance(result, cp.ndarray)
            self.assertEqual(result.device.id, device)
            self.assertEqual(result.dtype, np.float64)
            np.testing.assert_allclose(cp.asnumpy(result), reference, rtol=1e-13)
