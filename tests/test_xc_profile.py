import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np


UTILS = Path(__file__).resolve().parents[1] / "cc2cc" / "utils"
spec = importlib.util.spec_from_file_location("xc_profile", UTILS / "xc_profile.py")
profiling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiling)


def fake_backends():
    events = [Mock(name=name) for name in ("start", "end", "torch_end")]
    stream = Mock(name="cupy_stream")
    torch_stream = Mock(name="torch_stream")
    cuda = SimpleNamespace(
        Event=Mock(side_effect=events),
        get_current_stream=Mock(return_value=stream),
        ExternalStream=Mock(return_value=torch_stream),
        Device=Mock(return_value=SimpleNamespace(id=2)),
        get_elapsed_time=Mock(return_value=250.0),
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(
        current_stream=Mock(return_value=SimpleNamespace(cuda_stream=1234)),
    ))
    return SimpleNamespace(cuda=cuda), torch, stream, torch_stream, events


def load_nodes(path, nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0,
    )
    tree = ast.fix_missing_locations(ast.Module(
        body=[future] + nodes, type_ignores=[],
    ))
    exec(compile(tree, str(path), "exec"), namespace)


class TestXCStageProfiler(unittest.TestCase):
    def test_disabled_never_imports_gpu_backends_or_logs(self):
        with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "0"}):
            with patch("builtins.__import__", side_effect=AssertionError("import")):
                profile = profiling.XCStageProfiler("RKS")
                profile.record_block(2, 6)
                profile.mark("AO evaluation")
                logger = Mock()
                profile.report(logger, object())
        logger.info.assert_not_called()
        self.assertEqual(profile.totals, {})
        self.assertEqual(profile.blocks, [])

    def test_invalid_setting_is_explicit(self):
        with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "yes"}):
            with self.assertRaisesRegex(ValueError, "must be 0 or 1"):
                profiling.XCStageProfiler("RKS")

    def test_stream_ordering_accumulation_and_report(self):
        cp, torch, stream, torch_stream, events = fake_backends()
        start, end, torch_end = events
        order = Mock()
        for name, child in (("start", start), ("end", end),
                            ("torch_end", torch_end), ("cupy", stream),
                            ("torch", torch_stream)):
            order.attach_mock(child, name)
        with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "1"}), \
                patch.dict("sys.modules", {"cupy": cp, "torch": torch}), \
                patch.object(profiling, "perf_counter", side_effect=[0, 1, 2, 5, 6]):
            profile = profiling.XCStageProfiler("UKS")
            profile.record_block(2, 6)
            profile.record_block(1, 3)
            order.reset_mock()
            profile.mark("density")
            first_calls = [call[0] for call in order.mock_calls]
            self.assertEqual(first_calls, [
                "torch_end.record", "cupy.wait_event", "end.record",
                "end.synchronize", "start.record", "torch.wait_event",
            ])
            torch_end.record.assert_called_with(torch_stream)
            stream.wait_event.assert_called_with(torch_end)
            end.record.assert_called_with(stream)
            torch_stream.wait_event.assert_called_with(start)
            profile.mark("density")
        torch.cuda.current_stream.assert_called_once_with(device=2)
        cp.cuda.ExternalStream.assert_called_once_with(1234)
        cp.cuda.get_elapsed_time.assert_called_with(start, end)
        self.assertEqual(profile.totals["density"], (2, 4.0, 0.5))
        logger = Mock()
        owner = object()
        profile.report(logger, owner)
        self.assertEqual(logger.info.call_args_list[1].args[2:],
                         ("density", 2, 4.0, 0.5))
        self.assertEqual(logger.info.call_args_list[-2].args[2:], (1, 2, 6))
        self.assertEqual(logger.info.call_args_list[-1].args[2:], (2, 1, 3))


class TestProfiledIntegration(unittest.TestCase):
    def test_rks_uks_stages_preserve_integration_results(self):
        # Run the actual integration functions with small NumPy backend doubles.
        path = UTILS / "GridsGPU.py"
        cube_node = next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.ClassDef) and node.name == "GridCube")
        numint = SimpleNamespace(eval_ao=lambda mol, coords, **kwargs:
                                 np.ones((4, mol.nao, len(coords))))
        namespace = dict(
            cp=np, asarray=np.asarray, numint=numint,
            rho_evaluator=lambda ni, mol, ao, dm, **kwargs:
            np.full((4, ao.shape[-1]), dm.sum()),
        )
        load_nodes(path, [cube_node], namespace)
        cube_class = namespace["GridCube"]
        mol = SimpleNamespace(nao=2)
        opt = SimpleNamespace(
            _sorted_mol=mol,
            sort_orbitals=lambda data, **kwargs: data,
            unsort_orbitals=lambda data, **kwargs: data,
        )
        ni = SimpleNamespace(
            _xc_type=lambda code: "GGA",
            eval_xc_eff=lambda code, rho, **kwargs:
            (np.ones(rho.shape[-1]), np.ones_like(rho)),
        )
        for spin in ("rks", "uks"):
            for input_level in (4, 7):
                with self.subTest(spin=spin, input_level=input_level):
                    grid = SimpleNamespace(
                        input_level=input_level, cube_type="cube", cube_size=3,
                        cutoff=1e-15, opt=None,
                    )
                    grid.gen_cube = lambda mol, dm, coords: cube_class(
                        np.repeat(coords[:, None, :], 3, axis=1), grid, mol,
                    )
                    path = UTILS / f"modelscf_{spin}_gpu.py"
                    wrapper = next(node for node in ast.parse(path.read_text()).body
                                   if isinstance(node, ast.FunctionDef)
                                   and node.name == f"get_veff_modified_{spin}_gpu")
                    nr = next(node for node in wrapper.body
                              if isinstance(node, ast.FunctionDef)
                              and node.name == f"nr_{spin}")
                    logger = Mock()
                    profiles = []

                    def make_profile(label):
                        profile = profiling.XCStageProfiler(label)
                        profiles.append(profile)
                        return profile

                    def gemm(transa, transb, a, b, *, out, beta):
                        self.assertEqual((transa, transb, beta), ("N", "T", 1.0))
                        self.assertEqual(out.dtype, np.float64)
                        self.assertTrue(out.flags.c_contiguous)
                        out[:] = a @ b.T + beta * out
                        return out

                    buffers = []

                    def scale_ao(ao, wv, *, out):
                        buffers.append(out)
                        result = out[:ao[0].size].reshape(ao[0].shape)
                        np.einsum("xag,xg->ag", ao, wv, out=result)
                        return result

                    ns = dict(
                        cp=np, ks=object(), logger=logger,
                        gemm=Mock(side_effect=gemm),
                        XCStageProfiler=make_profile,
                        get_cached_gdftopt=lambda grid, mol: opt,
                        tag_array=lambda data, **kwargs: data,
                        iterate_grid_segments=lambda *args, **kwargs: iter([
                            (np.ones(2), np.zeros((2, 3))),
                            (np.ones(1), np.zeros((1, 3))),
                        ]),
                        eval_xc_eff_gpu=lambda model, rho, weights:
                        (rho.sum(axis=(1, 2)) * weights, np.ones_like(rho)),
                        _scale_ao=scale_ao,
                        transpose_sum=lambda mat: mat + mat.transpose(0, 2, 1),
                        _hermi_sum=lambda mat: mat + mat.transpose(0, 2, 1),
                    )
                    load_nodes(path, [nr], ns)
                    dm = np.eye(2) if spin == "rks" else np.stack([np.eye(2)] * 2)
                    args = (SimpleNamespace(model=SimpleNamespace(cube_size=3)),
                            ni, mol, grid, "b3lyp", dm, 4000)
                    with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "0"}):
                        expected = ns[f"nr_{spin}"](*args)
                    self.assertEqual(ns["gemm"].call_count, 2 if spin == "rks" else 4)
                    self.assertEqual(len(buffers), 2 if spin == "rks" else 4)
                    self.assertTrue(all(buf is buffers[0] for buf in buffers))
                    # With these constant AO/rho fixtures the original dot-and-add
                    # path gives this matrix after Hermitian symmetrization.
                    expected_value = 9 * 2 * (0.5 * input_level + 3 * (input_level - 2))
                    np.testing.assert_array_equal(expected[2], np.full(
                        (2, 2) if spin == "rks" else (2, 2, 2), expected_value,
                    ))
                    logger.info.assert_not_called()
                    cp, torch, *_ = fake_backends()
                    with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "1"}), \
                            patch.dict("sys.modules", {"cupy": cp, "torch": torch}):
                        actual = ns[f"nr_{spin}"](*args)
                    for before, after in zip(expected, actual):
                        np.testing.assert_array_equal(before, after)
                    stages = profiles[-1].totals
                    self.assertEqual(list(stages), [
                        "setup", "cube coordinates", "AO evaluation", "density",
                        "component XC", "cube reshape", "model + DLPack",
                        "XC contraction", "AO weighting", "matrix GEMM",
                        "symmetry + unsort",
                    ])
                    self.assertEqual(profiles[-1].blocks, [(2, 6), (1, 3)])
                    for name, (count, _, _) in stages.items():
                        self.assertEqual(count, 1 if name in (
                            "setup", "symmetry + unsort",
                        ) else 4 if spin == "uks" and name in (
                            "AO weighting", "matrix GEMM",
                        ) else 2)


@unittest.skipUnless(
    importlib.util.find_spec("cupy") is not None
    and importlib.util.find_spec("torch") is not None,
    "requires CuPy and PyTorch",
)
class TestCudaProfile(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("gpu4pyscf") is not None, "requires GPU4PySCF",
    )
    def test_weighted_ao_scratch_reuse_matches_original_assembly(self):
        import cupy as cp
        from cupy.cublas import gemm
        from gpu4pyscf.dft.numint import _scale_ao
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        rng = np.random.default_rng(42)
        device = cp.cuda.Device().id
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            for spin, shape in (("rks", (1, 7, 7)), ("uks", (2, 1, 7, 7))):
                path = UTILS / f"modelscf_{spin}_gpu.py"
                assembly = next(node for node in ast.walk(ast.parse(path.read_text()))
                                if isinstance(node, ast.For)
                                and isinstance(node.iter, ast.Call)
                                and isinstance(node.iter.func, ast.Name)
                                and node.iter.func.id == "block_loop")
                blocks = []
                actual = cp.asarray(rng.normal(size=shape), dtype=cp.float64)
                expected = actual.copy()
                for points in (11, 5, 13):
                    ao = cp.asarray(rng.normal(size=(4, 7, points)), dtype=cp.float64)
                    wshape = (4, points) if spin == "rks" else (2, 4, points)
                    wv = cp.asarray(rng.normal(size=wshape), dtype=cp.float64)
                    blocks.append((ao, wv))
                    weights = wv.copy()
                    if spin == "rks":
                        weights[0] *= 0.5
                        expected[0] += cp.dot(ao[0], _scale_ao(ao, weights).T)
                    else:
                        weights[:, 0] *= 0.5
                        for s in range(2):
                            expected[s, 0] += cp.dot(
                                ao[0], _scale_ao(ao, weights[s]).T,
                            )
                buffers = []

                def scale_ao(ao, wv, *, out):
                    buffers.append(out)
                    weighted = _scale_ao(ao, wv, out=out)
                    self.assertEqual(weighted.data.ptr, out.data.ptr)
                    return weighted

                ns = dict(
                    cp=cp, gemm=gemm, _scale_ao=scale_ao, vmat=actual,
                    block_loop=lambda deriv: iter(blocks), ao_deriv=1,
                    aow_buffer=None, profile=Mock(),
                )
                load_nodes(path, [assembly], ns)
                per_block = 1 if spin == "rks" else 2
                self.assertEqual(len(buffers), 3 * per_block)
                self.assertTrue(all(buf is buffers[0] for buf in buffers[:2 * per_block]))
                self.assertIsNot(buffers[-1], buffers[0])
                self.assertTrue(all(buf is buffers[-1] for buf in buffers[2 * per_block:]))
                cp.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_gemm_accumulates_into_rks_and_uks_views(self):
        import cupy as cp
        from cupy.cublas import gemm
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = cp.cuda.Device().id
        rng = np.random.default_rng(14)
        with cp.cuda.Device(device), cp.cuda.Stream(non_blocking=True):
            for spin, shape in (("rks", (1, 7, 7)), ("uks", (2, 1, 7, 7))):
                path = UTILS / f"modelscf_{spin}_gpu.py"
                calls = [node for node in ast.walk(ast.parse(path.read_text()))
                         if isinstance(node, ast.Expr)
                         and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Name)
                         and node.value.func.id == "gemm"]
                self.assertEqual(len(calls), 1 if spin == "rks" else 2)
                actual = cp.asarray(rng.normal(size=shape), dtype=cp.float64)
                expected = actual.copy()
                pointer = actual.data.ptr
                for points in (11, 5, 13):
                    ao = cp.asarray(rng.normal(size=(4, 7, points)), dtype=cp.float64)
                    aowa = cp.asarray(rng.normal(size=(7, points)), dtype=cp.float64)
                    aowb = cp.asarray(rng.normal(size=(7, points)), dtype=cp.float64)
                    if spin == "rks":
                        expected[0] += cp.dot(ao[0], aowa.T)
                    else:
                        expected[0, 0] += cp.dot(ao[0], aowa.T)
                        expected[1, 0] += cp.dot(ao[0], aowb.T)
                    load_nodes(path, calls, dict(
                        gemm=gemm, ao=ao, aow=aowa, aowa=aowa, aowb=aowb,
                        vmat=actual,
                    ))
                    self.assertEqual(actual.data.ptr, pointer)
                    cp.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_nondefault_streams(self):
        import cupy as cp
        import torch

        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        device = cp.cuda.Device().id
        cupy_stream = cp.cuda.Stream(non_blocking=True)
        torch_stream = torch.cuda.Stream(device=device)
        with patch.dict(os.environ, {"DFT2CC_PROFILE_XC": "1"}), \
                cupy_stream, torch.cuda.stream(torch_stream):
            profile = profiling.XCStageProfiler("RKS")
            values = cp.ones(8, dtype=cp.float64)
            profile.mark("CuPy")
            tensor = torch.utils.dlpack.from_dlpack(values)
            result = tensor * torch.full(
                (8,), 3.0, dtype=torch.float64, device=f"cuda:{device}",
            )
            profile.mark("PyTorch")
            output = cp.from_dlpack(result)
            output += 2
            profile.mark("CuPy")
        cp.testing.assert_array_equal(output, cp.full(8, 5.0, dtype=cp.float64))
        self.assertEqual(profile.totals["CuPy"][0], 2)
        self.assertEqual(profile.totals["PyTorch"][0], 1)
        for _, wall, gpu in profile.totals.values():
            self.assertGreaterEqual(wall, 0)
            self.assertGreaterEqual(gpu, 0)


if __name__ == "__main__":
    unittest.main()
