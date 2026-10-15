import importlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


database_module = importlib.import_module("cc2cc.utils.DataBase")
model_module = importlib.import_module("cc2cc.utils.ModelClass")


class TrackingArchive(dict):
    def __init__(self, data):
        super().__init__(data)
        self.accessed = []
        self.closed = False

    def __getitem__(self, key):
        self.accessed.append(key)
        return super().__getitem__(key)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class GridModel(torch.nn.Module):
    before_weight = False
    cube_type = "cube"
    cube_size = 1
    input_level = 1

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(0.5, dtype=torch.float64, device="cpu")
        )

    def forward(self, input_):
        return input_.sum(dim=(1, 2)).unsqueeze(-1) * self.scale


class TrainingMemoryTests(unittest.TestCase):
    def make_database(self, if_abs=False, if_grad=False, if_eval=False):
        database = database_module.DataBase.__new__(database_module.DataBase)
        database.args = SimpleNamespace(
            loss_multiplier=1.0, loss_multiplier_abs=1.0,
            loss_multiplier_grad=1.0, loss_multiplier_atomic=1.0,
            rho_input="dft", output_target="tol_delta_grids",
            if_abs=if_abs, if_grad=if_grad, if_atomic=False,
            if_relative_weight=False, if_relative_weight_abs=False,
            if_relative_weight_grad=False, mol0_weighting=1.0,
        )
        database.dtype = torch.float64
        database.if_eval = if_eval
        database.verbose = False
        database.process_input = lambda x: x
        database.process_grad2force = lambda x: x
        database.gpu_key = (
            "input", "weight", "output", "energy_target", "ae_target",
            "grad2force", "grad_cc_train",
        )
        return database

    def load_archives(self, database, base, addon=None):
        archives = [base] if addon is None else [base, addon]
        with patch.object(database_module.np, "load", side_effect=archives), \
                patch.object(Path, "exists", return_value=addon is not None):
            return database.load_data({"natm": 2, "elements": ["H", "H"]}, "test")

    def make_archive(self):
        return TrackingArchive({
            "weights": np.ones(3, dtype=np.float64),
            "rho_cube_dft": np.array([1.0, 0.0, 2.0]).reshape(3, 1, 1),
            "e_cc": np.array(6.0), "e_dft": np.array(1.0),
            "tol_delta_grids": np.array([2.0, 0.0, 3.0]),
            "grad2force": np.ones((3, 1, 1, 1), dtype=np.float64),
            "grad_cc_train": np.array([1.0]),
            "unused": np.ones(10, dtype=np.float64),
        })

    def test_lazy_addon_override_and_disabled_targets(self):
        database = self.make_database()
        base = self.make_archive()
        addon = TrackingArchive({"e_cc": np.array(9.0), "unused_addon": object()})
        _, batch = self.load_archives(database, base, addon)
        self.assertEqual(batch["energy_target"].item(), 8.0)
        self.assertEqual(batch["output"].ndim, 0)
        self.assertNotIn("e_cc", base.accessed)
        for archive in (base, addon):
            self.assertTrue(archive.closed)
            self.assertNotIn("unused", archive.accessed)
            self.assertNotIn("unused_addon", archive.accessed)
            self.assertNotIn("tol_delta_grids", archive.accessed)
            self.assertNotIn("grad2force", archive.accessed)
        torch.testing.assert_close(
            batch["input"],
            torch.tensor([1.0, 2.0], dtype=torch.float64, device="cpu").reshape(2, 1, 1),
        )
        self.assertEqual(batch["input"].device.type, "cpu")

    def test_enabled_targets_keep_filtering_and_values(self):
        database = self.make_database(if_abs=True, if_grad=True)
        base = self.make_archive()
        _, batch = self.load_archives(database, base)
        self.assertTrue(base.closed)
        torch.testing.assert_close(
            batch["output"],
            torch.tensor([[2.0], [3.0]], dtype=torch.float64, device="cpu"),
        )
        self.assertEqual(batch["grad2force"].shape, (2, 1, 1, 1))
        torch.testing.assert_close(
            batch["grad_cc_train"],
            torch.tensor([1.0], dtype=torch.float64, device="cpu"),
        )

    def test_evaluation_skips_targets(self):
        database = self.make_database(if_abs=True, if_grad=True, if_eval=True)
        base = self.make_archive()
        _, batch = self.load_archives(database, base)
        self.assertEqual(batch["output"].ndim, 0)
        self.assertEqual(batch["grad2force"].ndim, 0)
        self.assertNotIn("tol_delta_grids", base.accessed)
        self.assertNotIn("grad2force", base.accessed)
        self.assertTrue(base.closed)

    def test_archives_close_on_failure(self):
        database = self.make_database()
        base = self.make_archive()
        del base["weights"]
        addon = TrackingArchive({})
        with self.assertRaises(KeyError):
            self.load_archives(database, base, addon)
        self.assertTrue(base.closed)
        self.assertTrue(addon.closed)

    def make_model(self, if_grad=False, if_abs=False, if_atomic=False):
        wrapper = model_module.ModelClass(SimpleNamespace(
            device="cpu", precision="float64", model="test",
            if_grad=if_grad, if_abs=if_abs, if_atomic=if_atomic,
            optimizer="AdamW", lr=0.1, weight_decay=0.0, loss_type="MSELoss",
        ))
        wrapper.model = GridModel()
        wrapper.device = torch.device("cpu")
        wrapper.dtype = torch.float64
        wrapper.loss_fun = torch.nn.MSELoss(reduction="sum")
        return wrapper

    def make_batch(self, if_grad=False):
        _, batch = self.load_archives(
            self.make_database(if_abs=True, if_grad=if_grad), self.make_archive()
        )
        return batch

    def test_input_gradients_only_for_active_force_loss(self):
        for configured, requested, available in (
            (False, True, False), (True, False, True),
            (True, True, False), (True, True, True),
        ):
            with self.subTest(
                configured=configured, requested=requested, available=available
            ):
                wrapper = self.make_model(if_grad=configured)
                batch = self.make_batch(if_grad=available)
                loss, _, _ = wrapper.loss(batch, if_grad=requested)
                active = configured and requested and available
                self.assertEqual(batch["input"].requires_grad, active)
                expected = torch.tensor(24.5, dtype=torch.float64, device="cpu")
                torch.testing.assert_close(loss, expected)
                loss.backward()
                torch.testing.assert_close(
                    wrapper.model.scale.grad,
                    torch.tensor(-42.0, dtype=torch.float64, device="cpu"),
                )

    def test_force_loss_preserves_second_derivatives(self):
        wrapper = self.make_model(if_grad=True)
        batch = self.make_batch(if_grad=True)
        batch["grad_cc_train"].fill_(2.0)
        loss, _, _ = wrapper.loss(batch)
        loss.backward()
        torch.testing.assert_close(
            loss, torch.tensor(26.5, dtype=torch.float64, device="cpu")
        )
        torch.testing.assert_close(
            wrapper.model.scale.grad,
            torch.tensor(-50.0, dtype=torch.float64, device="cpu"),
        )

    def test_absolute_loss_preserves_parameter_gradients(self):
        wrapper = self.make_model(if_abs=True)
        batch = self.make_batch()
        loss, _, _ = wrapper.loss(batch)
        loss.backward()
        self.assertFalse(batch["input"].requires_grad)
        torch.testing.assert_close(
            loss, torch.tensor(37.0, dtype=torch.float64, device="cpu")
        )
        torch.testing.assert_close(
            wrapper.model.scale.grad,
            torch.tensor(-64.0, dtype=torch.float64, device="cpu"),
        )

    def test_atomic_forward_transfers_only_inputs_and_weights(self):
        wrapper = self.make_model(if_atomic=True)
        database = self.make_database()
        atomic_batch = self.make_batch()
        database.dataset = SimpleNamespace(get_from_name=lambda name: atomic_batch)
        database.atomic_name_dict = {"H": "hydrogen"}
        wrapper.database_train = database
        batch = self.make_batch()
        batch["scale_atomic"] = 1.0
        with patch.object(database, "process_batch", wraps=database.process_batch) as transfer:
            loss, _, _ = wrapper.loss(batch)
        self.assertEqual(set(transfer.call_args.args[0]), {"input", "weight"})
        loss.backward()
        torch.testing.assert_close(
            loss, torch.tensor(26.75, dtype=torch.float64, device="cpu")
        )
        torch.testing.assert_close(
            wrapper.model.scale.grad,
            torch.tensor(-33.0, dtype=torch.float64, device="cpu"),
        )

    def test_model_checkpoint_reference_released_after_restore(self):
        wrapper = self.make_model()
        wrapper.state_dict = {"scale": torch.tensor(2.0, dtype=torch.float64, device="cpu")}
        with patch.object(wrapper, "load_model"), \
                patch.object(wrapper, "_maybe_compile_model"), \
                patch.object(Path, "exists", return_value=True), \
                patch.object(
                    model_module, "__import__",
                    return_value=SimpleNamespace(Model=GridModel), create=True,
                ):
            wrapper.init_model()
        self.assertIsNone(wrapper.state_dict)
        self.assertEqual(wrapper.model.scale.item(), 2.0)

    def test_optimizer_checkpoint_reference_released_after_restore(self):
        wrapper = self.make_model()
        optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=0.02)
        wrapper.model.scale.sum().backward()
        optimizer.step()
        wrapper.optimizer_state_dict = optimizer.state_dict()
        with patch.object(torch.nn.MSELoss, "cuda", return_value=wrapper.loss_fun):
            wrapper.init_train()
        self.assertIsNone(wrapper.optimizer_state_dict)
        self.assertEqual(wrapper.optimizer.param_groups[0]["lr"], 0.02)
        self.assertIn("exp_avg", wrapper.optimizer.state[wrapper.model.scale])

    def test_checkpoint_size_counts_stream_without_buffering(self):
        wrapper = self.make_model()
        wrapper.optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=0.1)
        state_dict = {
            "large": torch.ones(150000, dtype=torch.float64, device="cpu")
        }
        payload = {
            "state_dict": state_dict, "model": wrapper.args.model,
            "optimizer": wrapper.optimizer.state_dict(),
        }
        with BytesIO() as actual, model_module._CheckpointSizeBuffer() as counter:
            torch.save(payload, actual)
            torch.save(payload, counter)
            self.assertEqual(counter.size, actual.tell())
            self.assertEqual(counter.getbuffer().nbytes, 0)
            self.assertEqual(wrapper._estimate_checkpoint_size(state_dict), actual.tell())

    def test_real_npz_tensors_remain_valid_after_archive_close(self):
        database = self.make_database()
        with TemporaryDirectory() as directory:
            np.savez(Path(directory) / "data_test.npz", **self.make_archive())
            with patch.object(database_module, "DATA_PATH", Path(directory)):
                _, batch = database.load_data(
                    {"natm": 2, "elements": ["H", "H"]}, "test"
                )
            self.assertEqual(batch["input"].sum().item(), 3.0)


if __name__ == "__main__":
    unittest.main()
