import importlib
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
)

from cc2cc.utils.schedulers import CosineAnnealingWarmRestarts2


training = importlib.import_module("cc2cc.train_model")


class TrainingCheckpointTests(unittest.TestCase):
    def test_resume_restores_model_optimizer_and_numeric_epoch(self):
        module = importlib.import_module("cc2cc.utils.ModelClass")
        model = torch.nn.Linear(1, 1, dtype=torch.float64, device="cpu")
        restored = torch.nn.Linear(1, 1, dtype=torch.float64, device="cpu")
        restored.cube_type, restored.cube_size, restored.input_level = "cube", 3, 1
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
        model(torch.ones(1, 1, dtype=torch.float64, device="cpu")).sum().backward()
        optimizer.step()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_source"
            path.mkdir()
            torch.save({
                "state_dict": model.state_dict(), "model": "test",
                "optimizer": optimizer.state_dict(),
            }, path / "2400.pth")
            args = SimpleNamespace(
                device="cpu", distributed=False, load="source", load_epoch="2400",
                if_resume=True, precision="float64", save_dir="resumed",
                optimizer="AdamW", lr=1e-5, weight_decay=1e-12, loss_type="MSELoss",
                scheduler="constant",
            )
            wrapper = training.ModelClass(args)
            with patch.object(module, "CHECKPOINTS_PATH", Path(directory)), \
                    patch.object(Path, "exists", return_value=True), \
                    patch.object(module, "__import__", return_value=SimpleNamespace(
                        Model=lambda: restored), create=True), \
                    patch.object(wrapper, "_maybe_compile_model"), \
                    patch.object(torch.nn.MSELoss, "cuda", return_value=torch.nn.MSELoss()):
                wrapper.init_model(init_train=True)
            self.assertEqual(wrapper.start_step, 2401)
            resumed, _, _, saved, updates = self.run_training(
                "constant", start_step=wrapper.start_step, stride=1, end_epoch=2401,
            )
            self.assertEqual(resumed.train_model.call_count, 1)
            self.assertEqual([entry[0] for entry in saved], [2401])
            self.assertEqual(len(updates), 2)
            self.assertIsNone(wrapper.state_dict)
            self.assertIsNone(wrapper.optimizer_state_dict)
            self.assertEqual(wrapper.optimizer.param_groups[0]["lr"], 0.0003)
            database = MagicMock()
            database.__len__.return_value = 572
            database.data_gpu = [None] * 572
            with patch.object(module, "DataBase", return_value=database):
                wrapper.init_database({}, {})
            self.assertEqual(wrapper.scheduler.last_epoch, 2401 * 572)
            self.assertEqual(wrapper.optimizer.param_groups[0]["lr"], 1e-5)
            for parameter, loaded in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(parameter, loaded))
                for key, value in optimizer.state[parameter].items():
                    self.assertTrue(torch.equal(wrapper.optimizer.state[loaded][key], value))
            for _ in range(3):
                wrapper.optimizer.step()
                wrapper.scheduler.step()
                self.assertEqual(wrapper.optimizer.param_groups[0]["lr"], 1e-5)

    def test_resume_requires_checkpoint_and_optimizer(self):
        module = importlib.import_module("cc2cc.utils.ModelClass")
        args = SimpleNamespace(
            device="cpu", distributed=False, load="", load_epoch="2400", if_resume=True,
        )
        wrapper = training.ModelClass(args)
        with self.assertRaisesRegex(ValueError, "requires --load"):
            wrapper.load_model()
        args.load = "source"
        with TemporaryDirectory() as directory, \
                patch.object(module, "CHECKPOINTS_PATH", Path(directory)):
            with self.assertRaises(FileNotFoundError):
                wrapper.load_model()
            path = Path(directory) / "checkpoint_source"
            path.mkdir()
            torch.save({"state_dict": {}, "model": "test"}, path / "2400.pth")
            with self.assertRaisesRegex(ValueError, "no optimizer state"):
                wrapper.load_model()

    def run_training(
        self, scheduler_kind, start_step=0, stride=160, cycle_epochs=2, end_epoch=13
    ):
        parameter = torch.nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64, device="cpu")
        )
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        if scheduler_kind == "cosine_warm":
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=cycle_epochs * 2, T_mult=2
            )
        elif scheduler_kind == "cosine_warm2":
            scheduler = CosineAnnealingWarmRestarts2(
                optimizer, T_0=cycle_epochs * 2, T_mult=2, eta_min=0.001,
                restart_step=4, restart_lrs=[0.02],
            )
        else:
            scheduler = ConstantLR(optimizer, factor=1.0, total_iters=100)
        if start_step:
            scheduler.step(start_step * 2)

        record = Mock()
        record.data_dict = {"loss_ene": np.array([1.0])}
        model = Mock()
        model.args = SimpleNamespace(distributed=False)
        model.scheduler = scheduler
        model.optimizer = optimizer
        model.start_step = start_step
        model.database_train.data_gpu = [None, None]
        model.train_model.return_value = record
        model.eval_model.return_value = record
        update_lrs = []
        saved = []

        def train_model(if_grad):
            for _ in range(2):
                update_lrs.append(optimizer.param_groups[0]["lr"])
                optimizer.zero_grad(set_to_none=True)
                parameter.square().backward()
                optimizer.step()
                scheduler.step()
            return record

        model.train_model.side_effect = train_model
        model.save_model.side_effect = lambda epoch: saved.append(
            (epoch, parameter.detach().clone(), len(update_lrs))
        )
        best_loss = Mock()
        best_loss.update.return_value = False
        logger = training._Logger(Mock(), Mock(), best_loss, Path("loss"), stride)
        args = SimpleNamespace(seed=None, epoch=end_epoch, grad_step=1, eval_step=5)
        with (
            patch.object(training, "ModelClass", return_value=model),
            patch.object(training._Logger, "setup", return_value=logger),
        ):
            training.train_model({}, {}, args)
        return model, logger, record, saved, update_lrs

    def test_last_evaluations_force_checkpoint_and_loss_records(self):
        for kind in ("cosine_warm", "cosine_warm2"):
            with self.subTest(scheduler=kind):
                model, logger, record, saved, lrs = self.run_training(kind)
                self.assertEqual([entry[0] for entry in saved], [0, 5, 10])
                self.assertEqual(model.eval_model.call_count, 4)
                self.assertEqual(
                    [call.args[0]["epoch"] for call in logger.run.log.call_args_list],
                    [0, 5, 10, 13],
                )
                self.assertEqual(
                    [call.args[0] for call in record.save.call_args_list],
                    [
                        Path("loss") / f"{prefix}-{epoch}.csv"
                        for epoch in (0, 5, 10)
                        for prefix in ("train", "eval")
                    ],
                )
                self.assertEqual(saved[1][2], 12)
                self.assertEqual(saved[2][2], 22)
                self.assertLess(lrs[3], lrs[4])
                self.assertLess(lrs[11], lrs[12])
                self.assertTrue(
                    torch.allclose(saved[2][1], saved[1][1] *
                                   np.prod([1 - 2 * lr for lr in lrs[12:22]]))
                )

    def test_resumed_scheduler_uses_remaining_cycle(self):
        _, _, _, saved, _ = self.run_training("cosine_warm", start_step=3)
        self.assertEqual([entry[0] for entry in saved], [5, 10])

    def test_requested_160_epoch_schedule(self):
        for kind in ("cosine_warm", "cosine_warm2"):
            with self.subTest(scheduler=kind), patch("builtins.print"):
                _, _, _, saved, _ = self.run_training(
                    kind, stride=0, cycle_epochs=160, end_epoch=10080
                )
                self.assertEqual(
                    [entry[0] for entry in saved],
                    [155, 475, 1115, 2395, 4955, 10075],
                )

    def test_restart_on_next_evaluation_does_not_save_early(self):
        _, _, _, saved, _ = self.run_training(
            "cosine_warm", stride=0, cycle_epochs=6, end_epoch=18
        )
        self.assertEqual([entry[0] for entry in saved], [5, 15])

    def test_other_schedulers_preserve_evaluation_and_save_stride(self):
        model, logger, _, saved, _ = self.run_training("constant", stride=10)
        self.assertEqual([entry[0] for entry in saved], [0, 10])
        self.assertEqual(model.eval_model.call_count, 4)
        self.assertEqual(
            [call.args[0]["epoch"] for call in logger.run.log.call_args_list],
            [0, 5, 10, 13],
        )

    def test_best_loss_still_saves_without_forcing(self):
        model = Mock()
        model.optimizer.param_groups = [{"lr": 0.1}]
        record = Mock()
        record.data_dict = {"loss_ene": np.array([1.0])}
        best_loss = Mock()
        best_loss.update.return_value = True
        logger = training._Logger(Mock(), Mock(), best_loss, Path("loss"), 160)
        logger.log(model, record, record, 7)
        model.save_model.assert_called_once_with(7)

    def test_forced_checkpoint_is_written_and_loadable(self):
        with TemporaryDirectory() as directory:
            model = training.ModelClass(
                SimpleNamespace(device="cpu", distributed=False, model="test")
            )
            model.model = torch.nn.Linear(
                1, 1, dtype=torch.float64, device="cpu"
            )
            model.optimizer = torch.optim.SGD(model.model.parameters(), lr=0.1)
            model.dir_checkpoint = Path(directory) / "checkpoint"
            best_loss = Mock()
            best_loss.update.return_value = False
            record = Mock()
            record.data_dict = {"loss_ene": np.array([1.0])}
            logger = training._Logger(
                Mock(), Mock(), best_loss, model.dir_checkpoint / "loss", 160
            )
            logger.log(model, record, record, 155, force_checkpoint=True)
            checkpoint = torch.load(
                model.dir_checkpoint / "155.pth", weights_only=True,
                map_location="cpu",
            )
            self.assertEqual(checkpoint["model"], "test")
            self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 0.1)
            for name, parameter in model.model.state_dict().items():
                self.assertTrue(torch.equal(checkpoint["state_dict"][name], parameter))


if __name__ == "__main__":
    unittest.main()
