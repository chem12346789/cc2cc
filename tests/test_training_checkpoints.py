import importlib
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
)

from cc2cc.utils.schedulers import CosineAnnealingWarmRestarts2


training = importlib.import_module("cc2cc.train_model")


class TrainingCheckpointTests(unittest.TestCase):
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
