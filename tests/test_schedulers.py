import unittest

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from cc2cc.utils.schedulers import CosineAnnealingWarmRestarts2


class CosineWarm2Tests(unittest.TestCase):
    def make_optimizer(self):
        parameters = [
            torch.nn.Parameter(torch.zeros(1, dtype=torch.float64, device="cpu"))
            for _ in range(2)
        ]
        return SGD(
            [{"params": [parameters[0]], "lr": 0.1},
             {"params": [parameters[1]], "lr": 0.2}]
        )

    def make_scheduler(self, optimizer):
        return CosineAnnealingWarmRestarts2(
            optimizer, T_0=4, T_mult=2, eta_min=0.001,
            restart_step=7, restart_lrs=[0.02, 0.04],
        )

    def test_restart_boundary_and_subsequent_cycles(self):
        optimizer = self.make_optimizer()
        scheduler = self.make_scheduler(optimizer)
        initial = CosineAnnealingWarmRestarts(
            self.make_optimizer(), T_0=4, T_mult=2, eta_min=0.001
        )
        for step in range(48):
            initial.base_lrs = [0.1, 0.2] if step < 12 else [0.02, 0.04]
            initial.step(step)
            reference = initial
            for actual, expected in zip(scheduler.get_last_lr(), reference.get_last_lr()):
                self.assertAlmostEqual(actual, expected)
            self.assertEqual(scheduler.last_epoch, step)
            self.assertEqual(scheduler.T_cur, reference.T_cur)
            self.assertEqual(scheduler.T_i, reference.T_i)
            optimizer.step()
            scheduler.step()

    def test_fixed_cycles_and_aligned_threshold(self):
        for threshold, first_restart in [(4, 4), (7, 8)]:
            with self.subTest(threshold=threshold):
                scheduler = CosineAnnealingWarmRestarts2(
                    self.make_optimizer(), T_0=4, T_mult=1, eta_min=0.001,
                    restart_step=threshold, restart_lrs=[0.02, 0.04],
                )
                reference = CosineAnnealingWarmRestarts(
                    self.make_optimizer(), T_0=4, T_mult=1, eta_min=0.001
                )
                for step in [0, 3.5, 4, 6.5, 7, 7.5, 8, 12]:
                    reference.base_lrs = (
                        [0.1, 0.2] if step < first_restart else [0.02, 0.04]
                    )
                    reference.step(step)
                    scheduler.step(step)
                    self.assertEqual(scheduler.get_last_lr(), reference.get_last_lr())
                    self.assertEqual(scheduler.T_cur, reference.T_cur)
                    self.assertEqual(scheduler.T_i, reference.T_i)

    def test_resume_and_state_dict(self):
        optimizer = self.make_optimizer()
        scheduler = self.make_scheduler(optimizer)
        for step in [0, 6, 7, 8, 11, 12, 13, 19, 27, 28]:
            scheduler.step(step)
            resumed = self.make_scheduler(self.make_optimizer())
            resumed.step(step)
            restored = self.make_scheduler(self.make_optimizer())
            restored.load_state_dict(scheduler.state_dict())
            self.assertEqual(resumed.get_last_lr(), scheduler.get_last_lr())
            scheduler.step()
            resumed.step()
            restored.step()
            self.assertEqual(resumed.get_last_lr(), scheduler.get_last_lr())
            self.assertEqual(restored.get_last_lr(), scheduler.get_last_lr())

    def test_invalid_restart_settings(self):
        for step, lrs in [
            (0, [0.02, 0.04]),
            (-1, [0.02, 0.04]),
            (1.5, [0.02, 0.04]),
            (7, [0.02]),
            (7, [float("nan"), 0.04]),
            (7, [0.0, 0.04]),
        ]:
            with self.subTest(step=step, lrs=lrs), self.assertRaises(ValueError):
                CosineAnnealingWarmRestarts2(
                    self.make_optimizer(), T_0=4, T_mult=2, eta_min=0.001,
                    restart_step=step, restart_lrs=lrs,
                )


if __name__ == "__main__":
    unittest.main()
