import math

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineAnnealingWarmRestarts2(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult, eta_min, restart_step, restart_lrs):
        if not isinstance(restart_step, int) or restart_step <= 0:
            raise ValueError("restart_step must be a positive optimizer-update count")
        if len(restart_lrs) != len(optimizer.param_groups):
            raise ValueError("restart_lrs must specify one peak per parameter group")
        if any(not math.isfinite(lr) or lr < eta_min for lr in restart_lrs):
            raise ValueError("restart learning rates must be finite and >= eta_min")
        self.restart_step = restart_step
        self.restart_lrs = list(restart_lrs)
        self.initial_peak_lrs = [
            group.get("initial_lr", group["lr"]) for group in optimizer.param_groups
        ]
        super().__init__(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)

    def step(self, epoch=None):
        absolute_step = self.last_epoch + 1 if epoch is None else epoch
        if absolute_step >= self.restart_step:
            self.base_lrs = list(self.restart_lrs)
            cycle_step = absolute_step - self.restart_step
        else:
            self.base_lrs = list(self.initial_peak_lrs)
            cycle_step = absolute_step
        # Reset both the phase and the growing period at the forced restart.
        super().step(cycle_step)
        self.last_epoch = math.floor(absolute_step)
