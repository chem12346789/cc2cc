"""Opt-in, synchronized profiling of GPU SCF integration stages."""

import os
from time import perf_counter


class XCStageProfiler:
    def __init__(self, spin):
        setting = os.environ.get("DFT2CC_PROFILE_XC", "0")
        if setting not in ("0", "1"):
            raise ValueError("DFT2CC_PROFILE_XC must be 0 or 1")
        self.enabled = setting == "1"
        self.spin = spin
        self.totals = {}
        if not self.enabled:
            return

        import cupy as cp
        import torch

        self.cp = cp
        self.stream = cp.cuda.get_current_stream()
        self.torch_stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(device=cp.cuda.Device().id).cuda_stream
        )
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()
        self.torch_end = cp.cuda.Event()
        self.stream.synchronize()
        self.torch_stream.synchronize()
        self._start()

    def _start(self):
        self.start.record(self.stream)
        self.torch_stream.wait_event(self.start)
        self.wall_start = perf_counter()

    def mark(self, stage):
        """Accumulate work since the previous boundary under ``stage``."""
        if not self.enabled:
            return
        # Join both libraries' streams before attributing their work to a stage.
        self.torch_end.record(self.torch_stream)
        self.stream.wait_event(self.torch_end)
        self.end.record(self.stream)
        self.end.synchronize()
        wall = perf_counter() - self.wall_start
        gpu = self.cp.cuda.get_elapsed_time(self.start, self.end) / 1000
        count, wall_total, gpu_total = self.totals.get(stage, (0, 0.0, 0.0))
        self.totals[stage] = (count + 1, wall_total + wall, gpu_total + gpu)
        self._start()

    def report(self, logger, owner):
        if not self.enabled:
            return
        logger.info(
            owner, "XC profile %s (synchronized; GPU elapsed includes host gaps)",
            self.spin,
        )
        wall_sum = sum(values[1] for values in self.totals.values())
        for stage, (count, wall, gpu) in self.totals.items():
            logger.info(
                owner,
                "XC profile %-20s calls=%d wall=%.6f s GPU elapsed=%.6f s",
                stage, count, wall, gpu,
            )
        logger.info(owner, "XC profile total staged wall=%.6f s", wall_sum)
