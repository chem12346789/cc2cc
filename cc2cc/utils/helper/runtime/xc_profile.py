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
        self.host_totals = {}
        self.pending_host_time = 0.0
        self.blocks = []
        self.ao_layout = None
        if not self.enabled:
            return

        import cupy as cp
        import torch

        self.cp = cp
        self.torch = torch
        self.device = cp.cuda.Device().id
        self.stream = cp.cuda.get_current_stream()
        self.torch_stream = cp.cuda.ExternalStream(
            torch.cuda.current_stream(device=self.device).cuda_stream
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

    def record_block(self, base_grids, cube_points):
        if self.enabled:
            self.blocks.append((base_grids, cube_points))

    def start_host_timer(self):
        return perf_counter() if self.enabled else None

    def record_host_time(self, stage, started):
        if started is None:
            return
        elapsed = perf_counter() - started
        count, total = self.host_totals.get(stage, (0, 0.0))
        self.host_totals[stage] = (count + 1, total + elapsed)
        self.pending_host_time += elapsed

    def record_ao_layout(self, ao, weights, buffer):
        if self.enabled and self.ao_layout is None:
            self.ao_layout = (
                ao.shape, ao.strides, str(ao.dtype), ao[0].flags.c_contiguous,
                weights.shape, weights.strides, str(weights.dtype),
                weights.flags.c_contiguous, buffer.nbytes,
            )

    def mark(self, stage):
        """Accumulate work since the previous boundary under ``stage``."""
        if not self.enabled:
            return
        # Join both libraries' streams before attributing their work to a stage.
        self.torch_end.record(self.torch_stream)
        self.stream.wait_event(self.torch_end)
        self.end.record(self.stream)
        sync_start = perf_counter()
        self.end.synchronize()
        sync_wait = perf_counter() - sync_start
        wall = (
            perf_counter() - self.wall_start - sync_wait - self.pending_host_time
        )
        gpu = self.cp.cuda.get_elapsed_time(self.start, self.end) / 1000
        count, wall_total, gpu_total, wait_total = self.totals.get(
            stage, (0, 0.0, 0.0, 0.0)
        )
        self.totals[stage] = (
            count + 1,
            wall_total + wall,
            gpu_total + gpu,
            wait_total + sync_wait,
        )
        self.pending_host_time = 0.0
        self._start()

    def report(self, logger, owner):
        if not self.enabled:
            return
        logger.info(
            owner, "XC profile %s (GPU elapsed includes host gaps)",
            self.spin,
        )
        wall_sum = sum(values[1] for values in self.totals.values())
        for stage, (count, wall, gpu, sync_wait) in self.totals.items():
            logger.info(
                owner,
                "XC profile %-20s calls=%d wall=%.6f s sync wait=%.6f s "
                "GPU elapsed=%.6f s",
                stage, count, wall, sync_wait, gpu,
            )
        for stage, (count, wall) in self.host_totals.items():
            logger.info(
                owner, "XC profile host %-16s calls=%d wall=%.6f s",
                stage, count, wall,
            )
        logger.info(owner, "XC profile total staged wall=%.6f s", wall_sum)
        free_bytes, total_bytes = self.cp.cuda.runtime.memGetInfo()
        cupy_pool = self.cp.get_default_memory_pool()
        torch_allocated = self.torch.cuda.memory_allocated(self.device)
        torch_reserved = self.torch.cuda.memory_reserved(self.device)
        logger.info(
            owner,
            "XC profile memory: GPU free=%d/%d bytes; CuPy pool used=%d "
            "total=%d bytes; PyTorch allocated=%d reserved=%d bytes",
            free_bytes, total_bytes,
            cupy_pool.used_bytes(), cupy_pool.total_bytes(),
            torch_allocated, torch_reserved,
        )
        if self.ao_layout is not None:
            logger.info(
                owner,
                "XC profile first block: AO shape=%s strides=%s dtype=%s "
                "row_contiguous=%s; weights shape=%s strides=%s dtype=%s "
                "contiguous=%s; scratch_bytes=%d",
                *self.ao_layout,
            )
