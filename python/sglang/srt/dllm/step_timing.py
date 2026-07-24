"""Lightweight per-step device/host timing for the dLLM decode loop.

Splits each denoise step into:
  - device: the model forward wall-clock, from two device timing events straddling
    the forward (matches a profiler trace's kernel activity, without per-op hooks).
  - host:   step wall-clock (period between consecutive forwards) minus device =
    the scheduling + result-processing CPU work (JointThreshold unmask decision,
    the per-request commit loop, streaming, KV free).

Only two events + a perf_counter read per step, so it does not perturb host timing
the way torch.profiler / msprof do. Reports a running mean every `interval` steps.

Gated by SGLANG_DEBUG_DLLM_STEP_TIMING; constructed only when enabled so the hot
path pays nothing when off.
"""

import contextlib
import logging
import time

import torch

logger = logging.getLogger(__name__)


class DllmStepTimer:
    """Accumulate device/host us per denoise step and log a running mean.

    Reads the *previous* step's events at each step boundary, so completion is
    guaranteed by the natural step cadence (a full step + its copy sync elapse
    before the events are read) — no extra device synchronize on the hot path.
    """

    def __init__(self, device: str, interval: int):
        self._module = torch.get_device_module(device)
        self._interval = max(1, interval)
        self._prev_start = None
        self._prev_end = None
        self._prev_t0 = None
        self._n = 0
        self._dev_us = 0.0
        self._host_us = 0.0

    @contextlib.contextmanager
    def measure(self):
        start = self._module.Event(enable_timing=True)
        end = self._module.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._account(t0)
            self._prev_start, self._prev_end, self._prev_t0 = start, end, t0

    def _account(self, t0: float):
        if self._prev_start is None:
            return  # first step: no complete previous events yet
        dev_us = self._prev_start.elapsed_time(self._prev_end) * 1e3  # ms -> us
        wall_us = (t0 - self._prev_t0) * 1e6  # full step period
        host_us = max(0.0, wall_us - dev_us)
        self._dev_us += dev_us
        self._host_us += host_us
        self._n += 1
        if self._n >= self._interval:
            self._flush()

    def _flush(self):
        dev = self._dev_us / self._n
        host = self._host_us / self._n
        logger.info(
            "[dllm-step-timing] over %d steps: device %.2f ms  host %.2f ms  "
            "step %.2f ms  (host share %.0f%%)",
            self._n,
            dev / 1e3,
            host / 1e3,
            (dev + host) / 1e3,
            host / (dev + host) * 100 if (dev + host) > 0 else 0,
        )
        self._n = 0
        self._dev_us = 0.0
        self._host_us = 0.0
