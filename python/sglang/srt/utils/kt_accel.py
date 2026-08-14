# SPDX-License-Identifier: Apache-2.0
"""CUDA / Ascend NPU helpers for KT heterogeneous MoE paths."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Optional

import torch


def _accelerator(device: torch.device) -> Any:
    if device.type == "cuda":
        return torch.cuda
    if device.type == "npu":
        import torch_npu  # noqa: F401

        return torch.npu
    raise TypeError(
        "KT heterogeneous MoE requires a CUDA or NPU device, " f"got {device.type!r}"
    )


def kt_device_synchronize(device: Optional[torch.device] = None) -> None:
    if device is not None:
        _accelerator(device).synchronize(device)
        return
    if torch.cuda.is_available() and getattr(torch.version, "cuda", None):
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def kt_current_stream(device: torch.device) -> Any:
    return _accelerator(device).current_stream(device)


def kt_current_stream_handle(device: torch.device) -> int:
    stream = kt_current_stream(device)
    if device.type == "npu":
        return int(stream.npu_stream)
    return int(stream.cuda_stream)


def kt_new_stream(device: torch.device) -> Any:
    """Create a fresh accelerator stream bound to ``device``."""
    return _accelerator(device).Stream(device=device)


def kt_new_event(device: torch.device) -> Any:
    """Create a fresh accelerator event usable for cross-stream fork/join."""
    return _accelerator(device).Event()


def kt_stream_context(stream: Any, device: torch.device) -> AbstractContextManager:
    """Make ``stream`` the current stream for the duration of the block.

    Inside the block ``kt_current_stream`` / ``kt_current_stream_handle`` resolve
    to ``stream``, so ops enqueued by callees land on it without threading the
    stream through their signatures.
    """
    return _accelerator(device).stream(stream)
