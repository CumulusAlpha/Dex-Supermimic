# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Shared curriculum helpers used across multiple tasks."""

from __future__ import annotations

import torch


def get_curriculum_value(env, term_name: str, default: float = 1.0) -> float:
    """Read scalar curriculum state from manager, with a safe default."""
    manager = getattr(env, "curriculum_manager", None)
    if manager is None:
        return default

    state = getattr(manager, "_curriculum_state", {})
    if term_name not in state or state[term_name] is None:
        return default

    value = state[term_name]
    if isinstance(value, torch.Tensor):
        return value.mean().item() if value.numel() > 0 else default
    if isinstance(value, dict):
        if not value:
            return default
        first = next(iter(value.values()))
        if isinstance(first, torch.Tensor):
            return first.mean().item() if first.numel() > 0 else default
        return float(first)
    return float(value)


def disturbance_speed_limit(env, min_speed: float = 0.05, max_speed: float = 0.15) -> float:
    """Single source of difficulty for real and imagined reorientation pushes.

    Missing curriculum (e.g. fixed evaluation) means full difficulty. Read the
    authoritative state directly so checkpoint restoration takes effect at once.
    """
    state = getattr(env, "_performance_disturbance_state", None)
    scale = 1.0 if state is None else float(state["value"])
    return float(min_speed) + (float(max_speed) - float(min_speed)) * max(0.0, min(scale, 1.0))
