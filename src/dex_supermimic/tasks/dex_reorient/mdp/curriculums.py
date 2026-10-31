# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch

from dex_supermimic.tasks.dex_reorient.mdp._env_utils import resolve_env_ids
from dex_supermimic.tasks.dex_reorient.utils.curriculum import get_curriculum_value  # noqa: F401 -- re-export

__all__ = ["performance_disturbance_curriculum", "get_curriculum_value", "resolve_env_ids"]


def performance_disturbance_curriculum(
    env,
    env_ids,
    command_name: str = "reorient_command",
    target_survival_s: float = 40.0,
    promote_goals_per_minute: float = 4.5,
    demote_goals_per_minute: float = 1.5,
    promote_survival: float = 0.8,
    demote_survival: float = 0.5,
    adjustment: float = 0.05,
    initial_scale: float = 0.05,
) -> dict[str, float]:
    """Use one completed episode per environment in each performance window.

    Called before reset clears episode metrics, for both automatic PPO resets and
    manual WM resets. Zero-length startup resets contribute no evidence. Repeated
    short episodes from one environment cannot outweigh slower environments.
    No imagined outcomes, iteration counts, or planned training budgets are used.
    """
    state = getattr(env, "_performance_disturbance_state", None)
    if state is None:
        state = dict(
            value=float(initial_scale),
            seen=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            goals=0.0, seconds=0.0, survived=0.0, windows=0,
            goals_per_minute=0.0, survival_fraction=0.0,
        )
        env._performance_disturbance_state = state
    ids = resolve_env_ids(env, env_ids)
    ids = ids[(env.episode_length_buf[ids] > 0) & ~state["seen"][ids]]
    if ids.numel():
        seconds = env.episode_length_buf[ids].float() * env.step_dt
        command = env.command_manager.get_term(command_name)
        state["goals"] += float(command.metrics["goal_reach_count"][ids].sum().item())
        state["seconds"] += float(seconds.sum().item())
        terminated = env.termination_manager.terminated[ids]
        state["survived"] += float(((seconds >= target_survival_s) & ~terminated).sum().item())
        state["seen"][ids] = True
        if bool(state["seen"].all()):
            rate = 60.0 * state["goals"] / max(state["seconds"], 1e-6)
            survival = state["survived"] / env.num_envs
            if rate >= promote_goals_per_minute and survival >= promote_survival:
                state["value"] = min(1.0, state["value"] + adjustment)
            elif rate < demote_goals_per_minute or survival < demote_survival:
                state["value"] = max(0.0, state["value"] - adjustment)
            state.update(goals_per_minute=rate, survival_fraction=survival,
                         goals=0.0, seconds=0.0, survived=0.0, windows=state["windows"] + 1)
            state["seen"].zero_()
    return dict(value=state["value"], goals_per_minute=state["goals_per_minute"],
                survival_fraction=state["survival_fraction"], windows=state["windows"],
                window_episodes=int(state["seen"].sum().item()))
