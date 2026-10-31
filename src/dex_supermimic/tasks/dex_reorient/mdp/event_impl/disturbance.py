# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from dex_supermimic.tasks.dex_reorient.mdp._env_utils import resolve_env_ids
from dex_supermimic.tasks.dex_reorient.utils.curriculum import disturbance_speed_limit
from dex_supermimic.tasks.dex_reorient.mdp.runtime_state import get_reorient_runtime_state

_DEFAULT_OBJECT_CFG = SceneEntityCfg("object")


def apply_velocity_disturbance(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg = _DEFAULT_OBJECT_CFG,
    min_speed: float = 0.05,
    max_speed: float = 0.15,
    warmup_time_s: float = 3.0,
) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    if env_ids.numel() == 0:
        return
    curr_max_speed = disturbance_speed_limit(env, min_speed, max_speed)
    state = get_reorient_runtime_state(env)
    assert state.pert_force_dir is not None
    assert state.pert_velocity_cache is not None

    episode_time = env.episode_length_buf[env_ids].float() * env.step_dt
    active_mask = episode_time >= float(warmup_time_s)
    active_local = active_mask.nonzero(as_tuple=False).reshape(-1)

    state.pert_force_dir[env_ids] = 0.0
    state.pert_velocity_cache[env_ids] = 0.0

    if active_local.numel() == 0:
        return

    active_env_ids = env_ids[active_local]

    vel_dir = torch.randn((active_env_ids.numel(), 3), device=env.device)
    vel_dir = vel_dir / vel_dir.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    speed = torch.empty(active_env_ids.numel(), 1, device=env.device).uniform_(
        float(min_speed),
        max(float(min_speed), curr_max_speed),
    )
    velocity = vel_dir * speed

    state.pert_force_dir[active_env_ids] = vel_dir
    state.pert_velocity_cache[active_env_ids] = torch.cat(
        [velocity, torch.zeros_like(velocity)],
        dim=-1,
    )

    asset: Entity = env.scene[asset_cfg.name]
    root_vel = asset.data.root_link_vel_w[active_env_ids].clone()
    root_vel[:, :3] += velocity
    asset.write_root_link_velocity_to_sim(root_vel, env_ids=active_env_ids)
