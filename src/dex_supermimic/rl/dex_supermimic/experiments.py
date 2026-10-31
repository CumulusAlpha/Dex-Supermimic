"""Multi-seed original-task training and real-simulator evaluation.

python -m dex_supermimic.rl.dex_supermimic.experiments --help
Preparation writes a reproducible manifest without starting expensive training.
--execute runs all training seeds, then evaluates each with common held-out seeds.
"""

import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
import torch
import tyro
from .config import DexSuperMimicCfg


@dataclass
class ExperimentConfig:
    output: str = "logs/dex_supermimic_experiments"
    seeds: tuple[int, ...] = (42, 43, 44)
    eval_seeds: tuple[int, ...] = (1001, 1002, 1003)
    eval_steps: int = 2000
    base_config: str | None = None
    model_checkpoint: str | None = None
    execute: bool = False
    evaluate_only: bool = False


@torch.no_grad()
def evaluate_goal_reversal(env, reader, actor, steps=32):
    """Two closed-loop goal directions from identical physics/controller memory."""
    from types import SimpleNamespace
    from .snapshot import WarpTaskSnapshot
    from .state import unpack
    from .rotations import mm, so3_exp, so3_log, mat_to_quat

    snap = WarpTaskSnapshot(SimpleNamespace(env=env, reader=reader))
    result = {}
    initial = unpack(reader.state())[3].clone()
    axis = torch.nn.functional.one_hot(torch.arange(env.num_envs, device=env.device) % 3, 3).float()
    try:
        for sign, name in ((1, "positive"), (-1, "negative")):
            snap.restore()
            cmd = env.command_manager.get_term("reorient_command")
            goal = mm(initial, so3_exp(sign * axis))
            cmd.goal_quat_w.copy_(mat_to_quat(mm(reader.palm_rot, goal)))
            for key in (
                "hold_counter",
                "goal_timer",
                "in_success_window",
                "window_timer",
                "reward_hold_counter",
                "reward_hold_counter_snapshot",
                "within_threshold",
            ):
                value = getattr(cmd, key, None)
                if value is not None:
                    value.zero_()
            obs = env.observation_manager.compute(update_history=True)["policy"]
            alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            final = initial.clone()
            for _ in range(steps):
                observation, _, terminated, truncated, _ = env.step(actor(obs))
                rotation = unpack(reader.reward_snapshot["physics"])[3]
                final = torch.where(alive[:, None, None], rotation, final)
                done = terminated | truncated
                alive &= ~done
                obs = observation["policy"]
                ids = done.nonzero().flatten()
                if ids.numel():
                    reset, _ = env.reset(env_ids=ids)
                    obs = torch.where(done[:, None], reset["policy"], obs)
            projection = (so3_log(mm(initial.transpose(-1, -2), final)) * axis).sum(-1)
            directed = projection * sign
            result[name + "/signed_axis_rotation_rad"] = projection.mean().item()
            result[name + "/toward_goal_fraction"] = (directed > 0.01).float().mean().item()
            result[name + "/survival"] = alive.float().mean().item()
        return result
    finally:
        snap.restore()


def evaluate(checkpoint, seed, steps, num_envs=64, device="cuda:0"):
    """Original training task, DR and termination; no simplified play task."""
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.task import checkpoint_task, make_env_cfg
    from .networks import Actor
    from .sim import SimReader
    from .state import unpack
    from .mdp import angle_error

    state = torch.load(checkpoint, map_location=device, weights_only=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = make_env_cfg(num_envs, state["config"]["control_hz"], play=False, task=checkpoint_task(state["config"]))
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum.clear()  # Fixed full difficulty, independent of policy performance.
    env = ManagerBasedRlEnv(cfg, device=device)
    actor = Actor().to(device)
    actor.load_state_dict(state["modules"]["actor"])
    actor.eval()
    reader = None
    try:
        obs = env.reset()[0]["policy"]
        reader = SimReader(env)
        reader.attach_capture()
        angles, gains, rewards = [], [], []
        goals, drops = 0, 0
        with torch.no_grad():
            for _ in range(steps):
                reader.refresh_frame()
                memory = reader.memory()
                goal = memory["goal"].reshape(-1, 3, 3)
                before = angle_error(unpack(reader.state())[3], goal)
                obs_dict, reward, terminated, truncated, _ = env.step(actor(obs))
                angle = angle_error(unpack(reader.reward_snapshot["physics"])[3], goal)
                angles.append(angle.cpu())
                gains.append((before - angle).cpu())
                rewards.append(reward.cpu())
                goals += int((reader.memory()["count"] - memory["count"]).sum())
                drops += int(env.termination_manager.get_term("cage_drop").sum())
                obs = obs_dict["policy"]
                done = terminated | truncated
                ids = done.nonzero().flatten()
                if ids.numel():
                    reset, _ = env.reset(env_ids=ids)
                    obs = torch.where(done[:, None], reset["policy"], obs)
        angle = torch.cat(angles)
        return dict(
            seed=seed,
            transitions=steps * num_envs,
            angle_rad_mean=angle.mean().item(),
            angle_rad_p95=angle.quantile(0.95).item(),
            target_progress_rad_mean=torch.cat(gains).mean().item(),
            goals_reached=goals,
            drops=drops,
            reward_mean=torch.cat(rewards).mean().item(),
            goal_reversal=evaluate_goal_reversal(env, reader, actor),
        )
    finally:
        if reader:
            reader.detach_capture()
        env.close()


def main():
    cfg = tyro.cli(ExperimentConfig)
    if len(set(cfg.seeds)) < 3:
        raise ValueError("Multi-seed validation requires at least three distinct training seeds")
    if not cfg.eval_seeds or cfg.eval_steps < 1:
        raise ValueError("Evaluation seeds and positive steps are required")
    base = DexSuperMimicCfg(**json.loads(Path(cfg.base_config).read_text())) if cfg.base_config else DexSuperMimicCfg()
    if base.checkpoint:
        raise ValueError("Experiments must initialize actor and critic fresh; use model_checkpoint")
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in cfg.seeds:
        folder = output / f"seed{seed}"
        train_cfg = replace(
            base,
            seed=seed,
            checkpoint=None,
            model_checkpoint=cfg.model_checkpoint or base.model_checkpoint,
            reset_model_statistics=bool(cfg.model_checkpoint or base.model_checkpoint),
            log_dir=str(folder),
            logger="none",
            wandb_mode=None,
        )
        runs.append(dict(folder=str(folder), config=train_cfg.to_dict()))
    manifest = dict(
        training_seeds=cfg.seeds,
        evaluation_seeds=cfg.eval_seeds,
        eval_steps=cfg.eval_steps,
        initialization="fresh actor/critic per training seed; model-only checkpoint optional",
        real_transitions=base.num_envs
        * (max(base.warmup_steps, base.wm_window) + int(base.max_iterations / base.utd)),
        runs=runs,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if not cfg.execute and not cfg.evaluate_only:
        print(f"Prepared {len(runs)} runs: {output / 'manifest.json'}")
        return
    results = []
    for run in runs:
        folder = Path(run["folder"])
        if not cfg.evaluate_only:
            if (folder / "model_final.pt").exists():
                raise FileExistsError(f"Refusing to overwrite completed run: {folder}")
            from .cli import train_config

            train_config(DexSuperMimicCfg(**run["config"]), folder)
        checkpoint = folder / "model_final.pt"
        evaluations = [evaluate(checkpoint, seed, cfg.eval_steps, device=base.device) for seed in cfg.eval_seeds]
        results.append(dict(train_seed=run["config"]["seed"], evaluation=evaluations))
        (output / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
