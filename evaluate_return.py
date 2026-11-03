"""Evaluate saved SuperMimic/PPO actors in the real simulator (no training).

Run with .venv/bin/python evaluate_return.py --help.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


@torch.inference_mode()
def rollout_batch(env, policy, max_steps):
    """One episode per slot; include terminal reward, exclude later resets.

    Wait for every slot, rather than preferentially collecting short episodes.
    The environment must use auto_reset=False.
    """
    obs, _ = env.reset()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    returns = torch.zeros(env.num_envs, dtype=torch.float64, device=env.device)
    lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    ended = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    for _ in range(max_steps):
        obs, reward, terminated, truncated, _ = env.step(policy(obs["policy"]))
        if not torch.isfinite(reward[active]).all():
            raise RuntimeError("Non-finite evaluation reward")
        returns += torch.where(active, reward.double(), 0.0)
        lengths += active.long()
        ended |= active & terminated
        timed_out |= active & truncated
        done = terminated | truncated
        active &= ~done
        if not active.any():
            break
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel():
            reset_obs, _ = env.reset(env_ids=ids)
            obs["policy"] = torch.where(done[:, None], reset_obs["policy"], obs["policy"])
    return [
        dict(
            slot=i,
            **{"return": float(returns[i])},
            steps=int(lengths[i]),
            survival_seconds=float(lengths[i]) * env.step_dt,
            terminated=bool(ended[i]),
            truncated=bool(timed_out[i]),
            window_limit=bool(active[i]),
        )
        for i in range(env.num_envs)
    ]


def load_actor(checkpoint, device):
    from dex_supermimic.rl.dex_supermimic.networks import Actor, GaussianPPOActor

    if checkpoint.get("format_version") not in (1, 2, 3, 4):
        raise ValueError("Unsupported checkpoint format")
    state = checkpoint["modules"]["actor"]
    is_ppo = checkpoint.get("format_version") == 4
    if is_ppo:
        actor = GaussianPPOActor(state["norm._mean"].shape[-1], state["net.6.bias"].numel() // 2)
    else:
        if "norm._mean" in state:
            actor = Actor(state["norm._mean"].shape[-1])
        else:
            from dex_supermimic.rl.dex_supermimic.common import RunningNorm

            actor = Actor(state["norm.mean"].numel())
            actor.norm = RunningNorm(state["norm.mean"].numel())
    actor.to(device).load_state_dict(state)
    actor.eval()  # Freeze observation normalization as well as network weights.
    return actor.mean if is_ppo else actor


def evaluate(args):
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.recipe import configure_wuji_ppo
    from dex_supermimic.tasks.dex_reorient.config.task import checkpoint_task, make_env_cfg

    path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(path, map_location=args.device, weights_only=False)
    train_cfg = checkpoint["config"]
    task = checkpoint_task(train_cfg)
    if task == "hand2-pen-spin":
        raise ValueError("This evaluation protocol currently supports cube reorientation only")
    policy = load_actor(checkpoint, args.device)
    rows = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg = make_env_cfg(args.num_envs, train_cfg["control_hz"], play=False, task=task)
        if checkpoint.get("format_version") == 4:
            configure_wuji_ppo(cfg, train_cfg["control_hz"])
        cfg.seed = seed
        cfg.auto_reset = False
        cfg.curriculum.clear()  # Missing curriculum means fixed full disturbance.
        cfg.episode_length_s = args.window_seconds
        env = ManagerBasedRlEnv(cfg, device=args.device)
        try:
            max_steps = math.ceil(args.window_seconds / env.step_dt)
            for batch in range(math.ceil(args.episodes / args.num_envs)):
                batch_rows = rollout_batch(env, policy, max_steps)
                count = min(args.num_envs, args.episodes - batch * args.num_envs)
                for row in batch_rows[:count]:
                    row.update(seed=seed, batch=batch, episode=batch * args.num_envs + row["slot"])
                    rows.append(row)
                print(f"seed={seed} episodes={min((batch + 1) * args.num_envs, args.episodes)}/{args.episodes}", flush=True)
        finally:
            env.close()

    values = np.asarray([row["return"] for row in rows])
    summary = dict(
        episodes=len(rows),
        return_mean=float(values.mean()),
        return_std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        return_min=float(values.min()),
        return_max=float(values.max()),
        survival_seconds_mean=float(np.mean([r["survival_seconds"] for r in rows])),
        termination_fraction=float(np.mean([r["terminated"] for r in rows])),
    )
    result = dict(
        checkpoint=str(path),
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        algorithm=train_cfg.get("algorithm", "world_model"),
        iteration=checkpoint["iteration"],
        task=task,
        protocol=dict(
            return_definition="sum of env.step rewards, undiscounted, including terminal step",
            policy="deterministic; checkpoint observation normalization frozen",
            domain_randomization=True,
            difficulty="full; curriculum disabled",
            seeds=args.seeds,
            episodes_per_seed=args.episodes,
            num_envs=args.num_envs,
            window_seconds=args.window_seconds,
            max_steps=max_steps,
            step_dt=env.step_dt,
        ),
        summary=summary,
        per_seed={str(s): float(np.mean([r["return"] for r in rows if r["seed"] == s])) for s in args.seeds},
        episodes=rows,
    )
    output = Path(args.output) if args.output else path.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="SuperMimic or PPO .pt file")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    parser.add_argument("--episodes", type=int, default=32, help="Episodes per evaluation seed")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--window-seconds", type=float, default=40.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", help="Default: checkpoint path with .evaluation.json suffix")
    args = parser.parse_args()
    if args.episodes < 1 or args.num_envs < 1 or not math.isfinite(args.window_seconds) or args.window_seconds <= 0:
        parser.error("episodes, num-envs and window-seconds must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    evaluate(args)


if __name__ == "__main__":
    main()
