"""Dedicated entry points; no PPO runner or reference-repository imports."""

import json
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import tyro
from .config import DexSuperMimicCfg


def train():
    cfg = tyro.cli(DexSuperMimicCfg)
    train_config(cfg)


def train_config(cfg, folder=None):
    if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU.")
    if cfg.compile or cfg.compile_policy:
        compiler = os.environ.get("CC")
        compiler = shutil.which(compiler) if compiler else (shutil.which("gcc") or shutil.which("clang"))
        if compiler is None:
            warnings.warn(
                "No C compiler found for Triton; disabling compile and compile_policy. "
                "Training will run in eager mode. Install a C compiler (Ubuntu/Debian: "
                "sudo apt-get install build-essential), or set CC to its executable path "
                "to enable torch.compile on the next run.",
                RuntimeWarning,
                stacklevel=2,
            )
            cfg.compile = False
            cfg.compile_policy = False
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.tasks.dex_reorient.config.recipe import configure_wuji_ppo
    from .runner import DexSuperMimicRunner

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    wp.set_device(cfg.device)
    env_cfg = make_env_cfg(cfg.num_envs, cfg.control_hz, task=cfg.task, recipe_path=cfg.recipe_path, clips_root=cfg.clips_root)
    if cfg.algorithm == "ppo":
        configure_wuji_ppo(env_cfg, cfg.control_hz)
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    log_root = Path(cfg.log_dir)
    log_root /= cfg.task
    folder = Path(folder) if folder else log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runner = None
    try:
        runner = DexSuperMimicRunner(env, cfg, folder)
        try:
            runner.learn()
        except KeyboardInterrupt:
            runner.save(folder / "model_interrupted.pt")
            print(f"Saved interrupted checkpoint to {folder}", flush=True)
    finally:
        if runner is not None:
            runner.close()
        env.close()


@dataclass
class PlayConfig:
    checkpoint: str
    recipe_path: str | None = None
    clips_root: str | None = None
    num_envs: int = 1
    device: str = "cuda:0"
    steps: int = 0
    """Playback control steps. Use 0 for unlimited playback in Viser."""
    viewer: bool = False
    viser: bool = True
    """Use the interactive Viser web viewer. --steps 0 runs until interrupted."""
    wm_horizon: int = 8
    """Reseed the world model from the real state every N steps in Viser."""
    wm_opacity: float = 0.4
    """Opacity of the light-blue predicted hand and object meshes."""
    export_onnx: str | None = None


def play():
    cfg = tyro.cli(PlayConfig)
    if cfg.viewer and cfg.viser:
        raise ValueError("Use --no-viser --viewer for the native viewer, or --viser for the web viewer.")
    if cfg.steps < 0 or (cfg.steps == 0 and not cfg.viser):
        raise ValueError("--steps must be positive, or 0 with --viser.")
    if cfg.wm_horizon < 1 or not 0 < cfg.wm_opacity <= 1:
        raise ValueError("Require wm_horizon >= 1 and 0 < wm_opacity <= 1")
    if cfg.viser:
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.task import checkpoint_task, make_env_cfg
    from .networks import Actor, GaussianPPOActor

    checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    task = checkpoint_task(checkpoint["config"])
    is_ppo = checkpoint.get("format_version") == 4
    if checkpoint.get("format_version") not in (1, 2, 3, 4):
        raise ValueError("Unsupported checkpoint format")
    is_pen = task == "hand2-pen-spin"
    state = checkpoint["modules"]["actor"]
    if is_ppo:
        actor = GaussianPPOActor(state["norm._mean"].shape[-1], state["net.6.bias"].numel() // 2)
    else:
        actor = Actor(state["norm.mean"].numel())
    actor = actor.to(cfg.device)
    actor.load_state_dict(state)
    actor.eval()
    policy_actor = actor.mean if is_ppo else actor
    if cfg.export_onnx:
        path = Path(cfg.export_onnx)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            policy_actor,
            torch.zeros(1, state["norm.mean"].numel(), device=cfg.device),
            str(path),
            input_names=["policy"],
            output_names=["action"],
            dynamic_axes={"policy": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    task=task,
                    control_hz=checkpoint["config"]["control_hz"],
                    joint_names=checkpoint["joint_names"],
                    history=1 if is_pen else 3,
                    terms=["motion", "joint_pos", "joint_vel", "object_pose", "last_action"] if is_pen else [
                        "noisy_joint_angles",
                        "qpos_error",
                        "cube_pos_in_tag",
                        "cube_ori_error",
                        "action_history",
                    ],
                    action_scale=0.5,
                    ema_alpha=0.5 if is_pen else (0.3 if checkpoint["config"]["control_hz"] == 50 else 0.5),
                    warmup_time_s=0.0 if is_pen else 0.4,
                    action_type="reference_plus_ema_residual" if is_pen else "absolute_ema",
                    residual_clip=[-0.6, 0.6] if is_pen else None,
                ),
                indent=2,
            )
        )
        print(f"Exported {path} (normalization included; EMA remains in the environment/controller)")
    env_cfg = make_env_cfg(cfg.num_envs, checkpoint["config"]["control_hz"], play=True, task=task,
        recipe_path=cfg.recipe_path or checkpoint["config"].get("recipe_path"),
        clips_root=cfg.clips_root or checkpoint["config"].get("clips_root"))
    if cfg.viser and not is_pen:
        from functools import partial

        object_cfg = env_cfg.scene.entities["object"]
        object_cfg.spec_fn = partial(object_cfg.spec_fn, viser=True)
    if cfg.viser and checkpoint.get("format_version") == 3 and "wm" in checkpoint["modules"]:
        # Preserve the exact postphysics target/control on terminal transitions.
        # DexViserPlayViewer explicitly resets completed environments afterward.
        env_cfg.auto_reset = False
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    env.max_common_steps = cfg.steps
    if cfg.viser:
        from .viser import DexViserPlayViewer

        def make_reader(play_env):
            from .pen import PenSimReader
            from .sim import SimReader

            return PenSimReader(play_env) if is_pen else SimReader(play_env)

        try:
            env.reset()
            viewer = DexViserPlayViewer(
                env, policy=lambda obs: policy_actor(obs["policy"]), checkpoint=checkpoint,
                wm_horizon=cfg.wm_horizon, wm_opacity=cfg.wm_opacity,
                make_reader=make_reader,
            )
            viewer.run(num_steps=cfg.steps or None)
        finally:
            env.close()
        return
    viewer = None
    if cfg.viewer:
        from mjlab.viewer import NativeMujocoViewer

        viewer = NativeMujocoViewer(env, policy=lambda obs: policy_actor(obs["policy"]))
        viewer.setup()
    try:
        obs, _ = env.reset()
        reward_sum = 0.0
        drop_sum = 0
        success_sum = 0
        for _ in range(cfg.steps):
            with torch.no_grad():
                action = policy_actor(obs["policy"])
            obs, reward, _, _, _ = env.step(action)
            reward_sum += float(reward.mean())
            if is_pen:
                drop_sum += int(env.termination_manager.terminated.sum())
                success_sum += int(env.termination_manager.get_term("reference_end").sum())
            else:
                drop_sum += int(env.termination_manager.get_term("cage_drop").sum())
                success_sum += int(env.command_manager.get_term("reorient_command").success_achieved.sum())
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync_env_to_viewer()
        print(
            json.dumps(
                dict(
                    mean_reward_sum=reward_sum,
                    drops=drop_sum,
                    goals_reached=success_sum,
                    steps=cfg.steps,
                )
            )
        )
    finally:
        if viewer is not None:
            viewer.close()
        env.close()
