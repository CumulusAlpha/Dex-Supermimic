"""Standalone PPO using mjlab/rsl-rl's unmodified learning loop and optimizer.

The Hand 2 50 Hz algorithm recipe follows wuji-mjlab's registered task:
https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/config/wuji_hand2/__init__.py
The environment (including its performance curriculum) remains this project's.
"""

import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from mjlab.rl import (
    MjlabOnPolicyRunner,
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.algorithms import PPO


@dataclass
class PPOTrainConfig:
    num_envs: int = 8192
    device: str = "cuda:0"
    seed: int = 42
    max_iterations: int = 5000
    """Learning iterations to run; on resume, additional iterations."""
    save_interval: int = 50
    log_dir: str = "logs/ppo/hand2-reorient"
    checkpoint: str | None = None
    """Resume a native rsl-rl checkpoint from this entry point."""
    logger: Literal["tensorboard", "wandb"] = "tensorboard"
    wandb_project: str = "dex-supermimic"


def runner_config(cfg: PPOTrainConfig) -> dict:
    """WujiHand2_Reorient_50Hz's PPO recipe, with local logging options."""
    config = asdict(RslRlOnPolicyRunnerCfg(
        seed=cfg.seed,
        obs_groups={"actor": ("policy",), "critic": ("critic",)},
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
            distribution_cfg={
                "class_name": "rsl_rl.modules:HeteroscedasticGaussianDistribution",
                "init_std": 0.5, "std_range": (0.2, 50.0), "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 512, 256, 128), activation="elu", obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=0.5, use_clipped_value_loss=False, clip_param=0.2,
            entropy_coef=0.001, num_learning_epochs=4, num_mini_batches=32,
            learning_rate=1.0e-4, schedule="fixed", gamma=0.996, lam=0.95,
            max_grad_norm=1.0,
            class_name="dex_supermimic.rl.ppo:LoggedPPO",
        ),
        num_steps_per_env=100, max_iterations=cfg.max_iterations,
        save_interval=cfg.save_interval, experiment_name="wuji_reorient_hand2",
        run_name="Reorient_Hand2_50Hz", logger=cfg.logger,
        wandb_project=cfg.wandb_project, upload_model=False,
    ))
    if cfg.logger == "wandb":
        config["logger"] = {
            "class_name": "WandbLogWriter", "project_name": cfg.wandb_project,
        }
    return config


class LoggedPPO(PPO):
    """Add an exact post-update KL diagnostic without changing PPO updates/RNG.

    All optimization is performed by super().update(). RSL's storage.clear()
    resets the write index while leaving the rollout tensors available here.
    """

    def update(self):
        losses = super().update()
        # This entry point only configures the upstream feedforward Gaussian MLP.
        # Evaluate deterministically so logging does not consume sampling RNG.
        with torch.inference_mode():
            observations = self.storage.observations.flatten(0, 1)
            old_params = tuple(p.flatten(0, 1) for p in self.storage.distribution_params)
            total = torch.zeros((), device=self.device, dtype=torch.float64)
            count = observations.batch_size[0]
            for start in range(0, count, 8192):
                chunk = slice(start, start + 8192)
                output = self.actor.mlp(self.actor.get_latent(observations[chunk]))
                self.actor.distribution.update(output)
                kl = self.actor.get_kl_divergence(
                    tuple(p[chunk] for p in old_params), self.actor.output_distribution_params,
                )
                total += kl.double().sum()
            losses["kl_rollout_to_updated"] = (total / count).item()
        return losses


def playback_actor_state(actor_state):
    """Translate only tensor names for the existing GaussianPPOActor player.

    Keep the native actor/critic/optimizer state too, so resume uses rsl-rl.
    The architecture, normalization and Gaussian mean are identical.
    """
    state = {}
    for name, value in actor_state.items():
        if name.startswith("mlp."):
            name = "net." + name.removeprefix("mlp.")
        elif name.startswith("obs_normalizer."):
            name = "norm." + name.removeprefix("obs_normalizer.")
        else:
            raise ValueError(f"Unsupported native PPO actor tensor: {name}")
        state[name] = value
    return state


class PPORunner(MjlabOnPolicyRunner):
    """Only extend checkpoint I/O; learn(), GAE and optimization stay upstream."""

    def save(self, path, infos=None):
        env = self.env.unwrapped
        infos = {
            **(infos or {}),
            "env_state": {"common_step_counter": env.common_step_counter},
            "performance_curriculum": getattr(env, "_performance_disturbance_state", None),
            "next_iteration": self.current_learning_iteration + 1,
        }
        state = self.alg.save()
        state.update(
            iter=self.current_learning_iteration, infos=infos,
            backend="rsl_rl", rsl_rl_version=version("rsl-rl-lib"),
            runner_config=self.saved_runner_config,
            num_envs=self.env.num_envs,
            # Compatibility view consumed by dex-supermimic-play/evaluate_return.
            format_version=4, iteration=infos["next_iteration"],
            config={"task": "hand2-reorient", "control_hz": 50, "algorithm": "ppo"},
            joint_names=list(env.scene["robot"].joint_names),
            modules={"actor": playback_actor_state(state["actor_state_dict"])},
        )
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def load(self, path, **kwargs):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("backend") != "rsl_rl":
            raise ValueError("Resume requires a checkpoint from dex-ppo-train; start a fresh baseline for old PPO weights")
        if state["num_envs"] != self.env.num_envs:
            raise ValueError("Resume requires the original --num-envs for the curriculum")
        # Prevent silent changes in the recipe when resuming an experiment.
        for key in ("actor", "critic", "algorithm", "obs_groups", "num_steps_per_env"):
            if state["runner_config"][key] != self.saved_runner_config[key]:
                raise ValueError(f"PPO resume configuration mismatch: {key}")
        infos = super().load(path, **kwargs)
        self.current_learning_iteration = infos["next_iteration"]
        curriculum = infos.get("performance_curriculum")
        if curriculum is not None:
            curriculum["seen"] = curriculum["seen"].to(self.device)
            if curriculum["seen"].numel() != self.env.num_envs:
                raise ValueError("Resume requires the original --num-envs for the curriculum")
            self.env.unwrapped._performance_disturbance_state = curriculum
        return infos


def train_config(cfg: PPOTrainConfig):
    if min(cfg.num_envs, cfg.max_iterations, cfg.save_interval) <= 0:
        raise ValueError("num_envs, max_iterations and save_interval must be positive")
    if not cfg.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU")
    import warp as wp
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.task import make_env_cfg
    from dex_supermimic.tasks.dex_reorient.config.recipe import configure_wuji_ppo

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    wp.set_device(cfg.device)
    env_cfg = configure_wuji_ppo(make_env_cfg(cfg.num_envs, 50, task="hand2-reorient"), 50)
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = True
    config = runner_config(cfg)
    folder = Path(cfg.log_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    (folder / "runner_config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    runner = None
    logging_stopped = False
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=config["clip_actions"])
        saved_config = copy.deepcopy(config)  # upstream consumes class_name keys
        runner = PPORunner(wrapped, config, str(folder), cfg.device)
        runner.saved_runner_config = saved_config
        if cfg.checkpoint:
            runner.load(cfg.checkpoint, map_location=cfg.device)
        env.max_common_steps = (runner.current_learning_iteration + cfg.max_iterations) * config["num_steps_per_env"]
        print(f"PPO backend: rsl-rl {version('rsl-rl-lib')}; output: {folder}", flush=True)
        runner.learn(cfg.max_iterations)
        logging_stopped = True  # upstream learn() stops external logging
        runner.save(folder / "model_final.pt")
    except KeyboardInterrupt:
        if runner is not None:
            runner.save(folder / "model_interrupted.pt")
        raise
    finally:
        if runner is not None and runner.logger.writer is not None:
            if not logging_stopped:
                runner.logger.stop_logging_writer()
            # tensorboard.close() also flushes the final scalar events.
            runner.logger.writer.close()
        env.close()


def train():
    train_config(tyro.cli(PPOTrainConfig))


if __name__ == "__main__":
    train()
