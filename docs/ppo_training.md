# Standalone PPO training

`dex-ppo-train` uses **rsl-rl 5.4.2** and mjlab's `RslRlVecEnvWrapper`
and `MjlabOnPolicyRunner`. It does not use the SuperMimic runner, world model,
or replay buffer. The rollout loop, observation normalization, timeout
bootstrapping, GAE, minibatches, optimizer, and separate actor/critic gradient
clipping execute in the installed upstream code.

The network and algorithm settings match Wuji's
[Hand 2 PPO configuration](https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/config/wuji_hand2/rsl_rl/ppo.py)
with its registered
[50 Hz overrides](https://github.com/wuji-technology/wuji-mjlab/blob/main/src/wuji_mjlab/tasks/reorient/config/wuji_hand2/__init__.py):
100 rollout steps, gamma 0.996, lambda 0.95, 4 epochs, 32 minibatches,
fixed Adam learning rate 1e-4, entropy coefficient 0.001,
value coefficient 0.5, no value clipping, policy clip 0.2, gradient norm 1.0,
and a state-dependent scalar Gaussian std clamped to [0.2, 50].
The default environment count is 8192; reduce it to fit the GPU.

**Environment scope:** this entry point uses this repository's Hand 2 50 Hz
environment, including its shared performance-based disturbance curriculum.
It does not restore Wuji's original curriculum, so this is an upstream PPO
algorithm baseline on the local task, not an assertion of identical task
conditions or learning results to the official repository.

## Train and visualize

Start a fresh run from the repository root:

```bash
uv run dex-ppo-train --num-envs 4096 --device cuda:0 --max-iterations 10000
```

Runs are saved in `logs/ppo/hand2-reorient/<timestamp>/` with a final
`model_final.pt`. TensorBoard logging is the default:

```bash
uv run tensorboard --logdir logs/ppo
```

For W&B logging, add `--logger wandb --wandb-project dex-supermimic`.
The upstream W&B writer uses `WANDB_USERNAME` for the entity when provided.
Model artifact upload is disabled; local checkpoint saves remain enabled.

The same checkpoint supports native rsl-rl resume and the existing player:

```bash
uv run dex-supermimic-play --checkpoint logs/ppo/hand2-reorient/<timestamp>/model_final.pt --viser
```

Checkpoints contain native actor, critic, optimizer and curriculum state, plus
a tensor-name-only compatibility view for playback and `evaluate_return.py`.
No training network is replaced or converted during optimization.

Resume with the same number of environments:

```bash
uv run dex-ppo-train --num-envs 4096 --checkpoint logs/ppo/hand2-reorient/<timestamp>/model_final.pt --max-iterations 5000
```

Here `--max-iterations 5000` means **5000 additional iterations**. The simulator
starts fresh episodes; normalization, optimizer, iteration, common step and
curriculum state are restored. This does not reproduce an uninterrupted
trajectory exactly. Old custom-runner PPO checkpoints are accepted for playback,
but are rejected by this native training entry point for resume.

## KL diagnostics and legacy correction

The native `LoggedPPO` calls `PPO.update()` unchanged, then records
`Loss/kl_rollout_to_updated`: the mean analytic **KL(old || updated)**,
summed over action dimensions and averaged across all rollout transitions.
The old Gaussian parameters are those saved by upstream rollout storage.
The updated policy uses the final network and normalization state after the
whole PPO update. Evaluation is chunked and does not sample actions or consume
random numbers. Fixed learning rate remains fixed; KL is diagnostic only.

The old `dex-supermimic-train --algorithm ppo` path also now stores the
rollout mean/std and computes KL against those saved values instead of
recomputing both distributions from the current policy. Its KL averages
pre-minibatch-update comparisons, so its values need not equal the native
post-update diagnostic. Legacy actor and critic gradients are now clipped
separately too. Use `dex-ppo-train` for upstream PPO training.

## Validation

```bash
.venv/bin/python -m unittest discover -s tests/dex_supermimic -v
```

Tests compare the native diagnostic wrapper against unwrapped rsl-rl PPO on
the same rollout, including terminal and timeout transitions: losses, actor
and critic parameters, Adam state, and RNG state must be identical. Additional
tests check analytic KL, the legacy KL regression, and exact deterministic
playback parity after tensor-name conversion.
