# Dex SuperMimic

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-555?logo=linux&logoColor=white)](#requirements)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

This repository reproduces [SuperMimic](https://supermimic.github.io/) on a dexterous hand, adapting its world-model-based training approach to in-hand cube reorientation with Wuji Hand 2. It is built on [mjlab](https://github.com/mujocolab/mjlab) and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).

The bundled assets cover Wuji Hand 1 and Hand 2 tasks, so installation does not require a Git submodule checkout.

## Demos

| Wuji Hand 1 · Cube Reorientation | Wuji Hand 2 · Cube Reorientation | Wuji Hand 2 · Pen Spin |
| :---: | :---: | :---: |
| <a href="docs/demos/hand1_reorient.gif"><img src="docs/demos/hand1_reorient.gif" alt="Wuji Hand 1 cube reorientation demo" width="310"></a> | <a href="docs/demos/hand2_reorient.gif"><img src="docs/demos/hand2_reorient.gif" alt="Wuji Hand 2 cube reorientation demo" width="310"></a> | <a href="docs/demos/hand2_pen_spin.gif"><img src="docs/demos/hand2_pen_spin.gif" alt="Wuji Hand 2 pen-spin demo" width="310"></a> |

## Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Tasks](#tasks)
- [Training](#training)
- [Visualization](#visualization)
- [Pretrained Checkpoint](#pretrained-checkpoint)
- [Acknowledgments](#acknowledgments)

## Overview

The project retains SuperMimic's three-phase idea:

1. collect short real-simulator rollouts into a replay buffer;
2. train a recurrent-style, multi-step world model on mixed normal and failure transitions;
3. optimize the policy and critic along imagined rollouts, periodically refreshing the buffer from the real simulator.

The implementation is task-aware: cube reorientation uses command/goal observations, an absolute EMA controller, contact and cage rewards, and a translucent world-model ghost in Viser. Pen spin tracks a packaged motion reference, uses a reference-plus-residual controller, and exposes the standard viewer without the cube-specific ghost overlay.

## Requirements

- Linux x86_64 with an NVIDIA GPU is required for training and playback. CUDA 12.8 is tested; Blackwell/RTX 50-series GPUs are supported by MuJoCo Warp.
- macOS arm64 can install and import the package, but training intentionally rejects non-CUDA devices.
- Python 3.10–3.13 is selected automatically by `uv`.
- GCC or Clang is needed for Triton / `torch.compile`. On Ubuntu/Debian, install it with `sudo apt-get update && sudo apt-get install build-essential`. If the compiler is outside `PATH`, set `CC=/absolute/path/to/gcc`. Without a compiler, training automatically disables `compile` and `compile_policy` and runs in eager mode, which may be slower.
- Triton also needs development headers matching the training Python version. For system Python 3.12 on Ubuntu/Debian, run `sudo apt-get install python3.12-dev`. A `fatal error: Python.h: No such file or directory` means these headers are missing.
- At least one CUDA-capable GPU with enough memory for the requested parallel environment count. `--num-envs 1024` is the default starting point.
- Disk space for the virtual environment, Wuji meshes, pen-spin motion catalog, checkpoints, and TensorBoard events.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Clone this repository and install the dependencies:

```bash
git clone https://github.com/CumulusAlpha/Dex-Supermimic.git
cd Dex-Supermimic
uv sync
```

## Tasks

| Task | Hand | Control rate | Data | Log directory |
|---|---|---:|---|---|
| `hand1-reorient` | Wuji Hand 1 (right) | 20 or 50 Hz | Bundled cube; no external dataset | `logs/dex_supermimic/hand1-reorient/<timestamp>` |
| `hand2-reorient` | Wuji Hand 2 (right) | 20 or 50 Hz | Bundled cube; no external dataset | `logs/dex_supermimic/<timestamp>` |
| `hand2-pen-spin` | Wuji Hand 2 (right) | 50 Hz only | Downloaded motion clips plus packaged recipe | `logs/dex_supermimic/hand2-pen-spin/<timestamp>` |

## Training

Run from the repository root. The first launch may spend extra time compiling Warp and imagination kernels.

```bash
# task: hand1-reorient | hand2-reorient | hand2-pen-spin
uv run dex-supermimic-train --task hand1-reorient
```

To train Hand 2 with W&B logging, authenticate once with `uv run wandb login`, then run:

```bash
uv run dex-supermimic-train \
  --task hand2-reorient \
  --algorithm world_model \
  --control-hz 50 \
  --num-envs 2048 \
  --max-iterations 10000 \
  --save-interval 1000 \
  --logger wandb \
  --wandb-project dex-supermimic \
  --wandb-entity cumulusalpha
```

Set `--wandb-entity` to your W&B user or team. Use `--wandb-mode offline` to save W&B logs locally without signing in. TensorBoard remains the default logger. To explicitly disable PyTorch compilation, pass both `--no-compile --no-compile-policy`.

Reorientation uses a shared performance curriculum for PPO and SuperMimic: real goal completion and survival determine disturbance strength, independently of the iteration budget. Checkpoints preserve curriculum state for continuation with `--checkpoint`. See [comparison and curriculum settings](docs/experiments/reorientation_comparison.md). Legacy time-curriculum checkpoints cannot resume under the new rules.

Before training `hand2-pen-spin`, download its upstream motion catalog:

```bash
uv run python -m dex_supermimic.tasks.pen_spin.tooling.download
```

## Visualization

Viser playback uses one environment, opens a URL in the terminal, and runs until interrupted:

```bash
uv run dex-supermimic-play --checkpoint <checkpoint.pt>
```

Inspect TensorBoard curves with:

```bash
uv run tensorboard --logdir logs/dex_supermimic
```

## Pretrained Checkpoint

Pretrained checkpoints are distributed separately as GitHub Release attachments. The `_8000` suffix indicates that these checkpoints were saved after 8,000 training iterations:

```bash
mkdir -p logs/dex_supermimic/pretrained

curl -fL --retry 3 -o logs/dex_supermimic/pretrained/model_hand1_reorient_8000.pt \
  https://github.com/CumulusAlpha/Dex-Supermimic/releases/download/pretrained-wuji-v2/model_hand1_reorient_8000.pt

curl -fL --retry 3 -o logs/dex_supermimic/pretrained/model_hand2_reorient_8000.pt \
  https://github.com/CumulusAlpha/Dex-Supermimic/releases/download/pretrained-wuji-v2/model_hand2_reorient_8000.pt

curl -fL --retry 3 -o logs/dex_supermimic/pretrained/model_hand2_pen_spin_8000.pt \
  https://github.com/CumulusAlpha/Dex-Supermimic/releases/download/pretrained-wuji-v2/model_hand2_pen_spin_8000.pt

uv run dex-supermimic-play \
  --checkpoint logs/dex_supermimic/pretrained/model_hand1_reorient_8000.pt
```

The `logs/` directory is Git-ignored, so the checkpoints remain separate from the source checkout. `model_hand2_reorient_8000.pt` is the previously released Hand 2 cube-reorientation checkpoint, renamed to identify its task and 8,000-iteration training point.

## Acknowledgments

We thank the [SuperMimic](https://supermimic.github.io/) authors for the original method.

We thank the authors and contributors of the following repositories:

1. [InstinctMJ](https://github.com/project-instinct/InstinctMJ) — the foundation of this repository and its mjlab-based simulation framework.
2. [Wuji MJLab](https://github.com/wuji-technology/wuji-mjlab) — the Wuji hand cube-reorientation and pen-spin task implementations.
3. [supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) — the SuperMimic reproduction.
