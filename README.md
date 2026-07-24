# ProtoMotions + Isaac Lab UV template

This repository is a ready-to-use downstream [UV](https://docs.astral.sh/uv/)
workspace for running
[ProtoMotions](https://github.com/carlosdp/ProtoMotions/tree/codex/protomotions-pypi-uv-packaging)
with Isaac Lab. It keeps the simulator in its own environment and records the
non-default indexes required by the NVIDIA and CUDA package stacks.

The workspace currently resolves:

- ProtoMotions 3.1 from the packaging branch
- Isaac Lab 2.3.2.post1
- Isaac Sim 5.1.0.0
- PyTorch 2.7.0 and torchvision 0.22.0 from the CUDA 12.8 wheel index

## Prerequisites

- Linux x86_64 with glibc 2.35 or newer
- An NVIDIA GPU and a driver compatible with the Isaac Sim 5.1 requirements
- Python 3.11
- [UV](https://docs.astral.sh/uv/getting-started/installation/) 0.11.32 or
  newer (Git-source LFS support is required)
- Git and [Git LFS](https://git-lfs.com/)
- Enough free disk space for the multi-gigabyte Isaac Sim, CUDA, and PyTorch
  wheels

This lock is intentionally scoped to Linux x86_64. Use separate projects and
locks for ProtoMotions' other simulator stacks.

## Set up

```bash
git lfs install
git clone https://github.com/carlosdp/protomotions-isaaclab-template.git
cd protomotions-isaaclab-template
uv sync --locked
```

UV fetches ProtoMotions from the configured Git branch with LFS enabled, so the
installed wheel contains materialized runtime robot assets rather than Git LFS
pointer files. No local ProtoMotions checkout is required.

## Included downstream experiment

`experiments/smpl_amass_mlp.py` is a project-owned MLP mimic experiment for an
SMPL humanoid trained on packaged AMASS motions. It follows the environment,
reward, evaluator, actor, and critic behavior of ProtoMotions'
`examples/experiments/mimic/mlp.py`, but imports only the installed
`protomotions` package. It does not depend on a ProtoMotions checkout or copy
any package modules into this repository.

Prepare AMASS as a ProtoMotions MotionLib `.pt` file using the upstream
[AMASS data preparation guide](https://protomotions.github.io/getting_started/amass_preparation.html).
Keep the resulting data outside this repository and pass an absolute path.

After reviewing and accepting the Isaac Sim EULA as described below, run the
upstream-scale four-GPU training configuration with Isaac Lab:

```bash
uv run protomotions train \
  --robot-name smpl \
  --simulator isaaclab \
  --experiment-path experiments/smpl_amass_mlp.py \
  --experiment-name smpl_amass_isaaclab_mlp \
  --motion-file /absolute/path/to/amass_train.pt \
  --num-envs 8192 \
  --batch-size 8192 \
  --ngpu 4 \
  --headless true
```

Results and resolved configurations are written to
`results/smpl_amass_isaaclab_mlp/`. Reduce `--num-envs`, `--batch-size`, and
`--ngpu` together for smaller hosts. For example, a single-GPU functional
smoke run can start with `--num-envs 64 --batch-size 64 --ngpu 1`, but those
values are not intended for production training.

After accepting the EULA yourself and supplying a valid MotionLib file, use a
bounded one-step startup smoke test before committing a large allocation:

```bash
timeout 180s uv run protomotions train \
  --robot-name smpl \
  --simulator isaaclab \
  --experiment-path experiments/smpl_amass_mlp.py \
  --experiment-name smpl_amass_isaaclab_smoke \
  --motion-file /absolute/path/to/amass_train.pt \
  --num-envs 64 \
  --batch-size 64 \
  --ngpu 1 \
  --headless true \
  --training-max-steps 1
```

The external timeout bounds simulator startup as well as the first training
step. Remove it and choose production-scale values only after this smoke test
reaches the training loop successfully.

To build and inspect the complete downstream configuration without starting
training, add `--create-config-only` to the command. ProtoMotions still imports
Isaac Lab before Torch for this path, but it exits before constructing the
simulation application.

## Validate the installation

The following checks do not initialize Isaac Sim:

```bash
uv run protomotions info --json
uv run protomotions train --help
uv run protomotions eval --help
uv run python -c "import protomotions; print(protomotions.__version__)"
uv pip check
```

Confirm that the downstream experiment is syntax-valid:

```bash
uv run python -m py_compile experiments/smpl_amass_mlp.py
```

The first Isaac Sim import or launch downloads extensions and can take several
minutes. GPU/display and headless-launch details depend on the host.

## Isaac Sim EULA

This repository does not accept NVIDIA's Omniverse license agreement
automatically. On first import or launch, review and accept the interactive
prompt yourself.

Only after the responsible user has accepted the agreement may an unattended
environment opt in explicitly:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Do not set this variable in shared shell profiles, images, or automation that
other users may run without reviewing the agreement.

## Current package-check caveat

Isaac Sim 5.1.0.0's installed `isaacsim-core` wheel declares exact versions of
`filelock`, `fsspec`, and `networkx` that differ from the versions selected by
NVIDIA's package index. As a result, `uv pip check` currently reports these
three metadata mismatches even though the complete environment resolves and
installs:

| Package | `isaacsim-core` declares | NVIDIA index selects |
| --- | --- | --- |
| `filelock` | 3.13.1 | 3.15.4 |
| `fsspec` | 2024.6.1 | 2024.10.0 |
| `networkx` | 3.3 | 3.4.2 |

Do not force the older versions: they conflict with the package set selected
from the Isaac Sim index.

## Updating

Review changes to `pyproject.toml`, then refresh and validate the lock on a
compatible Linux x86_64 host:

```bash
uv lock --upgrade
uv sync --locked
uv pip check
```

Commit `uv.lock` whenever the tested environment changes.
