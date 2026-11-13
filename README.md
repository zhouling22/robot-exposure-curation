# Robot Exposure Curation

This repository contains the source-only experiment adapters used to study
training exposure in fixed-budget robot imitation learning. The code makes the
sampling contract explicit: window-uniform (WU) exposure samples legal
training windows uniformly, while episode-uniform (EU) exposure gives every
demonstration equal total probability. Inverse-duration exposure is included
as a diagnostic control.

The authors are Ling Zhou (first author) and Chun Zhang (corresponding author),
Tsinghua University.

## Scope and reproducibility boundary

The repository does not include datasets, checkpoints, rollout logs, caches,
or private machine configuration. You must install the upstream Diffusion
Policy, robomimic, ACT, RoboSuite, and PyTorch dependencies and provide local
paths to the data and source checkouts. The adapters correspond to fixed-budget
adaptations used in the accompanying study; they are not full-schedule
reproductions of all upstream papers.

Training seeds are the replication unit. Rollout episodes estimate the success
rate of a fixed checkpoint and do not replace independent training repeats.
Keep selection membership, optimizer configuration, evaluation initial states,
and checkpoint rules fixed before comparing WU and EU.

## Install the reusable sampling API

```text
python -m pip install -e .
```

```python
from robot_exposure import (
    window_uniform_probabilities,
    episode_uniform_probabilities,
    inverse_duration_probabilities,
)

counts = [120, 80, 40]
print(window_uniform_probabilities(counts))
print(episode_uniform_probabilities(counts))
print(inverse_duration_probabilities(counts))
```

## Experiment adapters

- `experiments/diffusion_policy/episode_weights.py` computes per-episode
  losses and inverse-loss diagnostics for a local Diffusion Policy checkpoint.
- `experiments/bc_rnn/` prepares and audits the robomimic BC-RNN WU/EU
  comparison. Use `--project-root`, `--robomimic-root`, `--data-root`, and
  `--python` to point at local installations; the formal runner refuses to
  proceed past its configured deadline.
- `experiments/act/` contains the ACT Can image adapter, evaluation gate, and
  expert replay check. The ACT source, cache root, dataset asset, and output
  root are all explicit command-line arguments.

The upstream source trees are intentionally not vendored. Each adapter records
the source and data paths in local receipts so an audit can verify which
artifacts were used without publishing those artifacts.

## Minimal smoke checks

```text
python -m compileall src experiments
python -c "from robot_exposure import episode_probabilities; print(episode_probabilities([1, 2, 4], 0))"
```

GPU training requires the corresponding upstream repositories, a compatible
CUDA/PyTorch stack, and local datasets. Run a small smoke test before any
formal batch and include preparation, training, evaluation, and margin in the
same wall-clock budget.

## Citation and licenses

Please cite the accompanying paper and the upstream projects listed in
`third_party/README.md`. This repository is released under the MIT License;
third-party licenses remain in `third_party/`.
