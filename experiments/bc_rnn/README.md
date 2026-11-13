# Can-MH BC-RNN Exposure Transfer

This is a prospective policy-class transfer audit on the Can-MH low-dimensional
task. It compares fixed Random-96 and Demo-SCORE-96 memberships under window-
uniform (WU) and episode-uniform (EU) training exposure. Training uses the
upstream robomimic BC-RNN model, official `MetaDataset`, official `train.py`,
and the Can-MH environment metadata. The only intended WU/EU change is the
per-demonstration sampling weight: WU uses 1; EU uses 1 divided by the number
of legal sequence windows in that demonstration.

The formal matrix is two fixed selections x two exposure arms x three training
seeds. Each model receives 30,000 optimizer updates (100 updates per epoch,
batch size 256), followed by 100 common seeded initial states and a 500-step
horizon. The seed bank is 530000--530099. Training is a fixed-budget adaptation,
not an exact full-schedule reproduction of a published robomimic benchmark.

## Current engineering gates

- 500-update WU and EU smoke runs pass and save checkpoints.
- Corrected evaluation runs under RoboSuite 1.4.1, the version recorded by the
  Can-MH dataset. WU/EU smoke state and observation hashes match exactly.
- 5k training throughput measured on GPUs 0 and 2: 186.30--192.14 seconds per
  model. Evaluation smoke: 40.47--42.49 seconds per 3 episodes.
- Full batch admission uses two GPUs, includes 12 trainings and 1,200 rollouts,
  applies a conservative parallel-evaluation slowdown and reserves runtime
  margin within a 24-hour hard deadline.
- GPU 1 is occupied by another user and is never selected. The stopped
  `official_v3` queues are not restarted.

## Primary analysis

For each training seed, report paired EU-WU differences within each selection.
The primary interaction is
`(Demo-SCORE - Random)_EU - (Demo-SCORE - Random)_WU` at 30k updates. Training
seed is the replication unit (`n=3`); the 100 rollout states estimate each
trained model's success rate and do not count as independent training runs.
No outcome-driven extra seeds, checkpoint selection, or hyperparameter tuning.

## Nearby papers' validation patterns

- [CUPID](https://arxiv.org/abs/2506.19121) uses Diffusion Policy on RoboMimic,
  averages over 3 training seeds, and reports simulation performance from 50
  rollouts at each of the last 10 checkpoints (500 rollouts per seed). Its
  hardware experiments also include official pi0. This supports repeated
  training and substantial rollout evaluation, not an ACT-only requirement.
- [QoQ](https://arxiv.org/abs/2603.09056) reports 5 simulation runs and 3 real-
  robot runs, with a RoboMimic Diffusion Policy setup. The paper also evaluates
  multiple real robot tasks.
- [Demo-SCORE](https://arxiv.org/abs/2503.03707) evaluates Diffusion Policy on
  RoboSuite and ACT in ALOHA settings. Its rollout-based curation method samples
  policy checkpoints and cross-validates classifiers across rollout
  distributions; it is not a direct claim that ACT is the universal transfer
  architecture.
- [DataMIL](https://arxiv.org/abs/2505.09603) emphasizes task and data-domain
  breadth: MLP-style policies for MetaWorld, Octo for LIBERO/OXE, and multiple
  training seeds where specified (3 in MetaWorld and 5 for the cited OXE
  fine-tuning results). It is not a Can-specific ACT validation protocol.

These precedent patterns motivate a native recurrent BC baseline as a cleaner
second policy class for the same low-dimensional Can data. The ACT result remains
an audited boundary result and is not treated as proof of an ACT defect.

## Entrypoints

- `prepare_and_run.py`: verified smoke and throughput runner.
- `evaluate_bc.py`: official robomimic checkpoint loading and common-state
  evaluation under the pinned RoboSuite environment.
- `prepare_formal.py`: creates all frozen 30k configs and the batch manifest.
- `run_formal.py`: persistent per-GPU worker; each model is trained and then
  evaluated before that worker advances.
- `audit_formal.py`: after all jobs finish, verifies checkpoints/configs/common
  evaluation states and emits seed-level contrasts plus a compact Markdown report.
