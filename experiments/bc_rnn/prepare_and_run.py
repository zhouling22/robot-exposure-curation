#!/usr/bin/env python3
"""Prepare official robomimic BC-RNN exposure configs and run a smoke test.

The script deliberately stops after the smoke test. Formal training is started
only by run_formal.py after its measured cost check passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import os
import datetime as dt
from pathlib import Path

import h5py


REMOTE: Path | None = None
RM: Path | None = None
DATA: Path | None = None
ROOT: Path | None = None
SOURCE: dict[str, Path] = {}
PYTHON: Path | None = None
OFFICIAL_ENTRY: Path | None = None


def configure_paths(project_root: Path, data_root: Path, output_root: Path,
                    python_path: Path, official_entry: Path) -> None:
    """Configure all machine-specific paths for this local run."""
    global REMOTE, RM, DATA, ROOT, SOURCE, PYTHON, OFFICIAL_ENTRY
    REMOTE = project_root.expanduser().resolve()
    RM = REMOTE / 'robomimic'
    DATA = data_root.expanduser().resolve()
    ROOT = output_root.expanduser().resolve()
    SOURCE = {
        'random': DATA / 'can/subsets/random_k96_seed43.hdf5',
        'demo_score_official': DATA / 'can/subsets/demo_score_k96_fixed.hdf5',
    }
    PYTHON = python_path.expanduser().resolve()
    OFFICIAL_ENTRY = official_entry.expanduser().resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def make_masked_copy(source: Path, out: Path) -> list[tuple[str, int]]:
    if not out.exists():
        shutil.copy2(source, out)
    lengths: list[tuple[str, int]] = []
    with h5py.File(out, 'r') as f:
        for idx, demo in enumerate(sorted(f['data'].keys(), key=lambda x: int(x.split('_')[-1]))):
            n = int(f['data'][demo].attrs['num_samples'])
            lengths.append((demo, n))
        expected = {f'one_{idx:03d}': demo for idx, (demo, _n) in enumerate(lengths)}
        current = {}
        if 'mask' in f:
            for key in expected:
                if key in f['mask']:
                    values = [x.decode('utf-8') for x in f['mask'][key][:]]
                    current[key] = values[0] if len(values) == 1 else None
    if current != expected:
        with h5py.File(out, 'r+') as f:
            mask = f.require_group('mask')
            for key in list(mask.keys()):
                if key.startswith('one_'):
                    del mask[key]
            for key, demo in expected.items():
                mask.create_dataset(key, data=[demo.encode('utf-8')])
    return lengths


def config_for(selection: str, exposure: str, seed: int, steps: int, out_root: Path) -> tuple[Path, dict]:
    if ROOT is None or RM is None:
        raise RuntimeError('call configure_paths() before preparing a run')
    source = SOURCE[selection]
    masked = ROOT / 'datasets' / f'{selection}.hdf5'
    lengths = make_masked_copy(source, masked)
    data = []
    for idx, (_demo, n) in enumerate(lengths):
        weight = 1.0 if exposure == 'wu' else 1.0 / float(n)
        data.append({'path': str(masked), 'filter_key': f'one_{idx:03d}', 'weight': weight, 'do_eval': False})
    template = json.loads((RM / 'robomimic/exps/templates/bc.json').read_text())
    template['algo_name'] = 'bc'
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    exp_name = f'{selection}_{exposure}_s{seed}_{steps}_{run_id}'
    template['experiment'].update({
        'name': exp_name,
        'validate': False,
        'render_video': False,
        'rollout': {'enabled': False, 'n': 0, 'horizon': 400, 'rate': 50,
                    'warmstart': 0, 'terminate_on_success': True},
        'save': {'enabled': True, 'every_n_seconds': None, 'every_n_epochs': steps // 100,
                 'epochs': [steps // 100], 'on_best_validation': False,
                 'on_best_rollout_return': False, 'on_best_rollout_success_rate': False},
    })
    template['train'].update({
        'data': data,
        'output_dir': str(out_root / 'models'),
        'num_data_workers': 0,
        'hdf5_cache_mode': 'low_dim',
        'hdf5_use_swmr': True,
        'hdf5_load_next_obs': False,
        'hdf5_normalize_obs': False,
        'hdf5_filter_key': None,
        'hdf5_validation_filter_key': None,
        'seq_length': 10,
        'pad_seq_length': True,
        'frame_stack': 1,
        'pad_frame_stack': True,
        'dataset_keys': ['actions'],
        'action_config': {'actions': {'normalization': None}},
        'batch_size': 256,
        'num_epochs': steps // 100,
        'seed': seed,
    })
    template['experiment']['epoch_every_n_steps'] = 100
    template['algo']['rnn'].update({'enabled': True, 'horizon': 10, 'hidden_dim': 400,
                                    'rnn_type': 'LSTM', 'num_layers': 2, 'open_loop': False})
    template['algo']['gmm'].update({'enabled': True, 'num_modes': 5, 'min_std': 0.0001,
                                    'std_activation': 'softplus', 'low_noise_eval': True})
    template['algo']['actor_layer_dims'] = []
    template['algo']['optim_params']['policy']['learning_rate']['scheduler_type'] = 'constant_with_warmup'
    path = ROOT / 'configs' / f'{exp_name}.json'
    path.write_text(json.dumps(template, indent=2))
    return path, {'selection': selection, 'exposure': exposure, 'seed': seed,
                  'steps': steps, 'source': str(source), 'source_sha256': sha(source),
                  'masked': str(masked), 'masked_sha256': sha(masked),
                  'experiment_name': exp_name,
                  'demo_count': len(lengths), 'sequence_counts': [n for _, n in lengths],
                  'wu_total_weight': float(sum(n for _, n in lengths)),
                  'eu_total_weight': float(len(lengths)),
                  'expected_episode_probabilities': {
                      'wu': [float(n / sum(x for _, x in lengths)) for _, n in lengths],
                      'eu': [1.0 / len(lengths)] * len(lengths)},
                  'sampling_semantics': 'Official MetaDataset repeats each dataset weight for every sequence window; WU mass is proportional to n_i, EU mass is equal per demonstration.'}


def run(config: Path, gpu: int, log: Path, timeout: float | None = None) -> float:
    if PYTHON is None or RM is None or OFFICIAL_ENTRY is None:
        raise RuntimeError('call configure_paths() before running a job')
    started = time.time()
    cmd = [str(PYTHON), str(OFFICIAL_ENTRY), '--robomimic-train',
           str(RM / 'robomimic/scripts/train.py'), '--config', str(config)]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONPATH': str(RM),
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    with log.open('w') as f:
        proc = subprocess.run(cmd, cwd=RM, env={**__import__('os').environ, **env},
                              stdout=f, stderr=subprocess.STDOUT, check=False, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f'robomimic smoke failed: {proc.returncode}; see {log}')
    cfg = json.loads(config.read_text())
    exp_logs = Path(cfg['train']['output_dir']) / cfg['experiment']['name']
    runs = sorted(exp_logs.glob('*/logs/log.txt'))
    if not runs:
        raise RuntimeError(f'no robomimic training log created for {config}')
    train_log = runs[-1]
    log_text = train_log.read_text(errors='replace')
    if 'run failed with error:' in log_text or 'finished run successfully!' not in log_text:
        raise RuntimeError(f'robomimic reported training failure; see {train_log}')
    expected_epochs = int(cfg['train']['num_epochs'])
    epoch_marker = f'Train Epoch {expected_epochs}'
    if epoch_marker not in log_text:
        raise RuntimeError(f'expected {epoch_marker!r} not found; see {train_log}')
    checkpoints = list(train_log.parent.parent.glob('models/*.pth'))
    if not checkpoints:
        checkpoints = list(train_log.parent.parent.glob('*.pth'))
    if not checkpoints:
        raise RuntimeError(f'no checkpoint found for completed run; see {train_log.parent.parent}')
    return time.time() - started


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--steps', type=int, default=500)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--tag', default='smoke')
    ap.add_argument('--project-root', type=Path, required=True,
                    help='Directory containing the robomimic checkout')
    ap.add_argument('--data-root', type=Path, required=True,
                    help='Directory containing can/subsets/*.hdf5')
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True,
                    help='Python executable with robomimic dependencies')
    ap.add_argument('--official-entry', type=Path, required=True,
                    help='Path to experiments/bc_rnn/official_bc_entry.py')
    args = ap.parse_args()
    configure_paths(args.project_root, args.data_root, args.output_root,
                    args.python, args.official_entry)
    for p in [ROOT, ROOT / 'configs', ROOT / 'logs', ROOT / 'receipts', ROOT / 'models', ROOT / 'datasets']:
        p.mkdir(parents=True, exist_ok=True)
    rows = []
    for selection, exposure in [('random', 'wu'), ('random', 'eu')]:
        config, receipt = config_for(selection, exposure, seed=args.seed, steps=args.steps, out_root=ROOT)
        seconds = run(config, args.gpu, ROOT / 'logs' / f'{selection}_{exposure}_{args.tag}.log')
        receipt.update({'config': str(config), 'gpu': args.gpu, 'seconds': seconds, 'status': 'PASS'})
        (ROOT / 'receipts' / f'{selection}_{exposure}_{args.tag}.json').write_text(json.dumps(receipt, indent=2))
        rows.append(receipt)
    (ROOT / f'{args.tag}_completed.json').write_text(json.dumps({'status': 'PASS', 'rows': rows}, indent=2))
    print(json.dumps({'status': 'PASS', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
