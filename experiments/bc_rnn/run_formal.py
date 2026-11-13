#!/usr/bin/env python3
"""Run the frozen BC-RNN exposure-transfer matrix on one physical GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from prepare_and_run import configure_paths, run, sha


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, required=True, choices=(0, 2))
    ap.add_argument('--worker', type=int, required=True, choices=(0, 1))
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--python', type=Path, required=True)
    ap.add_argument('--official-entry', type=Path, required=True)
    ap.add_argument('--eval-python', type=Path, required=True)
    ap.add_argument('--evaluator', type=Path, required=True)
    args = ap.parse_args()
    configure_paths(args.project_root, args.data_root, args.output_root,
                    args.python, args.official_entry)
    from prepare_and_run import RM, ROOT
    evaluator = args.evaluator.expanduser().resolve()
    eval_python = args.eval_python.expanduser().resolve()
    manifest = json.loads((ROOT / 'formal_manifest.json').read_text())
    jobs = [job for job in manifest['jobs'] if job['gpu'] == args.gpu]
    assert jobs, f'no jobs assigned to GPU {args.gpu}'
    (ROOT / 'formal_logs').mkdir(exist_ok=True)
    (ROOT / 'formal_receipts').mkdir(exist_ok=True)
    (ROOT / 'formal_evaluations').mkdir(exist_ok=True)
    for job in jobs:
        receipt_path = ROOT / 'formal_receipts' / f"{job['job_tag']}.json"
        if receipt_path.exists():
            continue
        config = Path(job['config'])
        if sha(config) != job['config_sha256']:
            raise RuntimeError(f"frozen config changed: {config}")
        if time.time() >= manifest['hard_deadline_unix']:
            raise TimeoutError('24-hour comparison deadline reached')
        started = time.time()
        seconds = run(config, args.gpu, ROOT / 'formal_logs' / f"{job['job_tag']}.log",
                      timeout=manifest['hard_deadline_unix'] - time.time())
        cfg = json.loads(config.read_text())
        run_root = Path(cfg['train']['output_dir']) / cfg['experiment']['name']
        checkpoints = sorted(run_root.glob('*/models/model_epoch_300.pth'))
        if len(checkpoints) != 1:
            raise RuntimeError(f"expected one final checkpoint at {run_root}, found {len(checkpoints)}")
        checkpoint = checkpoints[0]
        evaluation_path = ROOT / 'formal_evaluations' / f"{job['job_tag']}.json"
        evaluation_log = ROOT / 'formal_logs' / f"{job['job_tag']}_eval.log"
        eval_cmd = [str(eval_python), str(evaluator), '--checkpoint', str(checkpoint),
                    '--output', str(evaluation_path), '--seed-start', '530000',
                    '--episodes', '100', '--horizon', '500', '--gpu', str(args.gpu)]
        eval_env = {**os.environ, 'PYTHONPATH': str(RM), 'MUJOCO_GL': 'egl',
                    'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'}
        eval_started = time.time()
        with evaluation_log.open('w') as f:
            proc = subprocess.run(eval_cmd, cwd=RM, env=eval_env,
                                  stdout=f, stderr=subprocess.STDOUT, check=False,
                                  timeout=manifest['hard_deadline_unix'] - time.time())
        if proc.returncode != 0 or not evaluation_path.exists():
            raise RuntimeError(f"evaluation failed for {job['job_tag']}; see {evaluation_log}")
        evaluation = json.loads(evaluation_path.read_text())
        receipt = dict(job, status='PASS', started_unix=started,
                       completed_unix=time.time(), training_seconds=seconds,
                       evaluation_seconds=time.time() - eval_started,
                       checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                       evaluation=str(evaluation_path), success_count=evaluation['success_count'],
                       success_rate=evaluation['success_rate'],
                       config_sha256=sha(config))
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status': 'PASS', 'job': job['job_tag'],
                          'training_seconds': seconds,
                          'evaluation_seconds': receipt['evaluation_seconds'],
                          'success_rate': evaluation['success_rate']}), flush=True)


if __name__ == '__main__':
    main()
