#!/usr/bin/env python3
"""Freeze configs and admission receipts for the BC-RNN comparison batch."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import time
from pathlib import Path

from prepare_and_run import ROOT, SOURCE, config_for, sha


def main() -> None:
    assert not (ROOT / 'formal_manifest.json').exists(), 'formal batch already frozen'
    for name in ['formal_logs', 'formal_receipts', 'formal_evaluations']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    jobs = []
    idx = 0
    for selection in ('random', 'demo_score_official'):
        for seed in (0, 1, 2):
            for exposure in ('wu', 'eu'):
                gpu = 0 if exposure == 'wu' else 2
                config, receipt = config_for(selection, exposure, seed, 30000, ROOT)
                jobs.append(dict(
                    index=idx, job_tag=f'formal_{selection}_{exposure}_s{seed}',
                    selection=selection, exposure=exposure, seed=seed,
                    gpu=gpu, config=str(config), config_sha256=sha(config),
                    source_sha256=sha(SOURCE[selection]),
                    expected_demo_count=receipt['demo_count'],
                    sequence_count_sum=sum(receipt['sequence_counts']),
                    effective_windows_equal_frame_counts=True,
                    expected_episode_probabilities=receipt['expected_episode_probabilities']))
                idx += 1
    started = time.time()
    parent_bytes = int(subprocess.check_output(
        ['du', '-sb', str(ROOT.parent)], text=True).split()[0])
    budget_bytes = 200_000_000_000
    planned_output_reserve = 2_000_000_000
    if parent_bytes + planned_output_reserve > budget_bytes:
        raise RuntimeError(f'200GB project budget admission failed: {parent_bytes} + {planned_output_reserve}')
    stamp = dt.datetime.fromtimestamp(started, dt.timezone(dt.timedelta(hours=8))).isoformat()
    training_seconds_per_gpu = 6 * 6 * 192.14
    serial_eval_seconds = 12 * 100 * ((40.47 + 42.49) / 6)
    conservative_parallel_eval_seconds = serial_eval_seconds * 2.0 / 2
    prep_and_audit_reserve = 1800
    conservative_batch_seconds = (training_seconds_per_gpu +
                                  conservative_parallel_eval_seconds +
                                  prep_and_audit_reserve)
    margin_fraction = 0.25
    manifest = dict(
        status='FROZEN', created=stamp, created_unix=started,
        hard_deadline_unix=started + 86400, hard_deadline_hours=24,
        task='Can-MH low-dimensional fixed-budget robomimic BC-RNN exposure transfer',
        steps=30000, batch_size=256, updates_per_epoch=100,
        seeds=[0, 1, 2], evaluation=dict(seed_start=530000, episodes=100, horizon=500),
        jobs=jobs,
        primary_estimands=[
            'paired EU-WU success difference within each selection and training seed',
            'sampler-by-selector interaction: (Demo-SCORE-Random)_EU - (Demo-SCORE-Random)_WU'],
        inference='Training seed is the replication unit (n=3); evaluation-state resampling is secondary and does not increase training n.',
        stopping_rule='Run each of the 12 frozen jobs once. No outcome-driven seed additions, checkpoint selection, or hyperparameter changes.',
        measured_admission=dict(
            precheck_updates=5000, per_model_training_seconds=[186.30, 189.30, 192.14, 187.78],
            extrapolated_30000_per_model_seconds=192.14 * 6,
            gpu_workers=2, model_count=12,
            training_wall_seconds_per_gpu=training_seconds_per_gpu,
            rollout_smoke_episodes=6, rollout_smoke_wall_seconds=[40.47, 42.49],
            extrapolated_serial_eval_wall_seconds=serial_eval_seconds,
            conservative_parallel_eval_slowdown=2.0,
            conservative_parallel_eval_wall_seconds=conservative_parallel_eval_seconds,
            preparation_and_audit_reserve_seconds=prep_and_audit_reserve,
            conservative_batch_wall_seconds=conservative_batch_seconds,
            margin_fraction=margin_fraction,
            conservative_batch_with_margin_seconds=conservative_batch_seconds * (1 + margin_fraction),
            admitted_inside_24h=conservative_batch_seconds * (1 + margin_fraction) < 86400),
        storage=dict(root=str(ROOT), current_size_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                     parent=str(ROOT.parent), parent_size_before_bytes=parent_bytes,
                     planned_output_reserve_bytes=planned_output_reserve,
                     max_project_budget_bytes=budget_bytes),
        claim_boundary='Fixed-budget BC-RNN transfer using the upstream robomimic trainer; not an exact full-schedule reproduction.')
    path = ROOT / 'formal_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'status': 'FROZEN', 'jobs': len(jobs), 'deadline_unix': started + 86400,
                      'estimated_conservative_hours_with_margin': manifest['measured_admission']['conservative_batch_with_margin_seconds']/3600,
                      'manifest': str(path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
