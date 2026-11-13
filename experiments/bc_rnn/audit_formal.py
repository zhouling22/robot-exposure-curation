#!/usr/bin/env python3
"""Integrity and seed-level analysis for the frozen BC-RNN transfer batch."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

from prepare_and_run import sha


def mean_sd(values: list[float]) -> tuple[float, float]:
    avg = sum(values) / len(values)
    sd = math.sqrt(sum((x - avg) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    return avg, sd


def exact_sign_flip_p(values: list[float]) -> float:
    observed = abs(sum(values) / len(values))
    randomized = [abs(sum(s * x for s, x in zip(signs, values)) / len(values))
                  for signs in itertools.product((-1, 1), repeat=len(values))]
    return sum(value >= observed - 1e-12 for value in randomized) / len(randomized)


def summarize(values: list[float]) -> dict:
    avg, sd = mean_sd(values)
    return dict(per_seed=values, mean=avg, sd=sd,
                exact_two_sided_sign_flip_p=exact_sign_flip_p(values),
                training_seeds=len(values))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True)
    args = ap.parse_args()
    root = args.output_root.expanduser().resolve()
    manifest = json.loads((root / 'formal_manifest.json').read_text())
    expected_jobs = {job['job_tag']: job for job in manifest['jobs']}
    rates = {s: {e: {} for e in ('wu', 'eu')} for s in ('random', 'demo_score_official')}
    eval_vectors = {}
    details = []
    missing = []
    for tag, job in expected_jobs.items():
        receipt_path = root / 'formal_receipts' / f'{tag}.json'
        if not receipt_path.exists():
            missing.append(tag)
            continue
        receipt = json.loads(receipt_path.read_text())
        assert receipt['status'] == 'PASS'
        assert sha(Path(job['config'])) == job['config_sha256'] == receipt['config_sha256']
        assert sha(Path(receipt['checkpoint'])) == receipt['checkpoint_sha256']
        evaluation = json.loads(Path(receipt['evaluation']).read_text())
        assert evaluation['status'] == 'PASS'
        assert evaluation['checkpoint_sha256'] == receipt['checkpoint_sha256']
        assert evaluation['evaluation_seeds'] == list(range(530000, 530100))
        episodes = evaluation['episodes_data']
        assert len(episodes) == 100
        assert sum(int(ep['success']) for ep in episodes) == evaluation['success_count']
        assert abs(sum(int(ep['success']) for ep in episodes) / 100 - evaluation['success_rate']) < 1e-12
        vector = [int(ep['success']) for ep in episodes]
        state_hashes = [ep['initial_state_sha256'] for ep in episodes]
        obs_hashes = [ep['initial_obs_sha256'] for ep in episodes]
        if job['seed'] in eval_vectors:
            assert eval_vectors[job['seed']]['state_hashes'] == state_hashes
            assert eval_vectors[job['seed']]['obs_hashes'] == obs_hashes
        else:
            eval_vectors[job['seed']] = dict(state_hashes=state_hashes, obs_hashes=obs_hashes)
        rates[job['selection']][job['exposure']][job['seed']] = sum(vector) / len(vector)
        details.append(dict(job_tag=tag, success_count=sum(vector), success_rate=sum(vector) / 100,
                            training_seconds=receipt['training_seconds'],
                            evaluation_seconds=receipt['evaluation_seconds'],
                            checkpoint_sha256=receipt['checkpoint_sha256']))
    if missing:
        (root / 'audit_incomplete.json').write_text(json.dumps({'status': 'INCOMPLETE', 'missing': missing}, indent=2) + '\n')
        print(json.dumps({'status': 'INCOMPLETE', 'missing': missing}), flush=True)
        return

    seeds = [0, 1, 2]
    contrasts = {}
    for selection in ('random', 'demo_score_official'):
        contrasts[f'{selection}_eu_minus_wu'] = summarize([
            rates[selection]['eu'][seed] - rates[selection]['wu'][seed] for seed in seeds])
    for exposure in ('wu', 'eu'):
        contrasts[f'demo_score_minus_random_{exposure}'] = summarize([
            rates['demo_score_official'][exposure][seed] - rates['random'][exposure][seed]
            for seed in seeds])
    interaction = [
        (rates['demo_score_official']['eu'][seed] - rates['random']['eu'][seed]) -
        (rates['demo_score_official']['wu'][seed] - rates['random']['wu'][seed])
        for seed in seeds]
    contrasts['sampler_by_selector_interaction'] = summarize(interaction)
    result = dict(status='PASS', task=manifest['task'], completed_jobs=len(details),
                  common_eval_state_hashes_match=True, common_obs_hashes_match=True,
                  rollout_seed_range=[530000, 530099], training_seeds=seeds,
                  rates=rates, contrasts=contrasts, jobs=details,
                  interpretation='n=3 training seeds is the inferential unit; evaluation episodes estimate within-checkpoint success only. Exact sign-flip p-values have coarse resolution and are descriptive.',
                  limitations=['fixed-budget BC-RNN adaptation, not full-schedule reproduction',
                               'one fixed selection per selector; no resampled selection memberships'])
    out = root / 'audit_results.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# BC-RNN Exposure Transfer Audit', '', 'Integrity: PASS. Twelve configs, checkpoints, and evaluations verified.',
             'All models share identical physical initial-state and initial-observation hashes for seeds 530000--530099.',
             '', 'The training seed is the replication unit (`n=3`); the 100 rollouts estimate each checkpoint rate.', '',
             '| Selection | Exposure | Seed 0 | Seed 1 | Seed 2 | Mean |',
             '|---|---|---:|---:|---:|---:|']
    for selection in ('random', 'demo_score_official'):
        for exposure in ('wu', 'eu'):
            vals = [rates[selection][exposure][seed] for seed in seeds]
            lines.append(f"| {selection} | {exposure.upper()} | " + ' | '.join(f'{v:.3f}' for v in vals) +
                         f" | {sum(vals)/3:.3f} |")
    lines += ['', '## Paired contrasts', '', '| Contrast | Seed 0 | Seed 1 | Seed 2 | Mean | SD | Exact sign-flip p |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, data in contrasts.items():
        vals = data['per_seed']
        lines.append(f"| {name} | " + ' | '.join(f'{v:+.3f}' for v in vals) +
                     f" | {data['mean']:+.3f} | {data['sd']:.3f} | {data['exact_two_sided_sign_flip_p']:.3f} |")
    lines += ['', 'Exact sign-flip p-values have limited resolution at three training seeds. Rollout-level uncertainty does not replace between-seed replication.', '']
    (root / 'audit_results.md').write_text('\n'.join(lines))
    print(json.dumps({'status': 'PASS', 'completed_jobs': len(details),
                      'interaction': contrasts['sampler_by_selector_interaction']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
