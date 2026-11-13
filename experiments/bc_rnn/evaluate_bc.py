#!/usr/bin/env python3
"""Evaluate a robomimic BC checkpoint on a reproducible seeded Can rollout stream."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--seed-start', type=int, default=530000)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--horizon', type=int, default=500)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    import torch
    torch.set_num_threads(4)
    random.seed(args.seed_start)
    np.random.seed(args.seed_start)
    torch.manual_seed(args.seed_start)
    torch.cuda.manual_seed_all(args.seed_start)

    from robomimic.utils import env_utils as EnvUtils
    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import train_utils as TrainUtils
    from robomimic.utils import obs_utils as ObsUtils

    started = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy, checkpoint = FileUtils.policy_from_checkpoint(
        device=device, ckpt_path=str(args.checkpoint), verbose=False)
    config, _ = FileUtils.config_from_checkpoint(
        algo_name=checkpoint['algo_name'], ckpt_dict=checkpoint, verbose=False)
    ObsUtils.initialize_obs_utils_with_config(config)
    episode_records = []
    try:
        for episode_idx in range(args.episodes):
            current: dict = {}
            episode_seed = args.seed_start + episode_idx
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            torch.cuda.manual_seed_all(episode_seed)
            env = EnvUtils.create_env_from_metadata(
                env_meta=checkpoint['env_metadata'], render=False, render_offscreen=False,
                use_image_obs=False, seed=episode_seed)
            env = EnvUtils.wrap_env_from_config(env, config=config)
            original_reset, original_step = env.reset, env.step

            def observed_reset(*reset_args, **reset_kwargs):
                obs = original_reset(*reset_args, **reset_kwargs)
                state = env.get_state()
                current.clear()
                current.update(
                    initial_state_sha256=digest(np.asarray(state['states']).tobytes()),
                    initial_model_sha256=digest(state.get('model', '').encode()),
                    initial_obs_sha256=digest(b''.join(
                        key.encode() + np.ascontiguousarray(obs[key]).tobytes()
                        for key in sorted(obs))),
                    rewards=[], successes=[])
                return obs

            def observed_step(action):
                result = original_step(action)
                _obs, reward, _done, info = result
                current['rewards'].append(float(np.asarray(reward).max()))
                metric = info.get('is_success', {})
                current['successes'].append(bool(metric.get('task', False)))
                return result

            env.reset = observed_reset
            env.step = observed_step
            try:
                result = TrainUtils.run_rollout(
                    policy=policy, env=env, horizon=args.horizon,
                    use_goals=False, render=False, terminate_on_success=True)
                assert current.get('rewards'), f'episode {episode_idx} recorded no rewards'
                success = bool(max(current['rewards']) >= 1.0)
                assert success == bool(any(current['successes']))
                episode_records.append(dict(
                    episode_index=episode_idx,
                    evaluation_seed=episode_seed,
                    initial_state_sha256=current['initial_state_sha256'],
                    initial_model_sha256=current['initial_model_sha256'],
                    initial_obs_sha256=current['initial_obs_sha256'],
                    rewards=list(current['rewards']),
                    success_steps=[i for i, value in enumerate(current['rewards']) if value >= 1.0],
                    success=success,
                    return_sum=float(sum(current['rewards'])),
                    length=len(current['rewards']),
                    rollout_return=float(result['Return']),
                    rollout_success=bool(result['Success_Rate'])))
            finally:
                env.close()
    finally:
        policy.policy.set_eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        status='PASS', completed_unix=time.time(), seconds=time.time() - started,
        checkpoint=str(args.checkpoint), checkpoint_sha256=digest(args.checkpoint.read_bytes()),
        algo=checkpoint['algo_name'], seed_start=args.seed_start,
        evaluation_seeds=list(range(args.seed_start, args.seed_start + args.episodes)),
        episodes=args.episodes, horizon=args.horizon,
        success_count=sum(int(row['success']) for row in episode_records),
        success_rate=float(np.mean([row['success'] for row in episode_records])),
        deterministic_policy_mode='robomimic RolloutPolicy eval mode; GMM low_noise_eval from checkpoint config',
        environment_seed=args.seed_start,
        episodes_data=episode_records)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: payload[k] for k in
                      ['status', 'seconds', 'success_count', 'success_rate', 'episodes']},
                     sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
