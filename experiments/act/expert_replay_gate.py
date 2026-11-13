"""Check the Can absolute-action evaluator with recorded expert actions.

This is an adapter gate, not a policy result. It resets the current RoboSuite
environment to recorded demo states and replays the matching low_dim_abs action
sequence under the same legacy-controller conversion used by ACT evaluation.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np


def make_env(dataset_path: str):
    from robomimic.utils import env_utils as EnvUtils
    from robomimic.utils import file_utils as FileUtils
    from robomimic.utils import obs_utils as ObsUtils
    try:
        from robosuite.controllers.composite.composite_controller_factory import (
            refactor_composite_controller_config,
        )
    except ImportError:
        refactor_composite_controller_config = None

    meta = copy.deepcopy(FileUtils.get_env_metadata_from_dataset(dataset_path))
    kwargs = meta.setdefault("env_kwargs", {})
    controller = kwargs.get("controller_configs")
    if refactor_composite_controller_config is not None and isinstance(controller, dict) and str(controller.get("type", "")).upper() in {
        "OSC_POSE", "JOINT_POSITION", "IK_POSE", "JOINT_VELOCITY",
    }:
        robots = kwargs.get("robots", ["Panda"])
        robot = robots[0] if isinstance(robots, (list, tuple)) else robots
        kwargs["controller_configs"] = refactor_composite_controller_config(
            controller_config=controller, robot_type=robot, arms=["right"]
        )
        kwargs["lite_physics"] = False
    elif isinstance(controller, dict):
        # RoboSuite 1.4.1 uses the archived OSC_POSE controller directly.
        # Keep its action convention and switch only to absolute targets.
        controller["control_delta"] = False
    kwargs["use_camera_obs"] = False
    kwargs["has_offscreen_renderer"] = False
    kwargs["has_renderer"] = False
    kwargs["use_object_obs"] = True
    # The low-dimensional replay does not use the image runner's shape_meta,
    # so initialize the same modality registry explicitly.
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "low_dim": [
                "object", "robot0_eef_pos", "robot0_eef_quat",
                "robot0_eef_vel_ang", "robot0_eef_vel_lin",
                "robot0_gripper_qpos", "robot0_gripper_qvel",
                "robot0_joint_pos", "robot0_joint_vel",
            ],
            "rgb": [],
        }
    )
    return EnvUtils.create_env_from_metadata(
        env_meta=meta, render=False, render_offscreen=False, use_image_obs=False
    )


def replay(dataset_path: str, demo_ids: list[int], out_path: str) -> None:
    env = make_env(dataset_path)
    results = []
    with h5py.File(dataset_path, "r") as f:
        for demo_id in demo_ids:
            group = f["data"][f"demo_{demo_id}"]
            states = np.asarray(group["states"])
            actions = np.asarray(group["actions"], dtype=np.float32)
            env.reset()
            reset_state = {"states": states[0]}
            if "model_file" in group.attrs:
                reset_state["model"] = group.attrs["model_file"]
            if "ep_meta" in group.attrs:
                reset_state["ep_meta"] = group.attrs["ep_meta"]
            env.reset_to(reset_state)
            rewards = []
            finite = True
            for action in actions:
                finite = finite and bool(np.all(np.isfinite(action)))
                _, reward, _, _ = env.step(action)
                rewards.append(float(reward))
            results.append(
                {
                    "demo": int(demo_id),
                    "steps": int(len(actions)),
                    "finite_actions": finite,
                    "max_reward": float(max(rewards) if rewards else 0.0),
                    "final_reward": float(rewards[-1] if rewards else 0.0),
                    "success": bool(max(rewards) > 0.0),
                }
            )
    env.close()
    summary = {
        "dataset": dataset_path,
        "demo_ids": demo_ids,
        "results": results,
        "successes": int(sum(r["success"] for r in results)),
        "total": len(results),
    }
    Path(out_path).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--demos", nargs="+", type=int, default=[0, 1, 3, 7, 18])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    replay(args.dataset, args.demos, args.out)


if __name__ == "__main__":
    main()
