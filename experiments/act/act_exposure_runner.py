"""Official ACT model on the frozen Can-MH RGB exposure contract.

The model and optimizer are imported from the upstream ACT repository. The
adapter reads the already frozen Can image cache, samples legal windows with
WU or EU exposure, pads Can's 9-D state and 10-D action to ACT's native 14-D
interface, and writes training receipts. Closed-loop evaluation is a separate
gate because the upstream ACT repository has ALOHA evaluators rather than a
robomimic Can evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


CAMERAS = ["agentview_image", "robot0_eye_in_hand_image"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CanCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        cache = root / "cache"
        contract = json.loads((root / "data_contract.json").read_text())
        self.selected_ids = contract["selected_ids"]
        self.window_counts = np.asarray(contract["window_counts"], dtype=np.int64)
        self.lengths = self.window_counts + 7
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths[:-1])])
        self.total = int(self.lengths.sum())
        self.pos = np.load(cache / "robot0_eef_pos.npy", mmap_mode="r")
        self.quat = np.load(cache / "robot0_eef_quat.npy", mmap_mode="r")
        self.grip = np.load(cache / "robot0_gripper_qpos.npy", mmap_mode="r")
        self.images = [np.load(cache / f"{k}.npy", mmap_mode="r") for k in CAMERAS]
        self.action = np.load(cache / "action.npy", mmap_mode="r")
        assert self.total == int(self.action.shape[0]) == int(contract["total_frames"])

    def stats(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        q = np.concatenate([np.asarray(self.pos), np.asarray(self.quat), np.asarray(self.grip)], axis=1)
        a = np.asarray(self.action)
        q_mean, q_std = q.mean(0), np.maximum(q.std(0), 1e-3)
        a_mean, a_std = a.mean(0), np.maximum(a.std(0), 1e-3)
        return q_mean, q_std, a_mean, a_std

    def sample(self, rng: np.random.Generator, batch: int, exposure: str, chunk: int,
               q_mean: np.ndarray, q_std: np.ndarray, a_mean: np.ndarray, a_std: np.ndarray,
               device: torch.device):
        if exposure == "wu":
            probs = self.window_counts / self.window_counts.sum()
        elif exposure == "eu":
            probs = np.full(len(self.lengths), 1.0 / len(self.lengths))
        else:
            raise ValueError(exposure)
        ep = rng.choice(len(self.lengths), size=batch, p=probs)
        starts = np.asarray([rng.integers(0, self.window_counts[i]) for i in ep], dtype=np.int64)
        t = self.offsets[ep] + starts
        q = np.concatenate([np.asarray(self.pos[t]), np.asarray(self.quat[t]), np.asarray(self.grip[t])], axis=1)
        q = (q - q_mean) / q_std
        q14 = np.zeros((batch, 14), dtype=np.float32); q14[:, :9] = q
        act = np.zeros((batch, chunk, 14), dtype=np.float32)
        pad = np.ones((batch, chunk), dtype=bool)
        for j, start in enumerate(t):
            n = min(chunk, self.total - int(start))
            # Do not cross selected demonstration boundaries.
            e = int(ep[j]); n = min(n, int(self.lengths[e] - starts[j]))
            raw = np.asarray(self.action[start:start + n])
            act[j, :n, :10] = (raw - a_mean) / a_std
            pad[j, :n] = False
        ims = []
        for arr in self.images:
            x = np.asarray(arr[t]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
            ims.append(x)
        image = np.stack(ims, axis=1)
        return (torch.from_numpy(q14).to(device, non_blocking=True),
                torch.from_numpy(image).to(device, non_blocking=True),
                torch.from_numpy(act).to(device, non_blocking=True),
                torch.from_numpy(pad).to(device, non_blocking=True))


def make_policy(act_root: Path, chunk: int):
    sys.path.insert(0, str(act_root))
    from policy import ACTPolicy
    cfg = {
        "lr": 1e-5, "lr_backbone": 1e-5, "num_queries": chunk,
        "kl_weight": 10, "hidden_dim": 512, "dim_feedforward": 3200,
        "backbone": "resnet18", "enc_layers": 4, "dec_layers": 7,
        "nheads": 8, "camera_names": CAMERAS,
    }
    return ACTPolicy(cfg), cfg


def run(args: argparse.Namespace) -> None:
    seed_all(args.seed)
    root = Path(args.data_root).expanduser().resolve()
    act_root = Path(args.act_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"{args.arm}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cache = CanCache(root)
    q_mean, q_std, a_mean, a_std = cache.stats()
    (run_dir / "data_receipt.json").write_text(json.dumps({
        "root": str(root), "selected_ids": cache.selected_ids,
        "window_counts": cache.window_counts.tolist(), "total_frames": cache.total,
        "state_dims": {"can": 9, "act_padded": 14},
        "action_dims": {"can": 10, "act_padded": 14}, "exposure": args.arm,
        "chunk_size": args.chunk, "seed": args.seed,
    }, indent=2))
    policy, cfg = make_policy(act_root, args.chunk)
    policy.to(device)
    policy.train(); optimizer = policy.configure_optimizers()
    rng = np.random.default_rng(args.seed)
    started = time.time(); losses = []
    for step in range(1, args.steps + 1):
        q, image, act, pad = cache.sample(rng, args.batch, args.arm, args.chunk,
                                          q_mean, q_std, a_mean, a_std, device)
        loss_dict = policy(q, image, act, pad)
        optimizer.zero_grad(set_to_none=True); loss_dict["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); optimizer.step()
        value = float(loss_dict["loss"].detach().cpu()); losses.append(value)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.time() - started
            print(json.dumps({"step": step, "loss": value, "mean_loss": float(np.mean(losses[-args.log_every:])), "elapsed_s": elapsed}), flush=True)
            if step == 1 or step == args.steps or step % 1000 == 0:
                torch.save({"model": policy.state_dict(), "config": cfg, "q_mean": q_mean, "q_std": q_std,
                            "a_mean": a_mean, "a_std": a_std, "step": step}, run_dir / f"checkpoint_{step:06d}.pt")
    (run_dir / "completed.json").write_text(json.dumps({"arm": args.arm, "seed": args.seed,
        "steps": args.steps, "elapsed_s": time.time() - started, "final_loss": losses[-1]}, indent=2))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["wu", "eu"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--data-root", required=True,
                   help="Directory containing data_contract.json and cache/")
    p.add_argument("--act-root", required=True, help="Upstream ACT checkout")
    p.add_argument("--out-root", required=True, help="Directory for checkpoints and receipts")
    p.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
