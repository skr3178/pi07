"""Reproducible matched A/B development training, with fixed offline diagnostics.

This is a small optimization pilot, not the final closed-loop benchmark.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "artifacts/lerobot_libero_setup/datasets_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from libero_runtime import load_smolvla_config
from goal_data import GoalIndex, GoalSamples, save_schedule
from goal_policy import CAMERAS, GOALS, GoalSmolVLAPolicy

BASE = Path("/media/skr/storage/models/smolvla_libero")


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2)+"\n"); tmp.replace(path)


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def flow_inputs(seed, count):
    # Explicit CPU generators make A/B noise independent of decoding/model RNG.
    noise = torch.randn(count, 50, 32, generator=torch.Generator().manual_seed(seed))
    # Inverse CDF of the upstream Beta(1.5, 1.0), with its same scale/offset.
    times = torch.from_numpy((np.random.default_rng(seed).random(count)**(1/1.5)*.999+.001).astype(np.float32))
    return noise, times


@torch.inference_mode()
def validate(policy, pre, samples, arm, batch_size):
    policy.eval()
    modes = ["dropped"] if arm == "A" else ["real_future", "dropped", "copied_current"]
    result = {}
    for mode in modes:
        numerator = denominator = 0
        for start in range(0, len(samples), batch_size):
            raw = default_collate(samples[start:start+batch_size])
            for camera, goal in zip(("observation.images.image", "observation.images.image2"), GOALS):
                raw[f"{goal}_padding_mask"][:] = mode != "dropped"
                if mode == "copied_current":
                    raw[goal] = raw[camera].clone()
            batch = pre(raw)
            noise, times = flow_inputs(900000+start, len(raw["task"]))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, details = policy(batch, noise=noise.cuda(), time=times.cuda())
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite validation loss")
            count = details["valid_action_components"]
            numerator += float(loss)*count; denominator += count
        result[mode] = {"flow_mse": numerator/denominator, "valid_action_components": denominator}
    return result


def main(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise RuntimeError("Preserve existing run; choose a fresh output directory")
    gate = json.loads((ROOT / "artifacts/goal_pipeline_checks/report.json").read_text())
    if gate["status"] != "passed":
        raise RuntimeError("Integration gates must pass first")
    torch.set_num_threads(4)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    split = index.split()
    write_json(out / "episode_split.json", split)
    train = index.schedule(split["train_episodes"], args.steps*args.batch_size, args.seed)
    # Ten held-out observations per task; all get a real goal for the diagnostic.
    parts = [index.schedule([e for e in split["validation_episodes"]
                             if index.tasks[index.bounds[e][0]] == task], 10, args.seed+100+task, 1.)
             for task in range(10)]
    validation = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    source_files = ["goal_policy.py", "goal_data.py", "libero_runtime.py", "train_goal_pilot.py"]
    manifest = {"purpose": "matched development training pilot, not final benchmark", "seed": args.seed,
        "steps_per_arm": args.steps, "batch_size": args.batch_size, "examples_per_arm": len(train["current"]),
        "checkpoint": str(BASE), "checkpoint_sha256": file_hash(BASE / "model.safetensors"),
        "source_sha256": {name: file_hash(ROOT / name) for name in source_files},
        "dependencies": json.loads((ROOT / "artifacts/lerobot_libero_setup/source_commits.json").read_text()),
        "train_schedule_sha256": save_schedule(out / "train_schedule.npz", train),
        "validation_schedule_sha256": save_schedule(out / "validation_schedule.npz", validation),
        "split_sha256": file_hash(out / "episode_split.json"),
        "dataset_revision": "66a4f67dd88bde16c2d55395edf23617d04972dd",
        "optimizer": {"name": "AdamW", "lr": args.lr, "betas": [.9,.95], "eps": 1e-8,
                      "weight_decay": 1e-10, "gradient_clip": 10., "warmup_steps": 20},
        "precision": "BF16 autocast; checkpoint/module parameter dtypes retained",
        "trainable": "action expert/projections and state projection; VLM frozen",
        "attention": "fixed four image slots; observations cannot see goals/text; identical in A/B",
        "goal_present_count_B": int(train["goal_present"].sum()),
        "goal_present_rate_B": float(train["goal_present"].mean()),
        "goal_sampling": "25% segment endpoint, otherwise uniform 0–40 remaining episode frames",
        "task_text": "same full task in both arms; no subtask text or metadata yet",
        "action_chunk": 50, "execute_steps": 8, "validation_observations": 100,
        "validation_limits": "fixed flow-noise diagnostic; not action rollout success or pretraining holdout"}
    write_json(out / "manifest.json", manifest)  # Persist choices before training.
    report = {"status": "running", "arms": {}}
    write_json(out / "report.json", report)
    try:
        dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                                 delta_timestamps={"action": [i/index.info["fps"] for i in range(50)]})
        heldout = GoalSamples(dataset, validation, "B")
        heldout = [heldout[i] for i in range(len(heldout))]
        for arm in ("A", "B"):
            torch.manual_seed(args.seed)
            cfg, _ = load_smolvla_config(BASE)
            policy = GoalSmolVLAPolicy.from_pretrained(BASE, config=cfg, strict=True)
            pre, post = make_pre_post_processors(cfg, pretrained_path=str(BASE))
            params = [p for p in policy.parameters() if p.requires_grad]
            frozen_before = state_hash(policy.model.vlm_with_expert.vlm)
            optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(.9,.95), eps=1e-8, weight_decay=1e-10)
            row = {"initial_state_sha256": state_hash(policy), "trainable_parameters": sum(p.numel() for p in params),
                   "validation": [], "updates": []}
            report["arms"][arm] = row
            row["validation"].append({"step": 0, **validate(policy, pre, heldout, arm, args.batch_size)})
            torch.cuda.reset_peak_memory_stats()
            loader = DataLoader(GoalSamples(dataset, train, arm), batch_size=args.batch_size, shuffle=False, num_workers=0)
            t0 = time.perf_counter(); flow_hash = hashlib.sha256()
            for step, raw in enumerate(loader, 1):
                begin = time.perf_counter()
                batch = pre(raw)
                noise, times = flow_inputs(args.seed+100000+step, len(raw["task"]))
                flow_hash.update(noise.numpy().tobytes()); flow_hash.update(times.numpy().tobytes())
                lr = args.lr*min(step/20, 1.)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                policy.train()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, details = policy(batch, noise=noise.cuda(), time=times.cuda())
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 10., error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                row["updates"].append({"step": step, "loss": float(loss.detach()), "gradient_norm": float(grad_norm),
                    "lr": lr, "seconds_excluding_decode": time.perf_counter()-begin,
                    "valid_action_components": details["valid_action_components"]})
                if step % 50 == 0 or step == args.steps:
                    metrics = validate(policy, pre, heldout, arm, args.batch_size)
                    row["validation"].append({"step": step, **metrics})
                    write_json(out / "report.json", report)
                    print(f"{arm} step {step}: {json.dumps(metrics)}", flush=True)
            row.update({"completed_updates": args.steps, "examples": len(train["current"]),
                "training_and_validation_wall_seconds": time.perf_counter()-t0,
                "flow_inputs_sha256": flow_hash.hexdigest(),
                "frozen_backbone_unchanged": frozen_before == state_hash(policy.model.vlm_with_expert.vlm),
                "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9})
            assert row["frozen_backbone_unchanged"]
            checkpoint = out / arm
            policy.save_pretrained(checkpoint)
            pre.save_pretrained(checkpoint); post.save_pretrained(checkpoint)
            write_json(checkpoint / "goal_interface.json", {"class": "goal_policy.GoalSmolVLAPolicy",
                "arm": arm, "current_cameras": CAMERAS, "goal_cameras": GOALS,
                "mask_suffix": "_padding_mask", "true_means": "goal present", "fixed_slots": True})
            torch.save({"optimizer": optimizer.state_dict(), "step": args.steps,
                        "seed": args.seed}, checkpoint / "optimizer.pt")
            row["checkpoint_sha256"] = file_hash(checkpoint / "model.safetensors")
            row["final_state_sha256"] = state_hash(policy)
            # Verify serialization without retaining two policy/optimizer copies on GPU.
            del policy, optimizer, params, loss, batch, pre, post
            gc.collect(); torch.cuda.empty_cache()
            reload_cfg, _ = load_smolvla_config(checkpoint)
            reloaded = GoalSmolVLAPolicy.from_pretrained(checkpoint, config=reload_cfg, strict=True)
            row["checkpoint_roundtrip_exact"] = state_hash(reloaded) == row["final_state_sha256"]
            assert row["checkpoint_roundtrip_exact"]
            del reloaded; gc.collect(); torch.cuda.empty_cache()
            write_json(out / "report.json", report)
        a,b = report["arms"]["A"], report["arms"]["B"]
        for key in ("initial_state_sha256", "trainable_parameters", "completed_updates", "examples", "flow_inputs_sha256"):
            assert a[key] == b[key], key
        report["matched_conditions_verified"] = True
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"; report["error"] = traceback.format_exc()
        raise
    finally:
        write_json(out / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/goal_training_pilot")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--lr", type=float, default=1e-4)
    main(p.parse_args())
