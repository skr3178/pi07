"""Offline action-sampling diagnostics for the saved matched training pilot."""
import argparse
import gc
import json
from pathlib import Path
import time

from train_goal_pilot import ROOT, GoalIndex, GoalSamples, LeRobotDataset, default_collate
from train_goal_pilot import GOALS, GoalSmolVLAPolicy, load_smolvla_config, make_pre_post_processors
from train_goal_pilot import flow_inputs, file_hash, write_json, np, torch


def main(args):
    run = args.run.resolve()
    out = run / "offline_action_diagnostics"
    if out.exists():
        raise RuntimeError("Preserve existing diagnostics; use a new run directory")
    training = json.loads((run / "report.json").read_text())
    if training["status"] != "complete" or not training["matched_conditions_verified"]:
        raise RuntimeError("Both matched training arms must finish successfully")
    out.mkdir()
    torch.set_num_threads(4)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    schedule = dict(np.load(run / "validation_schedule.npz"))
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i/index.info["fps"] for i in range(50)]})
    samples = GoalSamples(dataset, schedule, "B")
    samples = [samples[i] for i in range(len(samples))]
    manifest = {"purpose": "offline sampled-action diagnostics; real futures are unavailable in live rollouts",
        "conditions": ["A_dropped", "B_real_future", "B_dropped", "B_copied_current", "B_mismatched_task"],
        "validation_schedule_sha256": file_hash(run / "validation_schedule.npz"),
        "checkpoint_sha256": {a: file_hash(run / a / "model.safetensors") for a in ("A", "B")},
        "batch_size": 8, "noise_seed": "900000 + batch start", "metric_units": "checkpoint-normalized 7-D actions",
        "execution_horizon": 8, "full_horizon": 50, "denoising_steps": 10,
        "mismatch": "goal from the next task's held-out observation, cyclic shift by 10 rows",
        "source_sha256": file_hash(Path(__file__))}
    write_json(out / "manifest.json", manifest)
    report = {"status": "running", "conditions": {}}
    for arm in ("A", "B"):
        cfg, _ = load_smolvla_config(run / arm)
        policy = GoalSmolVLAPolicy.from_pretrained(run / arm, config=cfg, strict=True).eval()
        pre, _ = make_pre_post_processors(cfg, pretrained_path=str(run / arm))
        modes = ["dropped"] if arm == "A" else ["real_future", "dropped", "copied_current", "mismatched_task"]
        for mode in modes:
            rows, predictions = [], []
            start_time = time.perf_counter()
            for start in range(0, len(samples), 8):
                raw = default_collate(samples[start:start+8])
                for camera, goal in zip(("observation.images.image", "observation.images.image2"), GOALS):
                    raw[f"{goal}_padding_mask"][:] = mode != "dropped"
                    if mode == "copied_current":
                        raw[goal] = raw[camera].clone()
                    if mode == "mismatched_task":
                        raw[goal] = torch.stack([samples[(i+10)%len(samples)][goal]
                                                for i in range(start, start+len(raw["task"]))])
                batch = pre(raw)
                noise, _ = flow_inputs(900000+start, len(raw["task"]))
                with torch.inference_mode():
                    pred = policy.predict_action_chunk(batch, noise=noise.cuda())
                if not torch.isfinite(pred).all():
                    raise RuntimeError("Nonfinite sampled action")
                error = (pred-batch["action"]).float().square().cpu()
                valid = ~batch["action_is_pad"].cpu()
                predictions.extend(pred.cpu().numpy())
                for b in range(len(raw["task"])):
                    i = start+b; current = int(schedule["current"][i])
                    row = {"index": current, "episode": int(index.episodes[current]), "task_id": int(index.tasks[current])}
                    for horizon in (8,50):
                        values = error[b,:horizon][valid[b,:horizon]]
                        row[f"squared_error_sum_{horizon}"] = float(values.sum())
                        row[f"components_{horizon}"] = values.numel()
                        row[f"mse_{horizon}"] = float(values.mean())
                    rows.append(row)
            metrics = {}
            for horizon in (8,50):
                metrics[f"mse_{horizon}"] = sum(r[f"squared_error_sum_{horizon}"] for r in rows)/sum(r[f"components_{horizon}"] for r in rows)
            metrics["per_task_mse_8"] = {str(task): float(np.mean([r["mse_8"] for r in rows if r["task_id"]==task])) for task in range(10)}
            metrics["observations"] = len(rows)
            metrics["wall_seconds"] = time.perf_counter()-start_time
            key = f"{arm}_{mode}"
            np.savez_compressed(out / f"{key}.npz", actions=np.asarray(predictions), current=schedule["current"])
            report["conditions"][key] = {**metrics, "samples": rows}
            write_json(out / "report.json", report)
            print(f"{key}: {json.dumps(metrics)}", flush=True)
        del policy, pre
        gc.collect(); torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(out / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/goal_training_pilot")
    main(parser.parse_args())
