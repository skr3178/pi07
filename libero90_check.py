"""Is Kesvill/smolvla_libero_90 (LIBERO arm, never saw LIBERO-10 demos) a usable A/B start?

Untouched checkpoint, plain SmolVLA, the 100 held-out observations of goal_training_1000, same noise
as evaluate_goal_pilot.py. Compare with smolvla_libero's 0.123 (attention_check). Images are also
scored rotated 180 degrees, in case its nvidia/LIBERO_LeRobot_v3 renders use the other orientation.
"""
import json
from pathlib import Path

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from train_goal_pilot import (ROOT, GoalIndex, GoalSamples, LeRobotDataset, default_collate, flow_inputs,
                              load_smolvla_config, make_pre_post_processors, np, write_json)

CKPT = Path("/media/skr/storage/models/smolvla_libero_90")
CAMS = ("observation.images.image", "observation.images.image2")


def evaluate(policy, pre, samples, schedule, index, flip):
    sums = {8: 0.0, 50: 0.0}; counts = {8: 0, 50: 0}; per_task = {t: [] for t in range(10)}
    for start in range(0, len(samples), 8):
        raw = default_collate(samples[start:start + 8])
        if flip:
            for cam in CAMS:
                raw[cam] = raw[cam].flip(-1, -2)
        batch = pre(raw)
        noise, _ = flow_inputs(900000 + start, len(raw["task"]))
        with torch.inference_mode():
            pred = policy.predict_action_chunk(batch, noise=noise.cuda())
        err = (pred - batch["action"]).float().square().cpu(); valid = ~batch["action_is_pad"].cpu()
        for b in range(len(raw["task"])):
            for h in (8, 50):
                v = err[b, :h][valid[b, :h]]; sums[h] += float(v.sum()); counts[h] += v.numel()
            per_task[int(index.tasks[int(schedule["current"][start + b])])].append(float(err[b, :8][valid[b, :8]].mean()))
    return {"mse_8": sums[8] / counts[8], "mse_50": sums[50] / counts[50],
            "per_task_mse_8": {t: float(np.mean(v)) for t, v in per_task.items()}}


def main():
    out = ROOT / "artifacts/libero90_check"; out.mkdir(exist_ok=False)
    index = GoalIndex(ROOT / "data/libero_10_subtask")
    schedule = dict(np.load(ROOT / "artifacts/goal_training_1000/validation_schedule.npz"))
    dataset = LeRobotDataset("lerobot/libero_10_subtask", root=index.root, video_backend="torchcodec",
                             delta_timestamps={"action": [i / index.info["fps"] for i in range(50)]})
    samples = GoalSamples(dataset, schedule, "B"); samples = [samples[i] for i in range(len(samples))]
    # Newer-LeRobot config: drop its inactive torch.compile fields (compile_model is False).
    source = json.loads((CKPT / "config.json").read_text())
    assert source.pop("compile_model") is False; source.pop("compile_mode")
    clean = out / "config"; clean.mkdir(); write_json(clean / "config.json", source)
    cfg, _ = load_smolvla_config(clean)
    policy = SmolVLAPolicy.from_pretrained(CKPT, config=cfg, strict=True).eval()
    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(CKPT))
    report = {"checkpoint": str(CKPT), "observations": len(samples), "smolvla_libero_reference_mse_8": 0.1229}
    for name, flip in (("as_is", False), ("rotated_180", True)):
        report[name] = evaluate(policy, pre, samples, schedule, index, flip)
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "per_task_mse_8"}), flush=True)
    write_json(out / "report.json", report)


if __name__ == "__main__":
    main()
