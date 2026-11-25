"""Quantify decoding, action padding and episode boundaries across all 10 tasks."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / "datasets_cache"))

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/libero_10_subtask", root=PROJECT / "data/libero_10_subtask",
                         video_backend="torchcodec", delta_timestamps={"action": [i/10 for i in range(50)]})
report = {"status": "running", "episodes": len(dataset.meta.episodes), "frames": len(dataset),
          "samples": [], "decoded_samples": 0, "finite_samples": 0,
          "cross_episode_action_queries": 0, "action_padding_mismatches": 0}
for episode in range(0, 500, 50):
    meta = dataset.meta.episodes[episode]
    start, stop = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    for idx in (start, (start+stop-1)//2, stop-1):
        sample = dataset[idx]
        query, padding = dataset._get_query_indices(idx, episode)
        report["cross_episode_action_queries"] += sum(not start <= j < stop for j in query["action"])
        expected_pad = torch.tensor([idx+i >= stop for i in range(50)])
        report["action_padding_mismatches"] += int((sample["action_is_pad"] != expected_pad).sum())
        keys = ("observation.images.image", "observation.images.image2", "observation.state", "action")
        finite = all(torch.isfinite(sample[k]).all().item() for k in keys)
        assert sample["action"].shape == (50,7) and sample["observation.state"].shape == (8,)
        for key in keys[:2]:
            assert sample[key].shape == (3,256,256) and 0 <= sample[key].min() <= sample[key].max() <= 1
        # Match every sampled action against the original in-episode table rows.
        raw = torch.stack(dataset.hf_dataset[query["action"]]["action"])
        torch.testing.assert_close(sample["action"], raw, atol=0, rtol=0)
        report["decoded_samples"] += 1
        report["finite_samples"] += int(finite)
        report["samples"].append({"episode": episode, "frame": idx-start,
                                  "task": sample["task"], "padded_action_steps": int(expected_pad.sum())})
report["task_count"] = len({r["task"] for r in report["samples"]})
assert report["finite_samples"] == report["decoded_samples"] == 30
assert report["task_count"] == 10
assert report["cross_episode_action_queries"] == report["action_padding_mismatches"] == 0
report["status"] = "passed"
report["scope"] = "30 stratified runtime samples, not an exhaustive visual or semantic label audit; future-goal pair builder remains untested."
(HERE / "input_audit.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2))
