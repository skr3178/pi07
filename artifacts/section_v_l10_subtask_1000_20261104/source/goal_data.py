"""Episode-safe future-image planning and reproducible matched data schedules."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from goal_policy import GOALS


class GoalIndex:
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.table = pq.read_table(self.root / "data", columns=["index", "episode_index", "frame_index", "task_index", "subtask_index"])
        data = self.table.to_pandas().sort_values("index")
        self.episodes = data.episode_index.to_numpy()
        self.tasks = data.task_index.to_numpy()
        self.frames = data.frame_index.to_numpy()
        if not np.array_equal(data["index"].to_numpy(), np.arange(len(data))):
            raise ValueError("Dataset index must be contiguous")
        self.end = np.empty(len(data), dtype=np.int64)
        self.segment_end = np.empty(len(data), dtype=np.int64)
        self.bounds = {}
        self.repaired_artifact_episodes = []
        for ep, rows in data.groupby("episode_index", sort=True):
            idx = rows["index"].to_numpy(); labels = rows.subtask_index.to_numpy().copy()
            if not np.array_equal(idx, np.arange(idx[0], idx[-1]+1)):
                raise ValueError("An episode is not contiguous")
            if rows.task_index.iloc[0] == 3 and len(labels)>1 and labels[0]==15 and labels[1]!=15:
                labels[0] = labels[1]
                self.repaired_artifact_episodes.append(int(ep))
            starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1]
            ends = np.r_[starts[1:], len(labels)]
            for a,b in zip(starts,ends):
                self.segment_end[idx[a:b]] = idx[b-1]
            self.end[idx] = idx[-1]
            self.bounds[int(ep)] = (int(idx[0]), int(idx[-1])+1)

    def split(self, seed=20261029, validation_per_task=5):
        rng = np.random.default_rng(seed)
        train, validation = [], []
        for task in sorted(np.unique(self.tasks)):
            eps = np.unique(self.episodes[self.tasks==task]).copy()
            rng.shuffle(eps)
            validation.extend(int(e) for e in eps[:validation_per_task])
            train.extend(int(e) for e in eps[validation_per_task:])
        return {"seed": seed, "train_episodes": sorted(train), "validation_episodes": sorted(validation),
                "split_unit": "episode", "physical_initial_state_grouping_verified": False,
                "limitation": "No simulator initial-state IDs in this dataset; episode holdout is not a verified physical-state or pretraining holdout."}

    def schedule(self, episodes, n, seed, goal_probability=0.25):
        rng = np.random.default_rng(seed)
        allowed = np.flatnonzero(np.isin(self.episodes, episodes))
        current = rng.choice(allowed, n, replace=True)
        # Separate random draws ensure A/B use the same observations and targets.
        present = rng.random(n) < goal_probability
        endpoint = rng.random(n) < 0.25
        max_delta = np.minimum(40, self.end[current]-current)
        future = current + (rng.random(n)*(max_delta+1)).astype(np.int64)
        future[endpoint] = self.segment_end[current[endpoint]]
        if np.any(self.episodes[future] != self.episodes[current]) or np.any(future < current):
            raise AssertionError("Invalid future pair")
        return {"current": current, "future": future, "goal_present": present, "segment_endpoint": endpoint}


class GoalSamples(Dataset):
    def __init__(self, dataset, schedule, arm):
        if arm not in ("A", "B"):
            raise ValueError(arm)
        self.dataset, self.schedule, self.arm = dataset, schedule, arm

    def __len__(self):
        return len(self.schedule["current"])

    def __getitem__(self, index):
        current = int(self.schedule["current"][index]); future = int(self.schedule["future"][index])
        sample = self.dataset[current]
        keep = ("observation.images.image", "observation.images.image2", "observation.state", "action", "action_is_pad", "task")
        item = {k: sample[k] for k in keep}
        present = self.arm == "B" and bool(self.schedule["goal_present"][index])
        future_sample = self.dataset[future] if present else None
        for source, target in zip(keep[:2], GOALS):
            item[target] = future_sample[source] if present else torch.zeros_like(sample[source])
            item[f"{target}_padding_mask"] = torch.tensor(present)
        return item


def save_schedule(path, schedule):
    path = Path(path)
    np.savez_compressed(path, **schedule)
    return hashlib.sha256(path.read_bytes()).hexdigest()
