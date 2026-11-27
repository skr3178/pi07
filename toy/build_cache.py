"""Low-resolution frame cache of LIBERO-Goal for the toy Section V tests (results in todo.md).

Decodes every frame of both cameras once, resizes to 96x96 and stores uint8 arrays, so the tiny policy can
train from GPU memory without video decoding. Frame number in a video file = round(from_timestamp * fps) +
frame_index; a sample is cross-checked against decoding by timestamp.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchcodec.decoders import VideoDecoder

SRC = Path("/media/skr/storage/datasets/LIBERO-fastwam/lerobot_v30/libero_goal_no_noops_lerobot")
DST = Path("/media/skr/storage/datasets/pi07_toy/libero_goal_96")
CAMS = {"main": "observation.images.image", "wrist": "observation.images.wrist_image"}
SIZE = 96


def small(frames):
    x = frames.cuda().float()
    return torch.nn.functional.interpolate(x, size=(SIZE, SIZE), mode="area").round().clamp(0, 255).byte().cpu()


def main():
    DST.mkdir(parents=True, exist_ok=False)
    info = json.loads((SRC / "meta/info.json").read_text()); fps = info["fps"]
    eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "meta/episodes/*/*.parquet")))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(SRC / "data/*/*.parquet")))])
    data = data.sort_values("index").reset_index(drop=True)
    n = len(data); assert np.array_equal(data["index"].to_numpy(), np.arange(n)) and n == info["total_frames"]
    rng = np.random.default_rng(0); check = {}
    for name, key in CAMS.items():
        out = np.lib.format.open_memmap(DST / f"{name}.npy", mode="w+", dtype=np.uint8, shape=(n, 3, SIZE, SIZE))
        groups = eps.groupby([f"videos/{key}/chunk_index", f"videos/{key}/file_index"])
        worst = 0.0
        for (chunk, file), rows in groups:
            dec = VideoDecoder(str(SRC / f"videos/{key}/chunk-{chunk:03d}/file-{file:03d}.mp4"))
            assert len(dec) == int(rows["length"].sum()), (name, chunk, file, len(dec), int(rows["length"].sum()))
            for _, ep in rows.iterrows():
                first = int(round(ep[f"videos/{key}/from_timestamp"] * fps)); length = int(ep["length"])
                a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"]); assert b - a == length
                for s in range(0, length, 256):
                    e = min(s + 256, length)
                    out[a + s:a + e] = small(dec[first + s:first + e]).numpy()
                if rng.random() < 0.05:  # cross-check one frame of this episode against decode-by-timestamp
                    f = int(rng.integers(length))
                    ref = small(dec.get_frames_played_at([ep[f"videos/{key}/from_timestamp"] + f / fps]).data)[0]
                    worst = max(worst, float((ref.float() - torch.from_numpy(out[a + f]).float()).abs().mean()))
            print(name, chunk, file, "done", flush=True)
        out.flush(); check[name] = worst
    ends = np.zeros(n, dtype=np.int64); starts = np.zeros(n, dtype=np.int64)
    for _, ep in eps.iterrows():
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"]); starts[a:b] = a; ends[a:b] = b - 1
    np.savez(DST / "meta.npz", state=np.stack(data["observation.state"].to_numpy()).astype(np.float32),
             action=np.stack(data["action"].to_numpy()).astype(np.float32),
             episode=data["episode_index"].to_numpy(), task=data["task_index"].to_numpy(),
             frame=data["frame_index"].to_numpy(), ep_start=starts, ep_end=ends)
    tasks = pd.read_parquet(SRC / "meta/tasks.parquet")
    (DST / "info.json").write_text(json.dumps({"source": str(SRC), "fps": fps, "frames": n, "episodes": len(eps),
                                               "size": SIZE, "tasks": {int(v): k for k, v in tasks["task_index"].items()},
                                               "max_mean_abs_diff_vs_timestamp_decode": check}, indent=2))
    print("check", check, "frames", n)


if __name__ == "__main__":
    main()
