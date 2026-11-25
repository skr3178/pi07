"""Audit a local LeRobot v3 PushT snapshot without loading a policy.

Requires numpy, pyarrow, Pillow, ffprobe, and ffmpeg with libdav1d.
Writes a JSON report and a sheet of episode-safe current/future image pairs.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.dataset / "meta/info.json").read_text())
    rows = [r for p in sorted((args.dataset / "data").rglob("*.parquet"))
            for r in pq.read_table(p).to_pylist()]
    episodes = [r for p in sorted((args.dataset / "meta/episodes").rglob("*.parquet"))
                for r in pq.read_table(p).to_pylist()]
    episodes.sort(key=lambda x: x["episode_index"])
    states = np.asarray([r["observation.state"] for r in rows])
    actions = np.asarray([r["action"] for r in rows])
    fps = meta["fps"]
    checks = {
        "row_count_matches_metadata": len(rows) == meta["total_frames"],
        "episode_count_matches_metadata": len(episodes) == meta["total_episodes"],
        "state_shape": states.shape == (len(rows), 2),
        "action_shape": actions.shape == (len(rows), 2),
        "finite_states_and_actions": bool(np.isfinite(states).all() and np.isfinite(actions).all()),
        "global_indices_contiguous": [r["index"] for r in rows] == list(range(len(rows))),
    }
    episode_stats = []
    covered = []
    for ep in episodes:
        start, end = ep["dataset_from_index"], ep["dataset_to_index"]
        chunk = rows[start:end]
        covered.extend(range(start, end))
        if len(chunk) != ep["length"] or not chunk:
            raise ValueError(f"Invalid bounds for episode {ep['episode_index']}")
        if any(r["episode_index"] != ep["episode_index"] for r in chunk):
            raise ValueError("Episode bounds include another episode")
        if [r["frame_index"] for r in chunk] != list(range(len(chunk))):
            raise ValueError("Noncontiguous episode frame indices")
        if not np.allclose([r["timestamp"] for r in chunk], np.arange(len(chunk)) / fps, atol=1e-5):
            raise ValueError("Unexpected frame timestamps")
        if not np.isclose(ep["videos/observation.image/from_timestamp"], start / fps):
            raise ValueError("Video offset does not match this snapshot's concatenated rows")
        if not np.isclose(ep["videos/observation.image/to_timestamp"], end / fps):
            raise ValueError("Video endpoint does not match episode bounds")
        episode_stats.append({
            "episode_index": ep["episode_index"], "length": len(chunk),
            "has_success": any(r["next.success"] for r in chunk),
            "final_success": bool(chunk[-1]["next.success"]),
            "final_done": bool(chunk[-1]["next.done"]),
            "max_recorded_reward": max(r["next.reward"] for r in chunk),
            "full_16_action_windows": max(0, len(chunk) - 15),
            "unclipped_t_plus_20_pairs": max(0, len(chunk) - 20),
            "unclipped_t_plus_40_pairs": max(0, len(chunk) - 40),
        })
    checks["episode_bounds_cover_each_row_once"] = covered == list(range(len(rows)))

    videos = sorted((args.dataset / "videos/observation.image").rglob("*.mp4"))
    if len(videos) != 1:
        raise ValueError("This diagnostic expects the single-video PushT snapshot")
    video = videos[0]
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames", "-of", "json", str(video)
    ]))["streams"][0]
    checks["video_frame_count_matches_rows"] = int(probe["nb_frames"]) == len(rows)
    checks["video_shape"] = (probe["height"], probe["width"], 3) == tuple(meta["features"]["observation.image"]["shape"])
    rate_n, rate_d = map(int, probe["r_frame_rate"].split("/"))
    checks["video_fps_matches_metadata"] = rate_n / rate_d == fps

    selections = []
    for ep_idx in [0, 40, 80, 120, 160, 205]:
        ep = episodes[ep_idx]
        local_t = min(ep["length"] // 3, max(0, ep["length"] - 41))
        selections.append({
            "episode_index": ep["episode_index"], "t": local_t,
            "global_indices": [ep["dataset_from_index"] + min(local_t + d, ep["length"] - 1)
                               for d in [0, 20, 40]],
        })
    selected = sorted({i for sample in selections for i in sample["global_indices"]})
    select_filter = "select=" + "+".join(f"eq(n\\,{i})" for i in selected)
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-c:v", "libdav1d", "-i", str(video),
        "-vf", select_filter, "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
    ], check=True, capture_output=True)
    height, width = probe["height"], probe["width"]
    decoded = np.frombuffer(result.stdout, np.uint8).reshape(-1, height, width, 3)
    checks["selected_video_frames_decoded"] = len(decoded) == len(selected)
    if not all(checks.values()):
        raise ValueError(f"Failed checks: {[k for k, v in checks.items() if not v]}")
    images = dict(zip(selected, decoded))
    sheet = Image.new("RGB", (3 * 220, len(selections) * 240 + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), "PushT: observed frame | real t+20 | real t+40 (not generated)", fill="black")
    sample_dir = args.output / "sample_frames"
    sample_dir.mkdir(exist_ok=True)
    for row_idx, sample in enumerate(selections):
        for col, (offset, global_idx) in enumerate(zip([0, 20, 40], sample["global_indices"])):
            im = Image.fromarray(images[global_idx])
            name = f"ep{sample['episode_index']:03d}_t{sample['t'] + offset:03d}.png"
            im.save(sample_dir / name)
            x, y = col * 220 + 10, row_idx * 240 + 30
            sheet.paste(im.resize((192, 192), Image.Resampling.NEAREST), (x, y + 28))
            draw.text((x, y + 8), f"ep {sample['episode_index']} / t={sample['t'] + offset}", fill="black")
    sheet.save(args.output / "future_frame_pairs.png")

    lengths = [x["length"] for x in episode_stats]
    warnings = []
    if not any(x["has_success"] for x in episode_stats):
        warnings.append("All stored next.success values are false. Verify label provenance before deriving quality buckets; this does not establish that the demonstrations all failed.")
    report = {
        "checks": checks,
        "warnings": warnings,
        "total_episodes": len(episodes), "total_frames": len(rows), "fps": fps,
        "tasks": pq.read_table(args.dataset / "meta/tasks.parquet").to_pylist(),
        "state_range": {"min": states.min(0).tolist(), "max": states.max(0).tolist()},
        "action_range": {"min": actions.min(0).tolist(), "max": actions.max(0).tolist()},
        "episode_length_min_median_max": [min(lengths), float(np.median(lengths)), max(lengths)],
        "episodes_with_success": sum(x["has_success"] for x in episode_stats),
        "episodes_ending_success": sum(x["final_success"] for x in episode_stats),
        "episodes_ending_done": sum(x["final_done"] for x in episode_stats),
        "episode_max_recorded_reward_quantiles": np.quantile([x["max_recorded_reward"] for x in episode_stats], [0, .25, .5, .75, 1]).tolist(),
        "window_counts": {k: sum(x[k] for x in episode_stats) for k in ["full_16_action_windows", "unclipped_t_plus_20_pairs", "unclipped_t_plus_40_pairs"]},
        "video": probe, "image_samples": selections,
        "missing_annotations": ["block_pose", "subtask", "segment_end", "quality", "speed_bin", "mistake"],
        "limits": ["Does not run the LeRobot processor or policy.", "Does not generate images or validate a world model.", "Recorded reward is not assumed to be raw coverage.", "All episodes are in the published training split; create an episode-level evaluation split before training."],
        "episodes": episode_stats,
    }
    (args.output / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ["episodes", "image_samples"]}, indent=2))


if __name__ == "__main__":
    main()
