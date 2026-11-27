"""Side-by-side comparison videos from toy/rollout.py runs: same task, same start, one column per condition.

Each column: condition label and outcome | live camera | the goal image the policy was given (blank if none).
Shorter episodes hold their last frame. Writes artifacts/toy/rollouts/comparison/task{T}_start{S}.mp4,
a montage of the first frame of every comparison, and README.md with the success table.
"""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / "artifacts/toy/rollouts"


def frames(path):
    reader = imageio.get_reader(str(path)); out = [f[48:304] for f in reader]; reader.close()
    return out  # rows below the 48-px text bar: [live cam | live wrist | goal cam | goal wrist]


def label(text, sub, ok, width=256, height=44):
    bar = Image.new("RGB", (width, height), (25, 25, 25)); d = ImageDraw.Draw(bar)
    d.text((5, 4), text, fill=(255, 255, 255))
    d.text((5, 24), sub, fill=(120, 230, 120) if ok else (240, 130, 130) if ok is False else (200, 200, 200))
    return np.asarray(bar)


def main(a):
    runs = [r.split("=") for r in a.runs.split(",")]  # name=Column title
    reports = {n: json.loads((ROOT / n / "report.json").read_text()) for n, _ in runs}
    out = ROOT / "comparison"; out.mkdir(exist_ok=True)
    keys = sorted({(e["task"], e["start"]) for e in reports[runs[0][0]]["episodes"]})
    table, firsts = [], []
    for task, start in keys:
        cols, outcome = [], {}
        for name, title in runs:
            ep = next((e for e in reports[name]["episodes"] if (e["task"], e["start"]) == (task, start)), None)
            if ep is None:
                continue
            f = frames(ROOT / name / ep["video"]); outcome[title] = ep
            cols.append((title, ep, f))
        n = max(len(f) for _, _, f in cols); video = []
        for i in range(n):
            tiles = []
            for title, ep, f in cols:
                fr = f[min(i, len(f) - 1)]
                done = i >= len(f) - 1
                sub = ("SUCCESS" if ep["success"] else "FAILED") + f" ({ep['steps']} steps)" if done else "running"
                tiles.append(np.concatenate([label(title, sub, ep["success"] if done else None), fr[:, :256], fr[:, 512:768]], 0))
            video.append(np.concatenate([np.pad(t, ((0, 0), (0, 4), (0, 0))) for t in tiles], 1))
        head = Image.new("RGB", (video[0].shape[1], 28), (0, 0, 0))
        ImageDraw.Draw(head).text((6, 8), f"task {task}: {cols[0][1]['instruction']}   (start {start}; top: live camera, bottom: goal image given to the policy)", fill=(255, 255, 255))
        head = np.asarray(head)
        writer = imageio.get_writer(str(out / f"task{task}_start{start}.mp4"), fps=10, codec="libx264", quality=7)
        for fr in video:
            writer.append_data(np.concatenate([head, fr], 0))
        writer.close()
        firsts.append(np.concatenate([head, video[-1]], 0))
        table.append((task, start, cols[0][1]["instruction"], {t: e["success"] for t, e in outcome.items()}))
    Image.fromarray(np.concatenate(firsts[:: max(1, len(firsts) // 10)], 0)).save(out / "final_frames.png")
    titles = [t for _, t in runs]
    lines = ["# Toy policy in the LIBERO-Goal simulator: comparison videos", "",
             "Same task and starting scene in every column. Top: live camera. Bottom: the goal image given to the policy.", "",
             "| Task | Start | Instruction | " + " | ".join(titles) + " |", "|" + "---|" * (3 + len(titles))]
    for task, start, instr, ok in table:
        lines.append(f"| {task} | {start} | [{instr}](task{task}_start{start}.mp4) | "
                     + " | ".join("✓" if ok.get(t) else "✗" if t in ok else "–" for t in titles) + " |")
    lines += ["", "| Condition | Successes |", "|---|---|"]
    for name, title in runs:
        r = reports[name]; lines.append(f"| {title} | {r['successes']}/{r['total']} |")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-len(runs) - 2:]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True, help="comma list of rollout_name=Column title, in column order")
    main(p.parse_args())
