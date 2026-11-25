"""Per-task subtask order + contact sheets of segment start/end frames for libero_10_subtask."""
import collections
import glob
import os
import subprocess
import sys

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
fps = 10

ep_meta = pq.read_table(glob.glob(f"{root}/meta/episodes/*/*.parquet")[0]).to_pydict()
ep_video = {
    e: (c, f, t0)
    for e, c, f, t0 in zip(
        ep_meta["episode_index"],
        ep_meta["videos/observation.images.image/chunk_index"],
        ep_meta["videos/observation.images.image/file_index"],
        ep_meta["videos/observation.images.image/from_timestamp"],
    )
}
tasks = pq.read_table(f"{root}/meta/tasks.parquet").to_pydict()
task_name = dict(zip(tasks["task_index"], tasks["__index_level_0__"]))

# segments[ep] = list of [subtask_id, start_frame, end_frame]; ep_task[ep] = task
segments, ep_task = collections.defaultdict(list), {}
for f in sorted(glob.glob(f"{root}/data/*/*.parquet")):
    t = pq.read_table(f, columns=["episode_index", "frame_index", "task_index", "subtask_index"]).to_pydict()
    for e, fr, k, s in zip(t["episode_index"], t["frame_index"], t["task_index"], t["subtask_index"]):
        ep_task[e] = k
        seg = segments[e]
        if seg and seg[-1][0] == s:
            seg[-1][2] = max(seg[-1][2], fr)
        else:
            seg.append([s, fr, fr])

print("== subtask order per task (sequence: number of episodes) ==")
for k in sorted(task_name):
    orders = collections.Counter(tuple(s[0] for s in segments[e]) for e in segments if ep_task[e] == k)
    print(f"task {k}: {task_name[k]}")
    for seq, n in orders.most_common():
        print(f"    {' -> '.join(map(str, seq))}: {n}")

print("== segment length per subtask id (frames: min / median / max) ==")
by_id = collections.defaultdict(list)
for e, segs in segments.items():
    for s, a, b in segs:
        by_id[s].append((e, a, b))
for s in sorted(by_id):
    lens = sorted(b - a + 1 for _, a, b in by_id[s])
    tks = sorted({ep_task[e] for e, _, _ in by_id[s]})
    print(f"  id {s:2d}: n={len(lens):3d}  len {lens[0]}/{lens[len(lens)//2]}/{lens[-1]}  tasks {tks}")


def grab(e, frame, path):
    c, fi, t0 = ep_video[e]
    src = f"{root}/videos/observation.images.image/chunk-{c:03d}/file-{fi:03d}.mp4"
    ts = t0 + frame / fps
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{ts:.3f}", "-i", src, "-frames:v", "1", path],
        check=True,
    )
    return Image.open(path).convert("RGB")


thumb, per_id = 160, 3
ids = sorted(by_id)
for sheet_no, chunk in enumerate([ids[:8], ids[8:]]):
    label_w = 70
    sheet = Image.new("RGB", (label_w + per_id * 2 * thumb, len(chunk) * thumb), "white")
    draw = ImageDraw.Draw(sheet)
    for r, s in enumerate(chunk):
        picks = by_id[s][:: max(1, len(by_id[s]) // per_id)][:per_id]
        draw.text((5, r * thumb + 5), f"id {s}", fill="black")
        draw.text((5, r * thumb + 20), f"task {ep_task[picks[0][0]]}", fill="black")
        for j, (e, a, b) in enumerate(picks):
            for k, fr in enumerate((a, b)):
                im = grab(e, fr, f"{out}/id{s}_ep{e}_{'start' if k == 0 else 'end'}.png").resize((thumb, thumb))
                x = label_w + (2 * j + k) * thumb
                sheet.paste(im, (x, r * thumb))
                draw.text((x + 3, r * thumb + 3), ("S" if k == 0 else "E") + f" ep{e}", fill="yellow")
    p = f"{out}/sheet_{sheet_no}.png"
    sheet.save(p)
    print("wrote", p)
