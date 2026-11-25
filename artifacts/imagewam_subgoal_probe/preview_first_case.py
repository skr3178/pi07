"""First-case preview: input, real frames and ImageWAM predictions for one LIBERO case (main | wrist rows)."""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(ART), "data", "libero_10_subtask")
R = os.path.join(HERE, "results")
case_id = sys.argv[1] if len(sys.argv) > 1 else "drawer_close"
c = next(x for x in json.load(open(f"{ART}/bagel_subgoal_probe/cases.json")) if x["id"] == case_id)
ep_meta = pq.read_table(glob.glob(f"{DATA}/meta/episodes/*/*.parquet")[0]).to_pydict()
i = ep_meta["episode_index"].index(c["episode"])


def pair(frame):
    cams = []
    for cam in ("image", "image2"):
        ch, fi, t0 = (ep_meta[f"videos/observation.images.{cam}/{k}"][i] for k in ("chunk_index", "file_index", "from_timestamp"))
        with tempfile.TemporaryDirectory() as d:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{t0 + frame / 10:.3f}", "-i",
                            f"{DATA}/videos/observation.images.{cam}/chunk-{ch:03d}/file-{fi:03d}.mp4", "-frames:v", "1",
                            f"{d}/f.png"], check=True)
            cams.append(np.asarray(Image.open(f"{d}/f.png").convert("RGB").resize((224, 224), Image.BILINEAR)))
    return np.concatenate(cams, axis=1)


cols = [("input t", np.asarray(Image.open(f"{R}/{case_id}_input.png"))),
        ("real t+16 (1.6 s)", pair(c["frame"] + 16)),
        ("ImageWAM: task prompt", np.asarray(Image.open(f"{R}/{case_id}_task_seed0.png"))),
        ("ImageWAM: subtask prompt", np.asarray(Image.open(f"{R}/{case_id}_subtask_seed0.png"))),
        (f"real subtask end (t+{c['reference_frame'] - c['frame']})", pair(c["reference_frame"]))]
lab, cell = 20, 224
g = Image.new("RGB", (len(cols) * cell, 2 * cell + 2 * lab), "white")
d = ImageDraw.Draw(g)
for j, (name, im) in enumerate(cols):
    d.text((j * cell + 4, 4), name, fill="black")
    g.paste(Image.fromarray(im[:, :224]), (j * cell, lab))
    g.paste(Image.fromarray(im[:, 224:]), (j * cell, 2 * lab + cell))
d.text((4, cell + lab + 4), "wrist camera", fill="black")
g.save(f"{R}/{case_id}_preview.png")
print(f"{R}/{case_id}_preview.png")
