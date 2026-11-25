"""ImageWAM at LIBERO-10 subtask boundaries, prompted with our subtask wording (dataset.md table).

For one episode of each two-step task (tasks 0-8), at two boundaries:
  B0 = frame 0 (subtask A is about to start; subtask B could usually be done first instead)
  B1 = first frame of subtask B (A is done)
prompts: subtask A text, subtask B text, full task text, empty text.
Each (boundary, prompt): chains of 5 calls (80 steps) for seeds 0 and 1, state held at the boundary.
The first call of each chain is the single-step (16-step) prediction.

Scoring against the real continuation of the episode (which follows the aligned subtask):
  - aligned_wins: aligned-subtask prediction closer to the real frame than the other-subtask prediction
  - gap: d(other) - d(aligned), compared with the between-seed distance for the same prompt
At B0, a heading check: is the prediction closer to "A in progress" (real B0+80) or "B in progress"
(real B1+80, same episode)? Distances use both cameras at 56 px.
"""
import concurrent.futures as cf
import gc
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "libero_10_subtask")
M = "/media/skr/storage/models/imagewam"
CKPT_DIR = f"{M}/imagewam_release/libero/flux2_klein_4b"
FLUX2_SRC = "/home/skr/Downloads/pi_07/ImageWAM/third_party/flux2"
OUT = os.path.join(HERE, "results")
S, H, CHAIN, STEPS, SEEDS = 224, 16, 5, 10, (0, 1)
os.makedirs(f"{OUT}/predictions", exist_ok=True)

# Subtask wording from dataset.md (task -> [(id, text) of subtask A, of subtask B]).
SUBTASKS = {
    0: [(15, "turn on the stove"), (4, "put the moka pot on the stove")],
    1: [(8, "put the black bowl in the bottom drawer"), (2, "close the drawer")],
    2: [(14, "put the yellow and white mug in the microwave"), (1, "close the microwave")],
    3: [(5, "put the first moka pot on the stove"), (4, "put the moka pot on the stove")],
    4: [(9, "put the alphabet soup in the basket"), (12, "put the cream cheese box in the basket")],
    5: [(9, "put the alphabet soup in the basket"), (13, "put the tomato sauce in the basket")],
    6: [(12, "put the cream cheese box in the basket"), (11, "put the butter in the basket")],
    7: [(6, "put the white mug on the plate"), (7, "put the yellow and white mug on the right plate")],
    8: [(6, "put the white mug on the plate"), (3, "put the chocolate pudding to the right of the plate")],
}

# ---------- data ----------
FPS = json.load(open(f"{DATA}/meta/info.json"))["fps"]
ep_meta = pq.read_table(glob.glob(f"{DATA}/meta/episodes/*/*.parquet")[0]).to_pydict()
_t = pq.read_table(f"{DATA}/meta/tasks.parquet").to_pydict()
task_name = dict(zip(_t["task_index"], _t["__index_level_0__"]))
frames = pq.read_table(f"{DATA}/data", columns=["episode_index", "frame_index", "task_index", "subtask_index",
                                                 "observation.state"]).to_pandas()


def grab(ep, frame, cam):
    i = ep_meta["episode_index"].index(ep)
    c, f, t0 = (ep_meta[f"videos/observation.images.{cam}/{k}"][i] for k in ("chunk_index", "file_index", "from_timestamp"))
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{t0 + frame / FPS:.4f}", "-i",
                        f"{DATA}/videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4", "-frames:v", "1",
                        f"{d}/f.png"], check=True)
        return np.asarray(Image.open(f"{d}/f.png").convert("RGB").resize((S, S), Image.BILINEAR))


def pair(ep, frame):  # main | wrist (image2), uint8 [224, 448, 3]
    return np.concatenate([grab(ep, frame, "image"), grab(ep, frame, "image2")], axis=1)


rng = np.random.default_rng(0)
cases = []
for k, ((ida, ta), (idb, tb)) in SUBTASKS.items():
    eps = []
    for e, g in frames[frames.task_index == k].groupby("episode_index"):
        seq = g.sort_values("frame_index").subtask_index.to_numpy()
        starts = np.flatnonzero(seq == idb)
        if len(starts) and ida in seq[:starts[0]] and len(seq) > starts[0] + H * CHAIN:
            eps.append((e, int(starts[0]), len(seq)))
    e, b1, n = eps[int(rng.integers(len(eps)))]
    g = frames[frames.episode_index == e].set_index("frame_index")
    for name, b, aligned in (("B0", 0, 0), ("B1", b1, 1)):
        cases.append({"task": k, "task_text": task_name[k], "episode": int(e), "boundary": name, "frame": int(b),
                      "b1": b1, "length": n, "aligned": aligned, "texts": [ta, tb],
                      "state": np.asarray(g.loc[b, "observation.state"], dtype=np.float32)})

jobs = sorted({(c["episode"], min(c["frame"] + h, c["length"] - 1)) for c in cases for h in (0, H, H * CHAIN)} |
              {(c["episode"], min(c["b1"] + H * CHAIN, c["length"] - 1)) for c in cases})
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex:
    imgs = dict(zip(jobs, ex.map(lambda j: pair(*j), jobs)))
real = lambda c, h: imgs[(c["episode"], min(c["frame"] + h, c["length"] - 1))]  # noqa: E731
print(f"decoded {len(jobs)} frame pairs in {time.time() - t0:.0f} s; {len(cases)} boundaries", flush=True)

stats = json.load(open(f"{CKPT_DIR}/dataset_stats.json"))["state"]["default"]
s_min, s_max = np.array(stats["global_min"]), np.array(stats["global_max"])


def norm_state(s):
    return torch.tensor(np.clip(2 * (s - s_min) / (s_max - s_min) - 1, -5, 5), dtype=torch.float32)[None]


# ---------- model ----------
sys.path[:0] = ["/home/skr/Downloads/pi_07/ImageWAM/src", f"{FLUX2_SRC}/src", FLUX2_SRC]
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
from imagewam.runtime import create_imagewam_flux2_klein  # noqa: E402

t0 = time.time()
sched = {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
model = create_imagewam_flux2_klein(
    flux2_model_path=f"{M}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors",
    ae_model_path=f"{M}/flux2/FLUX.2-dev/ae.safetensors",
    flux2_src_path=FLUX2_SRC, variant="klein-base-4b", qwen3_model_spec="Qwen/Qwen3-4B", qwen_context_len=512,
    action_dit_config={"action_dim": 7, "hidden_dim": 1024, "num_heads": 24, "attn_head_dim": 128,
                       "num_layers_double": 5, "num_layers_single": 20, "mlp_ratio": 4.0,
                       "max_action_horizon": 64, "use_gradient_checkpointing": True},
    action_dit_pretrained_path=None, proprio_dim=8, load_text_encoder=False,
    video_scheduler=sched, action_scheduler=sched, loss={"lambda_video": 0.5, "lambda_action": 1.0},
    pack_proprio_after_text=True, flux2_lora_config={"enabled": False}, model_dtype=torch.bfloat16, device="cuda",
)
model.load_checkpoint(f"{CKPT_DIR}/model.pt")
model = model.to("cuda").eval()
print(f"model loaded in {time.time() - t0:.0f} s", flush=True)

prompts = sorted({t for c in cases for t in c["texts"]} | {c["task_text"] for c in cases} | {""})
ctx = {}
for p in prompts:
    h_, m_ = model._encode_flux2_prompts([p])
    ctx[p] = (h_.cpu(), m_.cpu())
model.text_encoder = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
times = []


def predict(rgb, prompt, st, seed):
    h_, m_ = ctx[prompt]
    x = torch.tensor(rgb).permute(2, 0, 1)[None].float() * (2 / 255) - 1
    t = time.time()
    out = model.infer_video_flux2(prompt=None, input_image=x, proprio=norm_state(st), context=h_, context_mask=m_,
                                  num_inference_steps=STEPS, seed=seed)
    torch.cuda.synchronize()
    times.append(time.time() - t)
    return ((out["image"].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()


KINDS = ("subtaskA", "subtaskB", "task", "empty")
preds = {}
for i, c in enumerate(cases):
    text = dict(zip(KINDS, c["texts"] + [c["task_text"], ""]))
    for kind in KINDS:
        for s in SEEDS:
            img = real(c, 0)
            for step in range(CHAIN):
                img = predict(img, text[kind], c["state"], s)
                if step == 0:
                    preds[(i, kind, s, H)] = img
            preds[(i, kind, s, H * CHAIN)] = img
    print(f"boundary {i + 1}/{len(cases)} done (task {c['task']} {c['boundary']}); {np.mean(times):.2f} s/image", flush=True)
for (i, kind, s, h), img in preds.items():
    c = cases[i]
    Image.fromarray(img).save(f"{OUT}/predictions/task{c['task']}_{c['boundary']}_{kind}_seed{s}_h{h}.png")


# ---------- scoring ----------
def feat(img):
    return np.concatenate([np.asarray(Image.fromarray(img[:, :S]).resize((56, 56), Image.BOX)),
                           np.asarray(Image.fromarray(img[:, S:]).resize((56, 56), Image.BOX))], 1).astype(np.float32) / 255


def dist(a, b):
    return float(np.mean((feat(a) - feat(b)) ** 2))


report = {"cases": [], "summary": {}}
for h in (H, H * CHAIN):
    wins, gaps, seed_d, instr_d, heading = [], [], [], [], []
    for i, c in enumerate(cases):
        al, ot = ("subtaskA", "subtaskB") if c["aligned"] == 0 else ("subtaskB", "subtaskA")
        r = real(c, h)
        row = {"task": c["task"], "boundary": c["boundary"], "episode": c["episode"], "frame": c["frame"], "h": h,
               "aligned_text": c["texts"][c["aligned"]], "other_text": c["texts"][1 - c["aligned"]],
               "d_real": {k: round(float(np.mean([dist(preds[(i, k, s, h)], r) for s in SEEDS])), 5) for k in KINDS},
               "d_real_copy": round(dist(real(c, 0), r), 5)}
        for s in SEEDS:
            wins.append(dist(preds[(i, al, s, h)], r) < dist(preds[(i, ot, s, h)], r))
            instr_d.append(dist(preds[(i, al, s, h)], preds[(i, ot, s, h)]))
        gaps.append(row["d_real"][ot] - row["d_real"][al])
        seed_d += [dist(preds[(i, k, 0, h)], preds[(i, k, 1, h)]) for k in (al, ot)]
        if c["boundary"] == "B0" and h == H * CHAIN:  # A in progress vs B in progress (same episode)
            ra, rb = real(c, h), imgs[(c["episode"], min(c["b1"] + H * CHAIN, c["length"] - 1))]
            row["heading"] = {f"{k}_seed{s}": "B" if dist(preds[(i, k, s, h)], rb) < dist(preds[(i, k, s, h)], ra) else "A"
                              for k in ("subtaskA", "subtaskB") for s in SEEDS}
            heading += [row["heading"][f"subtaskA_seed{s}"] == "A" for s in SEEDS] + \
                       [row["heading"][f"subtaskB_seed{s}"] == "B" for s in SEEDS]
        report["cases"].append(row)
    summ = {"aligned_closer_to_real_than_other": round(float(np.mean(wins)), 3),
            "mean_gap_other_minus_aligned": round(float(np.mean(gaps)), 5),
            "mean_distance_aligned_vs_other_same_seed": round(float(np.mean(instr_d)), 5),
            "mean_distance_between_seeds_same_prompt": round(float(np.mean(seed_d)), 5)}
    for b in ("B0", "B1"):
        idx = [j for j, c in enumerate(cases) if c["boundary"] == b]
        summ[f"{b}_aligned_closer"] = round(float(np.mean([wins[2 * j + s] for j in idx for s in range(2)])), 3)
    if heading:
        summ["B0_heading_matches_prompt"] = round(float(np.mean(heading)), 3)
    report["summary"][f"h{h}"] = summ
    print(f"h={h}: {json.dumps(summ)}", flush=True)
report["seconds_per_image"] = round(float(np.mean(times)), 2)
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)


# ---------- grids: one per boundary type, rows = task x prompt ----------
def grid(boundary, cam):
    sl, cell, lab, labw = (slice(0, S) if cam == "main" else slice(S, 2 * S)), 150, 16, 260
    hdr = ["start", "seed0 t+16", "seed1 t+16", "chain s0 t+80", "chain s1 t+80", "real t+16", "real t+80"]
    sel = [(i, c) for i, c in enumerate(cases) if c["boundary"] == boundary]
    g = Image.new("RGB", (labw + len(hdr) * cell, lab + len(sel) * 3 * cell), "white")
    d = ImageDraw.Draw(g)
    for j, t in enumerate(hdr):
        d.text((labw + j * cell + 3, 2), t, fill="black")
    for n, (i, c) in enumerate(sel):
        al = "subtaskA" if c["aligned"] == 0 else "subtaskB"
        ot = "subtaskB" if al == "subtaskA" else "subtaskA"
        for r, kind in enumerate((al, ot, "empty")):
            y = lab + (3 * n + r) * cell
            text = {"subtaskA": c["texts"][0], "subtaskB": c["texts"][1], "empty": ""}[kind]
            tag = {al: "ALIGNED", ot: "OTHER", "empty": "EMPTY"}[kind]
            d.text((3, y + 3), f"task {c['task']} {boundary} t={c['frame']} | {tag}", fill="black")
            for q, line in enumerate([text[k:k + 36] for k in range(0, len(text), 36)][:3]):
                d.text((3, y + 18 + 13 * q), line, fill="black")
            cells = [real(c, 0)] + [preds[(i, kind, s, H)] for s in SEEDS] + [preds[(i, kind, s, H * CHAIN)] for s in SEEDS]
            if r == 0:
                cells += [real(c, H), real(c, H * CHAIN)]
            for j, im in enumerate(cells):
                g.paste(Image.fromarray(im[:, sl]).resize((cell, cell)), (labw + j * cell, y))
    g.save(f"{OUT}/{boundary}_{cam}.png")


for b in ("B0", "B1"):
    for cam in ("main", "wrist"):
        grid(b, cam)
print("done", flush=True)
