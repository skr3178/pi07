"""Does ImageWAM's image path follow language when the image can't decide the outcome?

LIBERO-Goal: one kitchen, 10 instructions, all in ImageWAM's training data (its own 512 px renders,
yuanty/LIBERO-fastwam). From the first frame of an episode (arm at home, nothing moving yet) we ask for
each of the 10 instructions plus empty text:
  - single step (t+16), seeds 0-2
  - chain of 5 calls (t+80), seeds 0-1, robot state held at frame 0 (no future information)
Scoring: a nearest-reference classifier. Each prediction is compared with real frames of every task at
the same horizon (5 other episodes per task); the closest task is its "predicted task". Chance = 10%.
The same classifier on held-out real frames gives the ceiling: how separable the tasks are at that horizon.

Run with the ImageWAM venv; see README.md.
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
DATA = "/media/skr/storage/datasets/LIBERO-fastwam/lerobot_v30/libero_goal_no_noops_lerobot"
M = "/media/skr/storage/models/imagewam"
CKPT_DIR = f"{M}/imagewam_release/libero/flux2_klein_4b"
FLUX2_SRC = "/home/skr/Downloads/pi_07/ImageWAM/third_party/flux2"
OUT = os.path.join(HERE, "results")
S, HORIZON, CHAIN, STEPS = 224, 16, 5, 10
SEEDS, CHAIN_SEEDS, N_REF = (0, 1, 2), (0, 1), 5
START_TASKS = (0, 3, 9)  # start scenes taken from episodes of these tasks
os.makedirs(f"{OUT}/predictions", exist_ok=True)

# ---------- data ----------
info = json.load(open(f"{DATA}/meta/info.json"))
FPS = info["fps"]
ep_meta = pq.read_table(glob.glob(f"{DATA}/meta/episodes/*/*.parquet")[0]).to_pydict()
tasks = pq.read_table(f"{DATA}/meta/tasks.parquet").to_pandas()
task_text = {int(i): t for t, i in zip(tasks.index, tasks.task_index)}
frames = pq.read_table(f"{DATA}/data", columns=["episode_index", "frame_index", "task_index",
                                                 "observation.state"]).to_pandas()
ep_task = frames.groupby("episode_index").task_index.first().to_dict()
ep_len = dict(zip(ep_meta["episode_index"], ep_meta["length"]))


def grab(ep, frame, cam):
    i = ep_meta["episode_index"].index(ep)
    c, f, t0 = (ep_meta[f"videos/observation.images.{cam}/{k}"][i] for k in ("chunk_index", "file_index", "from_timestamp"))
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{t0 + frame / FPS:.4f}", "-i",
                        f"{DATA}/videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4", "-frames:v", "1",
                        f"{d}/f.png"], check=True)
        return np.asarray(Image.open(f"{d}/f.png").convert("RGB").resize((S, S), Image.BILINEAR))


def pair(ep, frame):  # main | wrist, uint8 [224, 448, 3], as ImageWAM's training video (horizontal concat)
    return np.concatenate([grab(ep, frame, "image"), grab(ep, frame, "wrist_image")], axis=1)


def state(ep, frame):
    row = frames[(frames.episode_index == ep) & (frames.frame_index == frame)]
    return np.asarray(row["observation.state"].iloc[0], dtype=np.float32)


rng = np.random.default_rng(0)
long_eps = {k: [e for e in sorted(ep_task) if ep_task[e] == k and ep_len[e] > HORIZON * CHAIN + 1] for k in range(10)}
start_eps = [int(rng.choice(long_eps[k])) for k in START_TASKS]
ref_eps = {k: [int(e) for e in rng.choice([e for e in long_eps[k] if e not in start_eps], N_REF, replace=False)]
           for k in range(10)}
jobs = [(e, 0) for e in start_eps] + [(e, h) for e in start_eps for h in (HORIZON, HORIZON * CHAIN)] + \
       [(e, h) for k in range(10) for e in ref_eps[k] for h in (HORIZON, HORIZON * CHAIN)]
t0 = time.time()
with cf.ThreadPoolExecutor(8) as ex:
    imgs = dict(zip(jobs, ex.map(lambda j: pair(*j), jobs)))
print(f"decoded {len(jobs)} frame pairs in {time.time() - t0:.0f} s; start episodes {start_eps}", flush=True)
for e in start_eps:
    Image.fromarray(imgs[(e, 0)]).save(f"{OUT}/start_ep{e}.png")

stats = json.load(open(f"{CKPT_DIR}/dataset_stats.json"))["state"]["default"]
s_min, s_max = np.array(stats["global_min"]), np.array(stats["global_max"])


def norm_state(s):  # min/max -> [-1, 1], clamp +-5 (SingleFieldLinearNormalizer)
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

conds = [(f"task{k}", task_text[k]) for k in range(10)] + [("empty", "")]
ctx = {}
for name, p in conds:
    h, m = model._encode_flux2_prompts([p])
    ctx[name] = (h.cpu(), m.cpu())
model.text_encoder = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

times = []


def predict(rgb, cond, st, seed):
    h, m = ctx[cond]
    x = torch.tensor(rgb).permute(2, 0, 1)[None].float() * (2 / 255) - 1
    t = time.time()
    out = model.infer_video_flux2(prompt=None, input_image=x, proprio=norm_state(st), context=h, context_mask=m,
                                  num_inference_steps=STEPS, seed=seed)
    torch.cuda.synchronize()
    times.append(time.time() - t)
    return ((out["image"].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()


preds = {}  # (start_ep, cond, seed, horizon) -> image
for e in start_eps:
    st0 = state(e, 0)
    for name, _ in conds:
        for s in SEEDS:
            img = predict(imgs[(e, 0)], name, st0, s)
            preds[(e, name, s, HORIZON)] = img
            if s in CHAIN_SEEDS:
                for _ in range(CHAIN - 1):
                    img = predict(img, name, st0, s)  # state held at frame 0: no future information
                preds[(e, name, s, HORIZON * CHAIN)] = img
    print(f"start episode {e} done; {len(times)} images, {np.mean(times):.2f} s each", flush=True)
for (e, name, s, h), img in preds.items():
    Image.fromarray(img).save(f"{OUT}/predictions/ep{e}_{name}_seed{s}_h{h}.png")


# ---------- scoring ----------
def feat(img):  # coarse layout: both cameras at 56 px, so arm pose and object positions dominate
    return np.concatenate([np.asarray(Image.fromarray(img[:, :S]).resize((56, 56), Image.BOX)),
                           np.asarray(Image.fromarray(img[:, S:]).resize((56, 56), Image.BOX))], 1).astype(np.float32) / 255


ref_feat = {(k, h): {e: feat(imgs[(e, h)]) for e in ref_eps[k]} for k in range(10) for h in (HORIZON, HORIZON * CHAIN)}


def classify(img, h, exclude=None):
    f = feat(img)
    d = [np.mean([np.mean((f - r) ** 2) for e, r in ref_feat[(k, h)].items() if e != exclude]) for k in range(10)]
    return int(np.argmin(d)), [round(float(x), 5) for x in d]


report = {"start_episodes": {e: task_text[ep_task[e]] for e in start_eps}, "reference_episodes": ref_eps,
          "dataset_fps_label": FPS, "horizons_steps": [HORIZON, HORIZON * CHAIN], "results": {}}
for h, seeds in ((HORIZON, SEEDS), (HORIZON * CHAIN, CHAIN_SEEDS)):
    # ceiling: real frames, leave-one-out
    ceil = [classify(imgs[(e, h)], h, exclude=e)[0] == k for k in range(10) for e in ref_eps[k]]
    rows, hits, distinct = [], [], []
    for e in start_eps:
        copy_class = classify(imgs[(e, 0)], h)[0]
        for s in seeds:
            got = [classify(preds[(e, f"task{k}", s, h)], h)[0] for k in range(10)]
            hits += [g == k for k, g in enumerate(got)]
            distinct.append(len(set(got)))
            rows.append({"start_episode": e, "seed": s, "predicted_task_per_instruction": got,
                         "empty_prompt_class": classify(preds[(e, "empty", s, h)], h)[0], "copy_class": copy_class})
    report["results"][f"h{h}"] = {
        "instruction_accuracy": round(float(np.mean(hits)), 3), "chance": 0.1,
        "ceiling_real_frames_leave_one_out": round(float(np.mean(ceil)), 3),
        "mean_distinct_classes_over_10_instructions": round(float(np.mean(distinct)), 2), "rows": rows}
    print(f"h={h}: instruction accuracy {np.mean(hits):.2f} (chance 0.10, real-frame ceiling {np.mean(ceil):.2f}); "
          f"distinct predicted tasks per 10 instructions {np.mean(distinct):.1f}", flush=True)


# instruction effect vs seed effect at each horizon (mean abs pixel difference, main camera, 0-255)
def mad(a, b):
    return float(np.mean(np.abs(a[:, :S].astype(np.float32) - b[:, :S].astype(np.float32))))


for h, seeds in ((HORIZON, SEEDS), (HORIZON * CHAIN, CHAIN_SEEDS)):
    across_instr = [mad(preds[(e, f"task{a}", s, h)], preds[(e, f"task{b}", s, h)])
                    for e in start_eps for s in seeds for a in range(10) for b in range(a + 1, 10)]
    across_seed = [mad(preds[(e, f"task{k}", a, h)], preds[(e, f"task{k}", b, h)])
                   for e in start_eps for k in range(10) for i, a in enumerate(seeds) for b in seeds[i + 1:]]
    report["results"][f"h{h}"]["main_mad_between_instructions_same_seed"] = round(float(np.mean(across_instr)), 2)
    report["results"][f"h{h}"]["main_mad_between_seeds_same_instruction"] = round(float(np.mean(across_seed)), 2)
report["seconds_per_image"] = round(float(np.mean(times)), 2)
report["peak_gpu_gb_without_qwen3"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)


# ---------- grids: per start scene, rows = instruction, main camera ----------
def grid(e, cam, path):
    sl, cell, lab, labw = (slice(0, S) if cam == "main" else slice(S, 2 * S)), 160, 16, 250
    hdr = ["seed0 t+16", "seed1 t+16", "seed2 t+16", f"chain s0 t+{HORIZON * CHAIN}", f"chain s1 t+{HORIZON * CHAIN}",
           f"real t+{HORIZON * CHAIN} (other ep)"]
    g = Image.new("RGB", (labw + len(hdr) * cell, lab + (len(conds) + 1) * cell), "white")
    d = ImageDraw.Draw(g)
    for j, t in enumerate(hdr):
        d.text((labw + j * cell + 3, 2), t, fill="black")
    d.text((3, lab + 3), f"START ep {e} ({cam})", fill="black")
    g.paste(Image.fromarray(imgs[(e, 0)][:, sl]).resize((cell, cell)), (labw, lab))
    for i, (name, text) in enumerate(conds, start=1):
        y = lab + i * cell
        d.text((3, y + 3), name, fill="black")
        for n, line in enumerate([text[k:k + 34] for k in range(0, len(text), 34)][:4]):
            d.text((3, y + 18 + 13 * n), line, fill="black")
        cells = [preds[(e, name, s, HORIZON)] for s in SEEDS] + [preds[(e, name, s, HORIZON * CHAIN)] for s in CHAIN_SEEDS]
        if name.startswith("task"):
            cells.append(imgs[(ref_eps[int(name[4:])][0], HORIZON * CHAIN)])
        for j, im in enumerate(cells):
            g.paste(Image.fromarray(im[:, sl]).resize((cell, cell)), (labw + j * cell, y))
    g.save(path)


for e in start_eps:
    for cam in ("main", "wrist"):
        grid(e, cam, f"{OUT}/start_ep{e}_{cam}.png")
print(json.dumps({k: v for k, v in report.items() if k != "results"}), flush=True)
print("done", flush=True)
