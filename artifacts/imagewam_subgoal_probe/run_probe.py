"""ImageWAM FLUX.2 4B LIBERO: can its image path serve as a pi0.7-style subgoal generator?

Uses only the image path (`infer_video_flux2`): current frame (main | wrist, 224x448) + prompt + state
-> the frame 16 steps (1.6 s at 10 fps) later, which is the training target (`endpoint_frames_only`).

  A. LIBERO single step: 3 cases x {task, subtask} prompt x 3 seeds, scored against the real t+16 frame
     and against a "copy the current frame" baseline.
  B. LIBERO chained: feed each prediction back in until near the subtask end (oracle dataset state each step).
  C. pi0.7 food steps (out of distribution): one real-robot camera, no wrist view, no state.

Run with the ImageWAM venv; see README.md for the command.
"""
import gc
import glob
import json
import math
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
ART = os.path.dirname(HERE)
DATA = os.path.join(os.path.dirname(ART), "data", "libero_10_subtask")
M = "/media/skr/storage/models/imagewam"
CKPT_DIR = f"{M}/imagewam_release/libero/flux2_klein_4b"
FLUX2_SRC = "/home/skr/Downloads/pi_07/ImageWAM/third_party/flux2"
OUT = os.path.join(HERE, "results")
H, W, HORIZON, FPS, STEPS = 224, 224, 16, 10, 10
SEEDS = (0, 1, 2)
os.makedirs(OUT, exist_ok=True)


# ---------- LIBERO inputs ----------
ep_meta = pq.read_table(glob.glob(f"{DATA}/meta/episodes/*/*.parquet")[0]).to_pydict()
tasks = pq.read_table(f"{DATA}/meta/tasks.parquet").to_pydict()
task_name = dict(zip(tasks["task_index"], tasks["__index_level_0__"]))
frames = pq.read_table(f"{DATA}/data", columns=[
    "episode_index", "frame_index", "task_index", "observation.state"]).to_pandas()


def grab(ep, frame, cam):
    i = ep_meta["episode_index"].index(ep)
    c = ep_meta[f"videos/observation.images.{cam}/chunk_index"][i]
    f = ep_meta[f"videos/observation.images.{cam}/file_index"][i]
    t0 = ep_meta[f"videos/observation.images.{cam}/from_timestamp"][i]
    src = f"{DATA}/videos/observation.images.{cam}/chunk-{c:03d}/file-{f:03d}.mp4"
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/f.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{t0 + frame / FPS:.3f}",
                        "-i", src, "-frames:v", "1", p], check=True)
        return np.asarray(Image.open(p).convert("RGB").resize((W, H), Image.BILINEAR))


def libero_pair(ep, frame):
    """main | wrist side by side, uint8 [224, 448, 3]; image = main camera, image2 = wrist."""
    return np.concatenate([grab(ep, frame, "image"), grab(ep, frame, "image2")], axis=1)


def libero_state(ep, frame):
    row = frames[(frames.episode_index == ep) & (frames.frame_index == frame)]
    return np.asarray(row["observation.state"].iloc[0], dtype=np.float32), int(row["task_index"].iloc[0])


stats = json.load(open(f"{CKPT_DIR}/dataset_stats.json"))["state"]["default"]
s_min, s_max = np.array(stats["global_min"]), np.array(stats["global_max"])


def norm_state(s):  # SingleFieldLinearNormalizer, mode min/max -> [-1, 1], clamp +-5
    return torch.tensor(np.clip(2 * (s - s_min) / (s_max - s_min) - 1, -5, 5), dtype=torch.float32)[None]


def to_input(rgb):  # uint8 HWC -> [1, 3, H, W] in [-1, 1], as eval_libero_single._obs_to_model_input
    return torch.tensor(rgb).permute(2, 0, 1)[None].float() * (2 / 255) - 1


def to_uint8(img):  # [3, H, W] in [-1, 1] -> uint8 HWC
    return ((img.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()


def psnr(a, b, mask=None):
    d = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    mse = d[mask].mean() if mask is not None else d.mean()
    return float(10 * math.log10(255 ** 2 / max(mse, 1e-10)))


libero_cases = json.load(open(f"{ART}/bagel_subgoal_probe/cases.json"))
for c in libero_cases:
    c["state"], c["task_index"] = libero_state(c["episode"], c["frame"])
    c["task"] = task_name[c["task_index"]]
    c["current_rgb"] = libero_pair(c["episode"], c["frame"])
    c["gt16_rgb"] = libero_pair(c["episode"], c["frame"] + HORIZON)
    c["ref_rgb"] = libero_pair(c["episode"], c["reference_frame"])
    c["chain_steps"] = (c["reference_frame"] - c["frame"]) // HORIZON
    c["chain_gt_rgb"] = libero_pair(c["episode"], c["frame"] + HORIZON * c["chain_steps"])
    c["chain_states"] = [libero_state(c["episode"], c["frame"] + HORIZON * k)[0] for k in range(c["chain_steps"])]
    c["subtask_prompt"] = c["subtask"].split(",")[0].rstrip(".").lower()
    Image.fromarray(c["current_rgb"]).save(f"{OUT}/{c['id']}_input.png")

food_cases = [x for x in json.load(open(f"{ART}/pi07_reference_subgoals/cases.json")) if x["prompt_variant"] == "bare"]


def center_crop_resize(img):  # as eval_libero_single._center_crop_resize
    s = max(W / img.width, H / img.height)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BILINEAR)
    l, t = (img.width - W) // 2, (img.height - H) // 2
    return np.asarray(img.crop((l, t, l + W, t + H)))


for c in food_cases:
    main = center_crop_resize(Image.open(c["current"]).convert("RGB"))
    c["current_rgb"] = np.concatenate([main, np.full_like(main, 128)], axis=1)  # no wrist view: mid-gray
    c["ref_rgb"] = center_crop_resize(Image.open(c["reference"]).convert("RGB"))

# ---------- model ----------
sys.path[:0] = ["/home/skr/Downloads/pi_07/ImageWAM/src", f"{FLUX2_SRC}/src", FLUX2_SRC]
import logging  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
from imagewam.runtime import create_imagewam_flux2_klein  # noqa: E402

t_load = time.time()
cfg_sched = {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
model = create_imagewam_flux2_klein(
    flux2_model_path=f"{M}/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors",
    ae_model_path=f"{M}/flux2/FLUX.2-dev/ae.safetensors",
    flux2_src_path=FLUX2_SRC, variant="klein-base-4b", qwen3_model_spec="Qwen/Qwen3-4B", qwen_context_len=512,
    action_dit_config={"action_dim": 7, "hidden_dim": 1024, "num_heads": 24, "attn_head_dim": 128,
                       "num_layers_double": 5, "num_layers_single": 20, "mlp_ratio": 4.0,
                       "max_action_horizon": 64, "use_gradient_checkpointing": True},
    action_dit_pretrained_path=None, proprio_dim=8, load_text_encoder=False,
    video_scheduler=cfg_sched, action_scheduler=cfg_sched, loss={"lambda_video": 0.5, "lambda_action": 1.0},
    pack_proprio_after_text=True, flux2_lora_config={"enabled": False},
    model_dtype=torch.bfloat16, device="cuda",
)
model.load_checkpoint(f"{CKPT_DIR}/model.pt")
model = model.to("cuda").eval()
print(f"model loaded in {time.time() - t_load:.0f} s; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

# Encode every prompt once with Qwen3 (lazy-loaded by the repo), then free it.
prompts = sorted({c["task"] for c in libero_cases} | {c["subtask_prompt"] for c in libero_cases}
                 | {c["subtask"] for c in food_cases})
ctx = {}
for p in prompts:
    h, m = model._encode_flux2_prompts([p])
    ctx[p] = (h.cpu(), m.cpu())
print(f"encoded {len(prompts)} prompts; peak GPU with Qwen3 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
model.text_encoder = None
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

times = []


def predict(rgb, prompt, state, seed):
    h, m = ctx[prompt]
    t = time.time()
    out = model.infer_video_flux2(prompt=None, input_image=to_input(rgb), proprio=state, context=h, context_mask=m,
                                  num_inference_steps=STEPS, seed=seed)
    torch.cuda.synchronize()
    times.append(time.time() - t)
    return to_uint8(out["image"])


# ---------- A. LIBERO single step ----------
report = {"single_step": [], "chained": [], "pi07_food": []}
for c in libero_cases:
    gt, cur = c["gt16_rgb"], c["current_rgb"]
    moving = np.abs(gt.astype(int) - cur.astype(int)).max(-1) > 20  # pixels that change over the 1.6 s
    row = {"id": c["id"], "episode": c["episode"], "frame": c["frame"], "task": c["task"],
           "subtask_prompt": c["subtask_prompt"], "moving_pixel_fraction": round(float(moving.mean()), 4),
           "copy_baseline": {"psnr_main": psnr(cur[:, :W], gt[:, :W]), "psnr_wrist": psnr(cur[:, W:], gt[:, W:]),
                             "psnr_moving": psnr(cur, gt, moving)}}
    for kind, prompt in (("task", c["task"]), ("subtask", c["subtask_prompt"])):
        runs = []
        for s in SEEDS:
            pred = predict(cur, prompt, norm_state(c["state"]), s)
            Image.fromarray(pred).save(f"{OUT}/{c['id']}_{kind}_seed{s}.png")
            runs.append({"seed": s, "psnr_main": psnr(pred[:, :W], gt[:, :W]), "psnr_wrist": psnr(pred[:, W:], gt[:, W:]),
                         "psnr_moving": psnr(pred, gt, moving)})
            if s == 0:
                c[f"pred_{kind}"] = pred
        row[kind] = {"prompt": prompt, "runs": runs,
                     **{f"mean_{k}": round(float(np.mean([r[k] for r in runs])), 2)
                        for k in ("psnr_main", "psnr_wrist", "psnr_moving")}}
    report["single_step"].append(row)
    print(json.dumps({k: row[k] for k in ("id", "copy_baseline")}),
          {k: {m: row[k][m] for m in row[k] if m.startswith("mean")} for k in ("task", "subtask")}, flush=True)

# ---------- B. LIBERO chained toward the subtask end ----------
for c in libero_cases:
    for kind, prompt in (("task", c["task"]), ("subtask", c["subtask_prompt"])):
        img = c["current_rgb"]
        for k in range(c["chain_steps"]):
            img = predict(img, prompt, norm_state(c["chain_states"][k]), 0)
        c[f"chain_{kind}"] = img
        Image.fromarray(img).save(f"{OUT}/{c['id']}_chain_{kind}.png")
        report["chained"].append({
            "id": c["id"], "prompt_kind": kind, "steps": c["chain_steps"],
            "end_frame": c["frame"] + HORIZON * c["chain_steps"], "reference_frame": c["reference_frame"],
            "psnr_main_vs_real_end": psnr(img[:, :W], c["chain_gt_rgb"][:, :W]),
            "psnr_main_vs_reference": psnr(img[:, :W], c["ref_rgb"][:, :W]),
            "copy_baseline_psnr_main_vs_reference": psnr(c["current_rgb"][:, :W], c["ref_rgb"][:, :W]),
        })

# ---------- C. pi0.7 food steps (out of distribution) ----------
mean_state = torch.zeros(1, 8)  # no state: mid-range of the normalized [-1, 1] state
for c in food_cases:
    c["pred"] = predict(c["current_rgb"], c["subtask"], mean_state, 0)
    Image.fromarray(c["pred"]).save(f"{OUT}/{c['id']}.png")
    report["pi07_food"].append({"id": c["id"], "subtask": c["subtask"]})

report["seconds_per_image"] = {"mean": round(float(np.mean(times)), 2), "min": round(min(times), 2), "n": len(times)}
report["peak_gpu_gb_without_qwen3"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)


# ---------- grids ----------
def grid(rows, header, path, cell=W):
    lab = 18
    g = Image.new("RGB", (len(header) * cell, len(rows) * (cell + lab) + lab), "white")
    d = ImageDraw.Draw(g)
    for j, h in enumerate(header):
        d.text((j * cell + 4, 3), h, fill="black")
    for i, (name, cells) in enumerate(rows):
        y = lab + i * (cell + lab)
        d.text((4, y + 2), name, fill="black")
        for j, im in enumerate(cells):
            g.paste(Image.fromarray(im).resize((cell, cell)), (j * cell, y + lab))
    g.save(path)


hdr = ["current t", "real t+16", "pred: task", "pred: subtask", "chain: task", "chain: subtask", "real chain end",
       "subtask end"]
for half, sl in (("main", slice(0, W)), ("wrist", slice(W, 2 * W))):
    grid([(f"{c['id']} (ep {c['episode']}, t={c['frame']}, chain {c['chain_steps']}x16)",
           [c[k][:, sl] for k in ("current_rgb", "gt16_rgb", "pred_task", "pred_subtask", "chain_task",
                                  "chain_subtask", "chain_gt_rgb", "ref_rgb")]) for c in libero_cases],
         hdr, f"{OUT}/libero_{half}_camera.png")
grid([(f"{c['id']}: {c['subtask']}", [c["current_rgb"][:, :W], c["pred"][:, :W], c["ref_rgb"]]) for c in food_cases],
     ["current", "ImageWAM (+1.6 s)", "pi0.7 subgoal"], f"{OUT}/pi07_food.png")

json.dump(report, open(f"{OUT}/report.json", "w"), indent=1)
print(json.dumps({k: report[k] for k in ("seconds_per_image", "peak_gpu_gb_without_qwen3")}), flush=True)
print(json.dumps(report["chained"], indent=1))
