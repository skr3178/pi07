"""Matched-seed prompt controls and frozen-state chaining for the released ImageWAM.

Uses existing local weights only. `prepare` decodes inputs, `run` performs inference,
and `analyze` regenerates metrics/grids without loading the model. Counterfactual
instructions have no ground-truth continuation; pixel differences are sensitivity,
not instruction-following success. All inference uses current/initial state only.
"""
import argparse
import gc
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
WORKSPACE = PROJECT.parent
DATA = PROJECT / "data/libero_10_subtask"
PREVIOUS = HERE.parent / "imagewam_subgoal_probe"
MODEL_ROOT = Path("/media/skr/storage/models/imagewam")
CHECKPOINT = MODEL_ROOT / "imagewam_release/libero/flux2_klein_4b"
FLUX_SRC = WORKSPACE / "ImageWAM/third_party/flux2"
OUT = HERE / "results"
SEEDS = [0, 1, 2]
KINDS = ["task", "aligned", "paraphrase", "counter", "empty"]
W = H = 224
HORIZON = 16

PROMPTS = {
    "drawer_close": {
        "aligned": "close the bottom drawer of the cabinet",
        "paraphrase": "push the bottom drawer shut",
        "counter": "pull the bottom drawer of the cabinet further open",
    },
    "microwave_close": {
        "aligned": "close the microwave door",
        "paraphrase": "push the microwave door shut",
        "counter": "pull the microwave door further open",
    },
    "pudding_place": {
        "aligned": "place the chocolate pudding to the right of the plate",
        "paraphrase": "put the chocolate pudding on the right side of the plate",
        "counter": "place the chocolate pudding to the left of the plate",
    },
}


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def make_grid(rows, headers, path):
    row_label, col_label = 30, 28
    canvas = Image.new("RGB", (len(headers) * W, col_label + len(rows) * (H + row_label)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(headers):
        draw.text((col * W + 5, 6), name, fill="black")
    for row, (name, images) in enumerate(rows):
        y = col_label + row * (H + row_label)
        draw.text((5, y + 5), name, fill="black")
        for col, img in enumerate(images):
            canvas.paste(Image.fromarray(img), (col * W, y + row_label))
    canvas.save(path)


def prepare():
    import pyarrow.parquet as pq

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inputs").mkdir(exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)
    info = json.loads((DATA / "meta/info.json").read_text())
    fps = float(info["fps"])
    meta = pq.read_table(DATA / "meta/episodes").to_pylist()
    meta = {int(row["episode_index"]): row for row in meta}
    tasks = pq.read_table(DATA / "meta/tasks.parquet").to_pylist()
    tasks = {int(row["task_index"]): row["__index_level_0__"] for row in tasks}
    frames = pq.read_table(DATA / "data", columns=["episode_index", "frame_index", "task_index", "subtask_index", "observation.state"]).to_pandas()
    originals = json.loads((HERE.parent / "bagel_subgoal_probe/cases.json").read_text())

    def pair(episode, frame):
        entry = meta[episode]
        images = []
        for camera in ["image", "image2"]:
            prefix = f"videos/observation.images.{camera}/"
            chunk, file = (int(entry[prefix + key]) for key in ["chunk_index", "file_index"])
            timestamp = float(entry[prefix + "from_timestamp"]) + frame / fps
            video = DATA / f"videos/observation.images.{camera}/chunk-{chunk:03d}/file-{file:03d}.mp4"
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "frame.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-c:v", "libdav1d", "-ss", f"{timestamp:.6f}", "-i", str(video), "-frames:v", "1", str(target)], check=True)
                images.append(np.asarray(Image.open(target).convert("RGB").resize((W, H), Image.Resampling.BILINEAR)))
        return np.concatenate(images, axis=1)

    cases = []
    for original in originals:
        episode = int(original["episode"])
        episode_rows = frames[frames.episode_index == episode].sort_values("frame_index")
        for stage, frame in [("original", original["frame"]), ("late", original["reference_frame"] - 32)]:
            row = episode_rows[episode_rows.frame_index == frame].iloc[0]
            following = episode_rows[(episode_rows.frame_index > frame) & (episode_rows.subtask_index != row.subtask_index)]
            segment_end = int(following.iloc[0].frame_index) - 1 if len(following) else int(episode_rows.frame_index.max())
            assert frame + HORIZON <= segment_end
            case_id = original["id"] + "_" + stage
            prompts = {"task": tasks[int(row.task_index)], **PROMPTS[original["id"]], "empty": ""}
            paths = {}
            chain_steps = (segment_end - frame) // HORIZON
            for name, idx in {"current": frame, "gt16": frame + HORIZON, "endpoint": segment_end, "chain_gt": frame + chain_steps * HORIZON}.items():
                path = OUT / "inputs" / f"{case_id}_{name}.png"
                Image.fromarray(pair(episode, idx)).save(path)
                paths[name] = str(path.relative_to(HERE))
            cases.append({"id": case_id, "base_id": original["id"], "stage": stage, "episode": episode, "frame": int(frame), "subtask_id": int(row.subtask_index), "segment_end": segment_end, "chain_steps": chain_steps, "chain_end": frame + chain_steps * HORIZON, "state": np.asarray(row["observation.state"]).tolist(), "prompts": prompts, "images": paths})

    manifest = {"created_date": "2026-10-29", "seeds": SEEDS, "horizon_dataset_steps": HORIZON, "dataset_fps": fps, "denoising_steps": 10, "cases": cases, "limitations": ["Six states from three demonstration episodes; not a generalization benchmark.", "Counterfactual prompts lack matched ground truth and may be outside the checkpoint's language training distribution.", "Empty means an empty user string passed through the usual chat template, not a zero text embedding.", "Frozen-state chains are an inconsistent-state stress test, not a robot dynamics model or a deployment method."]}
    save_json(HERE / "manifest.json", manifest)
    for camera, bounds in [("main", slice(0, W)), ("wrist", slice(W, 2 * W))]:
        rows = [(f"{c['id']} ep={c['episode']} t={c['frame']}", [read_image(HERE / c["images"][k])[:, bounds] for k in ["current", "gt16", "endpoint"]]) for c in cases]
        make_grid(rows, ["current", "real t+16", "real segment end"], OUT / f"inputs_{camera}.png")
    print(f"Prepared {len(cases)} states and {len(cases) * len(KINDS) * len(SEEDS)} single-step predictions", flush=True)


def run():
    # Prevent accidental downloads or writes into the installed environment.
    os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(HERE / "runtime_cache"))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU access outside the filesystem sandbox")
    sys.path[:0] = [str(WORKSPACE / "ImageWAM/src"), str(FLUX_SRC / "src"), str(FLUX_SRC)]
    from imagewam.runtime import create_imagewam_flux2_klein

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    manifest = json.loads((HERE / "manifest.json").read_text())
    sched = {"train_shift": 5.0, "infer_shift": 5.0, "num_train_timesteps": 1000}
    started = time.perf_counter()
    model = create_imagewam_flux2_klein(
        flux2_model_path=str(MODEL_ROOT / "flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"),
        ae_model_path=str(MODEL_ROOT / "flux2/FLUX.2-dev/ae.safetensors"),
        flux2_src_path=str(FLUX_SRC), variant="klein-base-4b", qwen3_model_spec="Qwen/Qwen3-4B", qwen_context_len=512,
        action_dit_config={"action_dim": 7, "hidden_dim": 1024, "num_heads": 24, "attn_head_dim": 128, "num_layers_double": 5, "num_layers_single": 20, "mlp_ratio": 4.0, "max_action_horizon": 64, "use_gradient_checkpointing": True},
        action_dit_pretrained_path=None, proprio_dim=8, load_text_encoder=False,
        video_scheduler=sched, action_scheduler=sched, loss={"lambda_video": 0.5, "lambda_action": 1.0},
        pack_proprio_after_text=True, flux2_lora_config={"enabled": False}, model_dtype=torch.bfloat16, device="cuda")
    model.load_checkpoint(str(CHECKPOINT / "model.pt"))
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    print(f"Loaded model in {load_seconds:.1f} s", flush=True)
    contexts = {}
    for prompt in sorted({p for c in manifest["cases"] for p in c["prompts"].values()}):
        h, m = model._encode_flux2_prompts([prompt])
        contexts[prompt] = (h.cpu(), m.cpu())
    peak_with_encoder = torch.cuda.max_memory_allocated()
    del h, m
    model.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"Cached {len(contexts)} prompts; released Qwen3", flush=True)
    # A CPU cache makes the exact encodings available without another encoder load.
    torch.save(contexts, HERE / "prompt_embeddings.pt")
    stats = json.loads((CHECKPOINT / "dataset_stats.json").read_text())["state"]["default"]
    low, high = np.array(stats["global_min"]), np.array(stats["global_max"])
    assert np.all(high - low >= 1e-4), "Use the normalizer's constant-dimension rule if this changes"
    timings = []

    def predict(rgb, prompt, state, seed):
        x = torch.tensor(rgb).permute(2, 0, 1)[None].float() * (2 / 255) - 1
        norm = torch.tensor(np.clip(2 * (np.array(state) - low) / (high - low) - 1, -5, 5), dtype=torch.float32)[None]
        h, m = contexts[prompt]
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = model.infer_video_flux2(prompt=None, input_image=x, proprio=norm, context=h, context_mask=m, num_inference_steps=10, seed=seed)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        return ((out["image"].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()

    for c in manifest["cases"]:
        current = read_image(HERE / c["images"]["current"])
        for kind in KINDS:
            for seed in SEEDS:
                pred = predict(current, c["prompts"][kind], c["state"], seed)
                Image.fromarray(pred).save(OUT / "predictions" / f"{c['id']}_{kind}_seed{seed}.png")
        print(f"Finished single-step controls: {c['id']} ({len(timings)} generations)", flush=True)

    first = manifest["cases"][0]
    repeated = predict(read_image(HERE / first["images"]["current"]), first["prompts"]["aligned"], first["state"], 0)
    Image.fromarray(repeated).save(OUT / "repeat_aligned_seed0.png")
    for c in manifest["cases"]:
        if c["stage"] != "original":
            continue
        # Future observations/states never enter inference. All steps use state(t0).
        img = read_image(HERE / c["images"]["current"])
        for step in range(c["chain_steps"]):
            img = predict(img, c["prompts"]["aligned"], c["state"], 0)
            Image.fromarray(img).save(OUT / "predictions" / f"{c['id']}_frozen_chain_step{step + 1}.png")
        print(f"Finished frozen-state chain: {c['id']}, {c['chain_steps']} steps", flush=True)

    def git_commit(path):
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()

    save_json(OUT / "runtime.json", {
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "imagewam_commit": git_commit(WORKSPACE / "ImageWAM"), "flux2_commit": git_commit(FLUX_SRC),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint": str(CHECKPOINT / "model.pt"), "checkpoint_size_bytes": (CHECKPOINT / "model.pt").stat().st_size,
        "load_seconds": load_seconds, "generation_count": len(timings),
        "seconds_per_image_mean": float(np.mean(timings)), "seconds_per_image_median": float(np.median(timings)),
        "seconds_per_image_p95": float(np.percentile(timings, 95)),
        "peak_allocated_bytes_with_text_encoder": peak_with_encoder,
        "peak_allocated_bytes_without_text_encoder": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes_without_text_encoder": torch.cuda.max_memory_reserved(),
        "prompt_tokens": {p: int(m.sum()) for p, (_, m) in contexts.items()},
        "offline_model_loading": True, "future_states_used_for_inference": False})
    print("Inference finished", flush=True)


def image_metrics(pred, target, mask=None):
    delta = pred.astype(np.float32) - target.astype(np.float32)
    if mask is not None:
        delta = delta[mask]
    if delta.size == 0:
        return {"mae_255": None, "psnr_db": None}
    mse = float(np.mean(delta ** 2))
    return {"mae_255": float(np.mean(np.abs(delta))), "psnr_db": float(10 * np.log10(255 ** 2 / max(mse, 1e-10)))}


def stats(values):
    values = [x for x in values if x is not None]
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values)), "n": len(values)} if values else None


def analyze():
    manifest = json.loads((HERE / "manifest.json").read_text())
    report = {"cases": [], "chains": [], "limitations": manifest["limitations"]}
    grids = {camera: [] for camera in ["main", "wrist"]}
    chains = {camera: [] for camera in ["main", "wrist"]}
    for c in manifest["cases"]:
        cur, gt = (read_image(HERE / c["images"][k]) for k in ["current", "gt16"])
        predictions = {kind: {seed: read_image(OUT / "predictions" / f"{c['id']}_{kind}_seed{seed}.png") for seed in SEEDS} for kind in KINDS}
        result = {"id": c["id"], "prompts": c["prompts"], "cameras": {}}
        for camera, sl in [("main", slice(0, W)), ("wrist", slice(W, 2 * W))]:
            target, current = gt[:, sl], cur[:, sl]
            moving = np.abs(target.astype(int) - current.astype(int)).max(-1) > 20
            masks = {"full": None, "moving": moving, "static": ~moving}
            metrics = {"moving_pixel_fraction": float(moving.mean()), "copy_baseline": {}, "gt_scores": {}, "sensitivity": {}}
            for name, mask in masks.items():
                metrics["copy_baseline"][name] = image_metrics(current, target, mask)
            for kind in KINDS:
                if kind == "counter":
                    continue  # No counterfactual reference exists; avoid misleading accuracy scores.
                metrics["gt_scores"][kind] = {}
                for name, mask in masks.items():
                    scores = [image_metrics(predictions[kind][seed][:, sl], target, mask) for seed in SEEDS]
                    metrics["gt_scores"][kind][name] = {metric: stats([s[metric] for s in scores]) for metric in ["mae_255", "psnr_db"]}
            for other in ["counter", "paraphrase", "empty", "task"]:
                metrics["sensitivity"][f"aligned_vs_{other}"] = {name: stats([image_metrics(predictions["aligned"][seed][:, sl], predictions[other][seed][:, sl], mask)["mae_255"] for seed in SEEDS]) for name, mask in masks.items()}
            metrics["sensitivity"]["aligned_between_seeds"] = {name: stats([image_metrics(predictions["aligned"][a][:, sl], predictions["aligned"][b][:, sl], mask)["mae_255"] for a, b in itertools.combinations(SEEDS, 2)]) for name, mask in masks.items()}
            result["cameras"][camera] = metrics
            grids[camera].append((f"{c['id']} ep={c['episode']} t={c['frame']} | seed 0", [current, target] + [predictions[k][0][:, sl] for k in KINDS]))
        report["cases"].append(result)
        # Compact per-case sheets show both views and every seed without selection.
        rows = []
        for seed in SEEDS:
            for camera, sl in [("main", slice(0, W)), ("wrist", slice(W, 2 * W))]:
                rows.append((f"{c['id']} | {camera} | seed {seed}", [cur[:, sl]] + [predictions[k][seed][:, sl] for k in KINDS]))
        make_grid(rows, ["current"] + KINDS, OUT / f"{c['id']}_all_seeds.png")
        if c["stage"] == "original":
            endpoint = read_image(HERE / c["images"]["endpoint"])
            real_chain = read_image(HERE / c["images"]["chain_gt"])
            frozen = read_image(OUT / "predictions" / f"{c['id']}_frozen_chain_step{c['chain_steps']}.png")
            oracle = read_image(PREVIOUS / "results" / f"{c['base_id']}_chain_subtask.png")
            chain_result = {"id": c["id"], "steps": c["chain_steps"], "end_frame": c["chain_end"], "segment_end": c["segment_end"], "cameras": {}}
            for camera, sl in [("main", slice(0, W)), ("wrist", slice(W, 2 * W))]:
                chain_result["cameras"][camera] = {name: {"vs_real_chain_end": image_metrics(img[:, sl], real_chain[:, sl]), "vs_segment_end": image_metrics(img[:, sl], endpoint[:, sl])} for name, img in [("copy", cur), ("frozen_state", frozen), ("previous_oracle_state", oracle)]}
                chains[camera].append((f"{c['base_id']} | {c['chain_steps']} generations", [cur[:, sl], frozen[:, sl], oracle[:, sl], real_chain[:, sl], endpoint[:, sl]]))
            first = read_image(OUT / "predictions" / f"{c['id']}_aligned_seed0.png")
            previous = read_image(PREVIOUS / "results" / f"{c['base_id']}_subtask_seed0.png")
            chain_result["first_step_matches_previous_probe"] = bool(np.array_equal(first, previous))
            chain_result["first_step_mae_vs_previous"] = image_metrics(first, previous)["mae_255"]
            report["chains"].append(chain_result)
    first = manifest["cases"][0]
    pred = read_image(OUT / "predictions" / f"{first['id']}_aligned_seed0.png")
    repeat = read_image(OUT / "repeat_aligned_seed0.png")
    report["same_input_same_seed_repeat"] = {"identical_uint8_pixels": bool(np.array_equal(pred, repeat)), **image_metrics(pred, repeat)}
    for camera in grids:
        make_grid(grids[camera], ["current", "real t+16"] + KINDS, OUT / f"prompt_controls_{camera}.png")
        make_grid(chains[camera], ["current", "frozen-state chain", "prior oracle-state chain", "real chain end", "real segment end"], OUT / f"chains_{camera}.png")
    report["runtime"] = json.loads((OUT / "runtime.json").read_text())
    save_json(OUT / "report.json", report)
    print("Saved metrics and all-seed image grids", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run", "analyze"])
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "analyze": analyze}[args.mode]()
