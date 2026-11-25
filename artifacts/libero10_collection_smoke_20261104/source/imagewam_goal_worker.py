"""Persistent, local JSON-lines ImageWAM worker for the separate ImageWAM env.

Inputs are current, already oriented main/wrist RGB images and measured state.
Only a single prediction call is used; no demonstration future is accessible.
"""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path("/media/skr/storage/models/imagewam")
CHECKPOINT = MODELS / "imagewam_release/libero/flux2_klein_4b"
FLUX_SOURCE = ROOT / "ImageWAM/third_party/flux2"
sys.path[:0] = [str(ROOT / "ImageWAM/src"), str(FLUX_SOURCE / "src"), str(FLUX_SOURCE)]
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def respond(value):
    print(json.dumps(value), flush=True)


def main(args):
    start = time.perf_counter()
    torch.set_num_threads(4)
    with contextlib.redirect_stdout(sys.stderr):
        from imagewam.runtime import create_imagewam_flux2_klein
        scheduler = {"train_shift": 5., "infer_shift": 5., "num_train_timesteps": 1000}
        model = create_imagewam_flux2_klein(
            flux2_model_path=str(MODELS / "flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"),
            ae_model_path=str(MODELS / "flux2/FLUX.2-dev/ae.safetensors"),
            flux2_src_path=str(FLUX_SOURCE), variant="klein-base-4b",
            qwen3_model_spec="Qwen/Qwen3-4B", qwen_context_len=512,
            action_dit_config={"action_dim": 7, "hidden_dim": 1024, "num_heads": 24, "attn_head_dim": 128,
                "num_layers_double": 5, "num_layers_single": 20, "mlp_ratio": 4.,
                "max_action_horizon": 64, "use_gradient_checkpointing": True},
            action_dit_pretrained_path=None, proprio_dim=8, load_text_encoder=False,
            video_scheduler=scheduler, action_scheduler=scheduler,
            loss={"lambda_video": .5, "lambda_action": 1.}, pack_proprio_after_text=True,
            flux2_lora_config={"enabled": False}, model_dtype=torch.bfloat16, device="cuda")
        model.load_checkpoint(str(CHECKPOINT / "model.pt"))
        model = model.to("cuda").eval()
        contexts = {}
        with torch.inference_mode():
            for prompt in json.loads(args.prompts.read_text()):
                h, mask = model._encode_flux2_prompts([prompt])
                contexts[prompt] = (h.cpu(), mask.cpu())
        model.text_encoder = None
        del h, mask
        gc.collect(); torch.cuda.empty_cache()
    stats = json.loads((CHECKPOINT / "dataset_stats.json").read_text())["state"]["default"]
    low, high = np.asarray(stats["global_min"]), np.asarray(stats["global_max"])
    assert low.shape == high.shape == (8,) and np.all(high > low)
    startup_peak = torch.cuda.max_memory_allocated()/1e9
    torch.cuda.reset_peak_memory_stats()
    respond({"status": "ready", "startup_seconds": time.perf_counter()-start,
             "startup_peak_allocated_gb": startup_peak, "text_encoder_unloaded": True,
             "allocated_gb": torch.cuda.memory_allocated()/1e9, "prompts": len(contexts)})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            respond({"status": "closed"}); break
        try:
            begin = time.perf_counter()
            with np.load(request["input"]) as current:
                cameras = [np.asarray(Image.fromarray(current[k]).resize((224,224), Image.Resampling.BILINEAR))
                           for k in ("main", "wrist")]
                state = current["state"].copy()
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Expected finite current 8-D robot state")
            rgb = np.concatenate(cameras, axis=1)
            x = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].float()*(2/255)-1
            normalized = np.clip(2*(state-low)/(high-low)-1, -5,5).astype(np.float32)
            h, mask = contexts[request["prompt"]]
            with torch.inference_mode(), contextlib.redirect_stdout(sys.stderr):
                output = model.infer_video_flux2(prompt=None, input_image=x,
                    proprio=torch.from_numpy(normalized)[None], context=h, context_mask=mask,
                    num_inference_steps=10, seed=int(request["seed"]))["image"]
            if output.shape != (3,224,448) or not torch.isfinite(output).all():
                raise ValueError("Invalid generated image")
            pixels = ((output.float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,0).numpy()
            Image.fromarray(pixels).save(request["output"])
            respond({"status": "ok", "seconds": time.perf_counter()-begin,
                     "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
                     "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9})
        except Exception:
            respond({"status": "error", "error": traceback.format_exc()})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    main(parser.parse_args())
