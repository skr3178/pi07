#!/usr/bin/env python3
"""Run the downloaded base BAGEL DF11 checkpoint on fixed LIBERO subgoal cases.

This is a qualitative image-edit probe, not a policy or simulator evaluation.
Run with ../.venv-bagel/bin/python from pi07_code, or use absolute paths.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path,
                        default=Path("/media/skr/storage/models/BAGEL-7B-MoT-DF11"))
    parser.add_argument("--source", type=Path, default=root.parent / "bagel_dfloat11")
    parser.add_argument("--cases", type=Path,
                        default=root / "artifacts/bagel_subgoal_probe/cases.json")
    parser.add_argument("--output", type=Path,
                        default=root / "artifacts/bagel_subgoal_probe/results")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu-offload-blocks", type=int, default=None,
                        help="Default: offload all compressed blocks; 0 disables offload.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cache = root.parent / ".cache-bagel"
    for key, suffix in [("MPLCONFIGDIR", "matplotlib"), ("CUPY_CACHE_DIR", "cupy"),
                        ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("HF_HOME", "hf")]:
        os.environ.setdefault(key, str(cache / suffix))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.source.resolve()))

    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torch.utils._pytree import tree_map
    from accelerate import init_empty_weights
    from dfloat11 import DFloat11Model
    from data.transforms import ImageTransform
    from data.data_utils import add_special_tokens
    from modeling.bagel import (BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
                               SiglipVisionConfig, SiglipVisionModel)
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from inferencer import InterleaveInferencer

    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("BAGEL DF11 requires a CUDA GPU")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start = time.monotonic()
    print("Loading BAGEL; GPU:", torch.cuda.get_device_name(), flush=True)
    print("Free/total GPU bytes:", torch.cuda.mem_get_info(), flush=True)

    llm_config = Qwen2Config.from_json_file(str(args.model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(str(args.model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=str(args.model_path / "vae/ae.safetensors"))
    config = BagelConfig(
        visual_gen=True, visual_und=True, llm_config=llm_config,
        vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm_config), SiglipVisionModel(vit_config), config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    model = model.to(torch.bfloat16)
    model.load_state_dict({
        name: torch.empty(param.shape, dtype=param.dtype, device="cpu")
        if param.device.type == "meta" else param
        for name, param in model.state_dict().items()
    }, assign=True)
    model = DFloat11Model.from_pretrained(
        str(args.model_path), bfloat16_model=model, device="cuda:0",
        cpu_offload=args.cpu_offload_blocks != 0,
        cpu_offload_blocks=args.cpu_offload_blocks,
    ).eval()
    vae_model = vae_model.to("cuda:0").eval()

    class PreparedOnDevice:
        """Place prepared inputs on this single GPU, without Accelerate dispatch hooks."""
        def __getattr__(self, name):
            value = getattr(model, name)
            if not name.startswith("prepare_"):
                return value
            def prepare(*positional, **keyword):
                return tree_map(
                    lambda x: x.to("cuda:0") if isinstance(x, torch.Tensor) else x,
                    value(*positional, **keyword),
                )
            return prepare

    inferencer = InterleaveInferencer(
        PreparedOnDevice(), vae_model, tokenizer,
        ImageTransform(1024, 512, 16), ImageTransform(980, 224, 14), new_token_ids,
    )
    settings = dict(think=False, cfg_text_scale=4.0, cfg_img_scale=2.0,
                    cfg_interval=[0.0, 1.0], timestep_shift=3.0,
                    num_timesteps=args.steps, cfg_renorm_min=1.0,
                    cfg_renorm_type="text_channel", do_sample=False)
    report = {
        "checkpoint": str(args.model_path), "scope": "base BAGEL zero-shot image editing",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "attention": "PyTorch SDPA fallback with bottom-right causal alignment",
        "device_placement": "single GPU prepared-input wrapper; DFloat11 CPU offload",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "cpu_offload_blocks": args.cpu_offload_blocks,
        "seed": args.seed, "settings": settings,
        "load_seconds": round(time.monotonic() - start, 2), "cases": [],
    }
    print("Loaded in", report["load_seconds"], "seconds", flush=True)
    cases = json.loads(args.cases.read_text())
    if args.limit:
        cases = cases[:args.limit]
    for case in cases:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        print("Generating", case["id"], flush=True)
        started = time.monotonic()
        with torch.inference_mode():
            generated = inferencer(image=Image.open(case["current"]).convert("RGB"),
                                   text=case["prompt"], **settings)["image"]
        torch.cuda.synchronize()
        path = args.output / f"{case['id']}_seed{args.seed}.png"
        generated.save(path)
        result = dict(case, generated=str(path.resolve()),
                      elapsed_seconds=round(time.monotonic() - started, 2),
                      peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                      generated_size=list(generated.size))
        report["cases"].append(result)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Saved", path, "in", result["elapsed_seconds"], "seconds", flush=True)
        # Save the comparison after every case, so partial results are reviewable.
        sheet = Image.new("RGB", (3 * 384, 430 * len(report["cases"])), "white")
        draw = ImageDraw.Draw(sheet)
        for row, item in enumerate(report["cases"]):
            draw.text((8, row * 430 + 4), item["id"], fill="black")
            for col, (key, title) in enumerate([
                ("current", "Current observation"), ("generated", "BAGEL generated subgoal"),
                ("reference", "Later demo frame (reference only)")
            ]):
                draw.text((col * 384 + 8, row * 430 + 22), title, fill="black")
                sheet.paste(Image.open(item[key]).convert("RGB").resize((384, 384)),
                            (col * 384, row * 430 + 42))
        sheet.save(args.output / "comparison.png")
    print("Done:", args.output / "comparison.png", flush=True)


if __name__ == "__main__":
    main()
