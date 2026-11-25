"""Check real dataset decoding, CUDA, policy loading, and LIBERO stepping.

This is an integration preflight, not a policy success-rate evaluation.
Run with the dedicated environment; see README.md.
"""

import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT))
ENV_CONFIG = HERE / "libero_config"
ENV_CONFIG.mkdir(exist_ok=True)
LIBERO_ROOT = PROJECT / "vendor/lerobot-libero/libero/libero"
# JSON is valid YAML; prepare this before import to avoid LIBERO's interactive setup.
(ENV_CONFIG / "config.yaml").write_text(json.dumps({
    "benchmark_root": str(LIBERO_ROOT),
    "bddl_files": str(LIBERO_ROOT / "bddl_files"),
    "init_states": str(LIBERO_ROOT / "init_files"),
    "assets": str(LIBERO_ROOT / "assets"),
    "datasets": str(PROJECT / "data"),
}, indent=2) + "\n")
os.environ["LIBERO_CONFIG_PATH"] = str(ENV_CONFIG)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / "datasets_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mpl_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import importlib.metadata as metadata
import numpy as np
from PIL import Image
import torch

REPORT = {"status": "running", "checks": {}, "simulator_runs": []}


def save():
    (HERE / "smoke_report.json").write_text(json.dumps(REPORT, indent=2) + "\n")


def tensor_info(x):
    assert torch.isfinite(x).all()
    return {"shape": list(x.shape), "dtype": str(x.dtype),
            "min": float(x.min()), "max": float(x.max())}


def main():
    REPORT["versions"] = {p: metadata.version(p) for p in (
        "torch", "torchvision", "torchcodec", "transformers", "lerobot", "libero", "mujoco", "numpy"
    )}
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x).all()
    torch.cuda.synchronize()
    REPORT["checks"]["cuda"] = {"gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
                                 "capability": list(torch.cuda.get_device_capability())}
    print("CUDA calculation passed", flush=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        "lerobot/libero_10_subtask", root=PROJECT / "data/libero_10_subtask",
        episodes=[0], video_backend="torchcodec",
        delta_timestamps={"action": [i / 10 for i in range(50)]},
    )
    sample = dataset[0]
    keys = ("observation.images.image", "observation.images.image2", "observation.state", "action")
    REPORT["checks"]["dataset"] = {k: tensor_info(sample[k]) for k in keys}
    assert sample["observation.state"].shape == (8,)
    assert sample["action"].shape == (50, 7)
    assert sample["observation.images.image"].shape == (3, 256, 256)
    for k in keys[:2]:
        assert 0 <= sample[k].min() and sample[k].max() <= 1
    REPORT["checks"]["dataset"]["task"] = sample["task"]
    print("Real AV1 dataset sample and 50-action window decoded", flush=True)
    save()

    # Only download config/tokenizer assets; the complete policy weights already exist locally.
    from huggingface_hub import snapshot_download
    assets = snapshot_download(
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja"],
    )
    REPORT["vlm_config_tokenizer_snapshot"] = assets

    from libero_runtime import load_smolvla_config
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    checkpoint = "/media/skr/storage/models/smolvla_libero"
    config, compatibility = load_smolvla_config(checkpoint)
    REPORT["config_compatibility"] = compatibility
    t0 = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint)
    REPORT["checks"]["policy_load"] = {"checkpoint": checkpoint, "strict": True,
        "seconds": time.perf_counter() - t0, "chunk_size": config.chunk_size,
        "n_action_steps": config.n_action_steps, "parameter_count": sum(p.numel() for p in policy.parameters())}
    with torch.inference_mode():
        policy.reset()
        dataset_input = {k: sample[k] for k in keys[:3]}
        dataset_input["task"] = sample["task"]
        action = postprocessor(policy.select_action(preprocessor(dataset_input)))
        REPORT["checks"]["dataset_policy_action"] = tensor_info(action)
        assert action.shape == (1, 7)
    print("Strict checkpoint load and dataset-to-action inference passed", flush=True)
    save()

    from lerobot.envs.libero import LiberoEnv, _get_suite, get_libero_dummy_action
    from lerobot.envs.utils import preprocess_observation
    for suite_name in ("libero_10", "libero_goal"):
        env = LiberoEnv(_get_suite(suite_name), 0, suite_name, obs_type="pixels_agent_pos",
                        observation_width=256, observation_height=256, num_steps_wait=0)
        try:
            # Check that reset actually applies two different requested initial states.
            errors = []
            for initial_id in (0, 1):
                env._init_state_id = initial_id
                obs, _ = env.reset(seed=0)
                actual = env._env.env.sim.get_state().flatten()
                target = np.asarray(env._init_states[initial_id])
                errors.append(float(np.max(np.abs(actual - target))))
                np.testing.assert_allclose(actual, target, atol=1e-7, rtol=0)
            env._init_state_id = 0
            obs, _ = env.reset(seed=0)
            for _ in range(10):
                obs, _, _, _, _ = env.step(np.asarray(get_libero_dummy_action(), dtype=np.float32))
            assert obs["agent_pos"].shape == (8,) and np.isfinite(obs["agent_pos"]).all()
            for camera, pixels in obs["pixels"].items():
                assert pixels.shape == (256, 256, 3) and pixels.dtype == np.uint8
                assert np.std(pixels) > 1, "Rendered image is nearly constant"
                Image.fromarray(pixels).save(HERE / f"{suite_name}_{camera}_start.png")
            policy.reset()
            # The unbatched wrapper rotates images using negative-stride views.
            # Vector environments normally stack/copy these before preprocessing.
            obs["pixels"] = {k: np.ascontiguousarray(v) for k, v in obs["pixels"].items()}
            inputs = preprocess_observation(obs)
            inputs["task"] = [env.task_description]
            t0 = time.perf_counter()
            with torch.inference_mode():
                action = postprocessor(policy.select_action(preprocessor(inputs)))
            torch.cuda.synchronize()
            action_seconds = time.perf_counter() - t0
            assert action.shape == (1, 7) and torch.isfinite(action).all()
            raw = action[0].cpu().numpy()
            applied = np.clip(raw, -1, 1).astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(applied)
            assert np.isfinite(obs["agent_pos"]).all()
            REPORT["simulator_runs"].append({
                "suite": suite_name, "task_id": 0, "task": env.task_description,
                "initial_state_max_abs_errors": errors,
                "control_freq": env._env.env.control_freq,
                "policy_input_shapes": {k: list(v.shape) for k,v in inputs.items() if torch.is_tensor(v)},
                "action": raw.tolist(), "clipped_action_components": int(np.sum(raw != applied)),
                "inference_seconds": action_seconds, "reward_after_one_policy_action": float(reward),
                "terminated": bool(terminated), "truncated": bool(truncated),
                "is_success": bool(info.get("is_success", False)),
            })
            print(f"{suite_name}: two initial-state resets, two cameras and policy action step passed", flush=True)
            save()
        finally:
            env.close()
    REPORT["peak_gpu_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    REPORT["status"] = "passed"
    REPORT["limitation"] = "Functional preflight only: no full episode evaluation, training or ImageWAM-policy integration."


if __name__ == "__main__":
    try:
        main()
    except Exception:
        REPORT["status"] = "failed"
        REPORT["error"] = traceback.format_exc()
        raise
    finally:
        save()
