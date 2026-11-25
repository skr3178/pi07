"""Small, fixed development pilot of the unmodified pretrained policy.

No training or world-model guidance. The manifest is saved before any rollouts.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = PROJECT / "artifacts/smolvla_reference_pilot"
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["LIBERO_CONFIG_PATH"] = str(HERE / "libero_config")
os.environ.setdefault("HF_HOME", "/media/skr/storage/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_CACHE", str(HERE / "datasets_cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "mpl_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import imageio.v2 as imageio
import numpy as np
import torch

from libero_runtime import load_smolvla_config
from lerobot.envs.libero import LiberoEnv, _get_suite, TASK_SUITE_MAX_STEPS
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    radius = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return [max(0., center-radius), min(1., center+radius)]


def summary(rows):
    n = len(rows)
    if not n:
        return {"episodes": 0}
    good = [r for r in rows if not r.get("error")]
    wins = sum(r.get("success", False) for r in good)
    successful_lengths = [r["steps"] for r in good if r["success"]]
    latencies = [v for r in good for v in r["chunk_inference_seconds"]]
    clipped = sum(r["clipped_components"] for r in good)
    components = sum(r["action_components"] for r in good)
    return {"attempted_episodes": n, "valid_episodes": len(good), "exceptions": n-len(good),
            "successes": wins, "success_rate": wins/len(good) if good else None,
            "wilson_95": wilson(wins, len(good)),
            "timeouts": sum(r["timeout"] for r in good),
            "mean_steps_to_success": float(np.mean(successful_lengths)) if successful_lengths else None,
            "action_clipping_fraction": clipped/components if components else None,
            "chunk_latency_p50_seconds": float(np.percentile(latencies, 50)) if latencies else None,
            "chunk_latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else None,
            "rollout_wall_seconds": sum(r["wall_seconds"] for r in good)}


def save_report(rows, status):
    report = {"status": status, "purpose": "development pilot, not final benchmark",
              "overall": summary(rows), "suites": {}, "tasks": {}, "episodes": rows,
              "peak_gpu_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
              "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved()/1e9}
    for suite in ("libero_10", "libero_goal"):
        report["suites"][suite] = summary([r for r in rows if r["suite"] == suite])
        for task in (0, 4, 8):
            report["tasks"][f"{suite}/{task}"] = summary([r for r in rows if r["suite"] == suite and r["task_id"] == task])
    tmp = OUT / "report.tmp.json"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(OUT / "report.json")


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / "manifest.json").exists():
        raise RuntimeError("Pilot output already exists; preserve it and choose a new run directory")
    checkpoint = Path("/media/skr/storage/models/smolvla_libero")
    config, adapter = load_smolvla_config(checkpoint)
    suites = {name: _get_suite(name) for name in ("libero_10", "libero_goal")}
    jobs = [{"suite": name, "task_id": task, "task": suite.get_task(task).language,
             "initial_state_id": init, "seed": si*10000+task*100+init,
             "max_steps": TASK_SUITE_MAX_STEPS[name]}
            for si, (name, suite) in enumerate(suites.items()) for task in (0, 4, 8) for init in (0, 1)]
    with (checkpoint / "model.safetensors").open("rb") as f:
        model_hash = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {"checkpoint": str(checkpoint), "model_sha256": model_hash, "adapter": adapter,
                "source_commits": json.loads((HERE / "source_commits.json").read_text()),
                "development_initial_states": [0, 1], "exclude_from_final_evaluation": True,
                "chunk_size": config.chunk_size, "execute_steps": config.n_action_steps,
                "denoising_steps": config.num_steps, "settling_steps": 10,
                "action_clipping": [-1, 1], "video_every_simulator_steps": 8, "jobs": jobs}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.set_num_threads(4)
    policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint))
    torch.cuda.reset_peak_memory_stats()
    rows = []
    save_report(rows, "running")
    for job in jobs:
        env = None
        writer = None
        row = dict(job)
        t0 = time.perf_counter()
        try:
            seed = job["seed"]
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            env = LiberoEnv(suites[job["suite"]], job["task_id"], job["suite"],
                            obs_type="pixels_agent_pos", episode_index=job["initial_state_id"], num_steps_wait=10)
            obs, _ = env.reset(seed=seed)
            policy.reset()
            name = f"{job['suite']}_task{job['task_id']}_init{job['initial_state_id']}"
            video = OUT / f"{name}.mp4"
            writer = imageio.get_writer(str(video), fps=env._env.env.control_freq / 8,
                                        codec="libx264", quality=7, macro_block_size=16)
            writer.append_data(np.concatenate([obs["pixels"][c] for c in ("image", "image2")], axis=1))
            times, actions, states, clipped = [], [], [], 0
            success, ended = False, False
            for step in range(job["max_steps"]):
                obs["pixels"] = {k: np.ascontiguousarray(v) for k,v in obs["pixels"].items()}
                inputs = preprocess_observation(obs)
                inputs["task"] = [env.task_description]
                if step % config.n_action_steps == 0:
                    torch.cuda.synchronize()
                    pred_start = time.perf_counter()
                with torch.inference_mode():
                    action = post(policy.select_action(pre(inputs)))[0].cpu().numpy()
                if step % config.n_action_steps == 0:
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - pred_start)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise RuntimeError("Nonfinite action or unexpected action shape")
                bounded = np.clip(action, -1, 1).astype(np.float32)
                clipped += int(np.sum(action != bounded))
                actions.append(action); states.append(obs["agent_pos"].copy())
                obs, _, terminated, truncated, info = env.step(bounded)
                if not np.isfinite(obs["agent_pos"]).all():
                    raise RuntimeError("Nonfinite simulator state")
                success = bool(info.get("is_success", False))
                ended = bool(terminated or truncated)
                if (step+1) % 8 == 0 or ended:
                    writer.append_data(np.concatenate([obs["pixels"][c] for c in ("image", "image2")], axis=1))
                if ended:
                    break
            writer.close(); writer = None
            np.savez_compressed(OUT / f"{name}_trajectory.npz", actions=np.asarray(actions),
                                states=np.asarray(states), chunk_inference_seconds=np.asarray(times))
            row.update({"success": success, "steps": step+1, "timeout": not ended,
                        "clipped_components": clipped, "action_components": 7*len(actions),
                        "finite_action_components": 7*len(actions), "chunk_inference_seconds": times,
                        "wall_seconds": time.perf_counter()-t0, "video": str(video.relative_to(OUT))})
            print(f"{name}: success={success} steps={step+1} wall={row['wall_seconds']:.1f}s", flush=True)
        except Exception:
            row["error"] = traceback.format_exc()
            row["wall_seconds"] = time.perf_counter()-t0
            print(row["error"], flush=True)
        finally:
            if writer is not None:
                writer.close()
            if env is not None:
                env.close()
        rows.append(row)
        save_report(rows, "running")
        if row.get("error"):
            save_report(rows, "runtime_error")
            raise RuntimeError("Stop pilot to diagnose a runtime error")
    save_report(rows, "complete")
    print(json.dumps(summary(rows)), flush=True)


if __name__ == "__main__":
    main()
