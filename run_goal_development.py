"""Matched closed-loop development episodes; final evaluation starts excluded."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from train_goal_pilot import ROOT, file_hash, write_json, np, torch
from train_goal_pilot import GOALS, GoalSmolVLAPolicy, load_smolvla_config, make_pre_post_processors

SETUP = ROOT / "artifacts/lerobot_libero_setup"
os.environ["LIBERO_CONFIG_PATH"] = str(SETUP / "libero_config")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("NUMBA_CACHE_DIR", str(SETUP / "numba_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(SETUP / "mpl_cache"))
import imageio.v2 as imageio
from PIL import Image
from lerobot.envs.libero import LiberoEnv, _get_suite
from lerobot.envs.utils import preprocess_observation


class GoalWorker:
    def __init__(self, out, prompts):
        self.out = out
        write_json(out / "prompts.json", sorted(set(prompts)))
        self.log = (out / "imagewam_worker.log").open("w")
        self.process = subprocess.Popen([
            "/media/skr/storage/envs/imagewam-venv/bin/python", "-u", str(ROOT / "imagewam_goal_worker.py"),
            "--prompts", str(out / "prompts.json")], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1)
        self.ready = self.receive()
        if self.ready["status"] != "ready":
            raise RuntimeError(self.ready)

    def receive(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ImageWAM worker exited; inspect imagewam_worker.log")
        response = json.loads(line)
        if response.get("status") == "error":
            raise RuntimeError(response["error"])
        return response

    def predict(self, obs, prompt, seed, name):
        source, target = self.out / f"{name}_current.npz", self.out / f"{name}_goal.png"
        np.savez_compressed(source, main=obs["pixels"]["image"], wrist=obs["pixels"]["image2"], state=obs["agent_pos"])
        self.process.stdin.write(json.dumps({"input": str(source), "output": str(target), "prompt": prompt, "seed": seed})+"\n")
        self.process.stdin.flush()
        response = self.receive()
        rgb = np.asarray(Image.open(target).convert("RGB"))
        assert rgb.shape == (224,448,3)
        return [torch.from_numpy(np.asarray(Image.fromarray(view).resize((256,256), Image.Resampling.BILINEAR)).copy())
                .permute(2,0,1)[None].float()/255 for view in (rgb[:,:224],rgb[:,224:])], response

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.stdin.write('{"command":"close"}\n'); self.process.stdin.flush()
                self.receive(); self.process.wait(timeout=30)
        except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                self.process.terminate(); self.process.wait(timeout=30)
        finally:
            self.log.close()


def summaries(rows):
    output = {}
    for condition in ("A_dropped", "B_dropped", "B_generated"):
        selected = [r for r in rows if r["condition"] == condition]
        good = [r for r in selected if not r.get("error")]
        output[condition] = {"episodes": len(good), "exceptions": len(selected)-len(good),
            "successes": sum(r["success"] for r in good),
            "success_rate": float(np.mean([r["success"] for r in good])) if good else None,
            "timeouts": sum(r["timeout"] for r in good),
            "action_components": sum(r["action_components"] for r in good),
            "clipped_components": sum(r["clipped_components"] for r in good),
            "finite_action_components": sum(r["action_components"] for r in good),
            "wall_seconds": sum(r["wall_seconds"] for r in good)}
    return output


def main(args):
    run, out = args.run.resolve(), args.output.resolve()
    if (out / "manifest.json").exists():
        raise RuntimeError("Preserve existing development run; choose a fresh output")
    if json.loads((run / "report.json").read_text())["status"] != "complete":
        raise RuntimeError("Both training arms must finish first")
    if not set(args.init_ids) <= {0,1}:
        raise ValueError("This development script may only use reserved development starts 0 and 1")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    suite = _get_suite("libero_10")
    jobs = [{"task_id": task, "task": suite.get_task(task).language, "initial_state_id": init,
             "seed": task*100+init} for task in args.task_ids for init in args.init_ids]
    manifest = {"purpose": "development only; not the final 100-episode benchmark", "training_run": str(run),
        "checkpoint_sha256": {a:file_hash(run / a / "model.safetensors") for a in ("A","B")},
        "conditions": ["A_dropped", "B_dropped", "B_generated"], "jobs": jobs,
        "max_steps": args.max_steps, "settling_steps": 10, "action_chunk": 50, "execute_steps": 8,
        "goal_refresh_simulator_actions": 16, "world_model_calls_per_refresh": 1,
        "world_model_denoising_steps": 10, "goal_seed": "500000 + episode seed * 1000 + refresh simulator step",
        "goal_prompt": "full task text", "image_convention": "already rotated 180 degrees by LeRobot; main then wrist",
        "image_sizes": "256px observations; per-camera resize to 224 for ImageWAM; generated goals resize to 256 for policy",
        "state_convention": "current raw EEF xyz, axis-angle rotation, two gripper joints; ImageWAM min/max normalization",
        "timing": "simulation pauses during generation/inference; no real-time claim",
        "sources_sha256": {name:file_hash(ROOT / name) for name in ("run_goal_development.py","imagewam_goal_worker.py","goal_policy.py")},
        "world_model": "/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b/model.pt"}
    model_path = Path(manifest["world_model"])
    manifest["world_model_sha256"] = file_hash(model_path)
    manifest["world_model_config_sha256"] = file_hash(model_path.with_name("config.yaml"))
    manifest["world_model_stats_sha256"] = file_hash(model_path.with_name("dataset_stats.json"))
    manifest["imagewam_source_commits"] = (ROOT / "artifacts/imagewam_setup/source_commits.txt").read_text()
    write_json(out / "manifest.json", manifest)
    report = {"status":"running", "episodes":[], "summaries":{}}
    worker = None
    try:
        for condition in manifest["conditions"]:
            if condition == "B_generated":
                worker = GoalWorker(out, [j["task"] for j in jobs])
                report["world_model_startup"] = worker.ready
                print(f"ImageWAM ready: {json.dumps(worker.ready)}", flush=True)
            cfg, _ = load_smolvla_config(run / condition[0])
            policy = GoalSmolVLAPolicy.from_pretrained(run / condition[0], config=cfg, strict=True).eval()
            pre, post = make_pre_post_processors(cfg, pretrained_path=str(run / condition[0]))
            torch.cuda.reset_peak_memory_stats()
            for job in jobs:
                row = {"condition":condition, **job}; env = writer = None
                begin = time.perf_counter()
                try:
                    seed = job["seed"]
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    env = LiberoEnv(suite, job["task_id"], "libero_10", obs_type="pixels_agent_pos",
                                    episode_index=job["initial_state_id"], num_steps_wait=10)
                    obs, _ = env.reset(seed=seed); policy.reset()
                    name = f"{condition}_task{job['task_id']}_init{job['initial_state_id']}"
                    writer = imageio.get_writer(str(out / f"{name}.mp4"), fps=2.5, codec="libx264", quality=7)
                    actions, states, times, generations = [], [], [], []
                    clipped = 0; ended = success = False
                    goals = [torch.zeros(1,3,256,256) for _ in GOALS]
                    for step in range(args.max_steps):
                        obs["pixels"] = {k:np.ascontiguousarray(v) for k,v in obs["pixels"].items()}
                        if step % 8 == 0:
                            writer.append_data(np.concatenate([obs["pixels"][k] for k in ("image","image2")],axis=1))
                        if worker is not None and step % 16 == 0:
                            goals, timing = worker.predict(obs, job["task"], 500000+seed*1000+step, f"{name}_step{step:04d}")
                            generations.append({"step":step, **timing})
                        inputs = preprocess_observation(obs); inputs["task"] = [job["task"]]
                        for key,value in zip(GOALS,goals):
                            inputs[key] = value
                            inputs[f"{key}_padding_mask"] = torch.tensor([worker is not None])
                        if step % 8 == 0:
                            torch.cuda.synchronize(); prediction_start = time.perf_counter()
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(inputs)))[0].cpu().numpy()
                        if step % 8 == 0:
                            torch.cuda.synchronize(); times.append(time.perf_counter()-prediction_start)
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise RuntimeError("Invalid policy action")
                        bounded = np.clip(action,-1,1).astype(np.float32)
                        clipped += int(np.sum(action != bounded))
                        actions.append(action); states.append(obs["agent_pos"].copy())
                        obs, _, terminated, truncated, info = env.step(bounded)
                        if not np.isfinite(obs["agent_pos"]).all():
                            raise RuntimeError("Nonfinite simulator state")
                        success = bool(info.get("is_success",False)); ended = bool(terminated or truncated)
                        if ended: break
                    writer.append_data(np.concatenate([obs["pixels"][k] for k in ("image","image2")],axis=1))
                    writer.close(); writer = None
                    np.savez_compressed(out / f"{name}_trajectory.npz", actions=actions, states=states)
                    row.update({"success":success,"steps":step+1,"timeout":not ended,"clipped_components":clipped,
                        "action_components":7*len(actions),"chunk_seconds":times,"generations":generations,
                        "wall_seconds":time.perf_counter()-begin,"video":f"{name}.mp4"})
                    print(f"{name}: success={success} steps={step+1} wall={row['wall_seconds']:.1f}s",flush=True)
                except Exception:
                    row["error"] = traceback.format_exc()
                    raise
                finally:
                    if writer: writer.close()
                    if env: env.close()
                    report["episodes"].append(row); report["summaries"] = summaries(report["episodes"])
                    report["policy_peak_allocated_gb"] = torch.cuda.max_memory_allocated()/1e9
                    write_json(out / "report.json",report)
            del policy,pre,post
            gc.collect(); torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"; report["error"] = traceback.format_exc()
        raise
    finally:
        if worker is not None:
            worker.close()
        write_json(out / "report.json",report)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run",type=Path,default=ROOT / "artifacts/goal_training_1000")
    p.add_argument("--output",type=Path,default=ROOT / "artifacts/goal_closed_loop_dev")
    p.add_argument("--task-ids",type=int,nargs="+",default=[0,4,8])
    p.add_argument("--init-ids",type=int,nargs="+",default=[0,1])
    p.add_argument("--max-steps",type=int,default=520)
    main(p.parse_args())
