"""Declared, paired LIBERO development evaluation of saved policy checkpoints.

Unlike historical runners, loads each checkpoint's actual attention layout.
Inputs contain current observations and optional generated goals, never demo
future frames or demonstration subtask times. Final benchmark starts stay unused.
"""
import argparse
import gc
import json
import random
import shutil
import time
import traceback
from pathlib import Path

from run_goal_development import (ROOT, GoalWorker, LiberoEnv, _get_suite,
                                 preprocess_observation, imageio, np, torch,
                                 file_hash, write_json, load_smolvla_config,
                                 make_pre_post_processors, GOALS)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from goal_policy import GoalSmolVLAPolicy
from attention_check import AltGoalPolicy
from section_v_prompt import PromptOptions, runtime_prompt, make_section_v_processors


def load_policy(checkpoint, layout, device="cuda"):
    cfg, _ = load_smolvla_config(checkpoint, device=device)
    classes = {"plain": SmolVLAPolicy, "codex": GoalSmolVLAPolicy, "alt": AltGoalPolicy}
    metadata = checkpoint / "run_args.json"
    if metadata.exists() and json.loads(metadata.read_text()).get("layout", layout) != layout:
        raise ValueError("Evaluation layout disagrees with saved training arguments")
    interface = checkpoint / "section_v_interface.json"
    options = None
    if interface.exists():
        config = json.loads(interface.read_text())
        expected = {"alt": "attention_check.AltGoalPolicy", "codex": "goal_policy.GoalSmolVLAPolicy"}
        if config["policy_class"] != expected.get(layout):
            raise ValueError("Section V checkpoint layout mismatch")
        options = PromptOptions(**config["prompt_options"])
        pre, post = make_section_v_processors(cfg, checkpoint, max_length=config["token_limit"])
    else:
        pre, post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
    policy = classes[layout].from_pretrained(checkpoint, config=cfg, strict=True).eval()
    return policy, pre, post, options


def predicates(env):
    domain = env._env.env
    goals = domain.parsed_problem["goal_state"]
    return goals, [bool(domain._eval_predicate(goal)) for goal in goals]


def summarize(rows):
    result = {}
    for name in dict.fromkeys(r["condition"] for r in rows):
        selected = [r for r in rows if r["condition"] == name]
        good = [r for r in selected if "error" not in r]
        result[name] = {"episodes": len(good), "errors": len(selected)-len(good),
            "successes": sum(r["success"] for r in good),
            "success_rate": float(np.mean([r["success"] for r in good])) if good else None,
            "mean_max_predicate_fraction": float(np.mean([r["max_predicate_fraction"] for r in good])) if good else None,
            "wall_seconds": sum(r["wall_seconds"] for r in good)}
    return result


def main(args):
    out = args.output.resolve()
    if (out / "manifest.json").exists():
        raise ValueError("Choose a fresh directory; saved runs are never overwritten")
    spec = json.loads(args.spec.read_text())
    if spec["suite"] not in ("libero_10", "libero_goal"):
        raise ValueError("Unsupported suite")
    if not set(spec["init_ids"]) <= {0, 1} or not spec["init_ids"]:
        raise ValueError("This runner uses development initial states 0/1 only")
    conditions = spec["conditions"]
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Condition names must be unique")
    for c in conditions:
        if c["goals"] not in ("off", "copy", "generated"):
            raise ValueError("Only current-observation goal sources permitted")
        if c["layout"] == "plain" and c["goals"] != "off":
            raise ValueError("Plain policy has no trained goal interface")
        if not c["name"].replace("_", "").isalnum():
            raise ValueError("Use simple condition names")
        for name in ("model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
            if not (Path(c["checkpoint"]) / name).is_file():
                raise FileNotFoundError(Path(c["checkpoint"]) / name)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU access is required for this evaluator")
    refresh = spec.get("goal_refresh_steps", 16)
    if refresh < 8 or refresh % 8:
        raise ValueError("Refresh must align with the eight-action execution chunk")
    if spec["max_steps"] < 1:
        raise ValueError("Positive episode length required")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    suite = _get_suite(spec["suite"])
    jobs = [{"task_id": t, "task": suite.get_task(t).language,
             "initial_state_id": i, "seed": 100*t+i}
            for t in spec["task_ids"] for i in spec["init_ids"]]
    sources = ("evaluate_libero.py", "run_goal_development.py", "imagewam_goal_worker.py",
               "goal_policy.py", "attention_check.py", "libero_runtime.py", "section_v_prompt.py")
    manifest = {"purpose": "development evaluation; final starts excluded", "spec": spec, "jobs": jobs,
        "checkpoint_sha256": {c["name"]: file_hash(Path(c["checkpoint"])/"model.safetensors") for c in conditions},
        "source_sha256": {f: file_hash(ROOT/f) for f in sources}, "execute_steps": 8,
        "settling_steps": 10, "control_hz": 20, "world_model_calls_per_refresh": 1,
        "goal_prompt": "full task text", "live_subtask_controller": None,
        "world_model": "/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b/model.pt",
        "timing": "simulator pauses for generation and policy inference",
        "progress_definition": "fraction of BDDL terminal predicates currently true; not a quality label"}
    if any(c["goals"] == "generated" for c in conditions):
        model = Path(manifest["world_model"])
        manifest["world_model_sha256"] = file_hash(model)
        manifest["world_model_config_sha256"] = file_hash(model.with_name("config.yaml"))
        manifest["world_model_stats_sha256"] = file_hash(model.with_name("dataset_stats.json"))
    write_json(out/"manifest.json", manifest)
    (out/"source").mkdir()
    for f in sources:
        shutil.copy2(ROOT/f, out/"source"/f)
    report = {"status": "running", "episodes": [], "summaries": {}}
    worker = None
    reset_states = {}
    try:
        for c in conditions:
            if c["goals"] == "generated" and worker is None:
                worker = GoalWorker(out, [j["task"] for j in jobs])
                report["world_model_startup"] = worker.ready
            policy, pre, post, options = load_policy(Path(c["checkpoint"]), c["layout"])
            torch.cuda.reset_peak_memory_stats()
            for job in jobs:
                env = writer = None
                row = {"condition": c["name"], **job}
                begin = time.perf_counter()
                try:
                    seed = job["seed"]
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    env = LiberoEnv(suite, job["task_id"], spec["suite"], obs_type="pixels_agent_pos",
                                    episode_index=job["initial_state_id"], num_steps_wait=10)
                    obs, _ = env.reset(seed=seed); policy.reset()
                    full_state = np.asarray(env._env.env.sim.get_state().flatten()).copy()
                    key = (job["task_id"], job["initial_state_id"])
                    prior = reset_states.setdefault(key, full_state)
                    row["reset_state_max_difference"] = float(np.abs(prior-full_state).max())
                    if row["reset_state_max_difference"] > 1e-8:
                        raise RuntimeError("Paired conditions did not start at the same full simulator state")
                    name = f"{c['name']}_task{job['task_id']}_init{job['initial_state_id']}"
                    writer = imageio.get_writer(str(out/f"{name}.mp4"), fps=2.5, codec="libx264", quality=7)
                    actions, states, progress, generations, inference = [], [], [], [], []
                    goal_names, initial_progress = predicates(env)
                    row["goal_predicates"] = goal_names
                    progress.append(initial_progress)
                    clipped = 0; ended = success = False
                    goals = [torch.zeros(1, 3, 256, 256) for _ in GOALS]
                    for step in range(spec["max_steps"]):
                        obs["pixels"] = {k: np.ascontiguousarray(v) for k, v in obs["pixels"].items()}
                        if step % 8 == 0:
                            writer.append_data(np.concatenate([obs["pixels"][k] for k in ("image", "image2")], axis=1))
                        if c["goals"] == "generated" and step % refresh == 0:
                            goals, timing = worker.predict(obs, job["task"], 500000+seed*1000+step, f"{name}_step{step:04d}")
                            generations.append({"step": step, **timing})
                        inputs = preprocess_observation(obs)
                        if c["goals"] == "copy":
                            goals = [inputs[f"observation.images.{k}"].clone() for k in ("image", "image2")]
                        text = job["task"] if options is None else runtime_prompt(job["task"], options=options,
                            goal_available=c["goals"] != "off").text
                        inputs["task"] = [text]
                        if c["layout"] != "plain":
                            for k, value in zip(GOALS, goals):
                                inputs[k] = value
                                inputs[f"{k}_padding_mask"] = torch.tensor([c["goals"] != "off"])
                        if step % 8 == 0:
                            torch.cuda.synchronize(); t = time.perf_counter()
                        with torch.inference_mode():
                            action = post(policy.select_action(pre(inputs)))[0].cpu().numpy()
                        if step % 8 == 0:
                            torch.cuda.synchronize(); inference.append(time.perf_counter()-t)
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise RuntimeError("Nonfinite or incorrectly shaped action")
                        bounded = np.clip(action, -1, 1).astype(np.float32)
                        clipped += int(np.sum(action != bounded))
                        actions.append(action); states.append(obs["agent_pos"].copy())
                        obs, _, terminated, truncated, info = env.step(bounded)
                        success = bool(info.get("is_success", False)); ended = bool(terminated or truncated)
                        # The LeRobot wrapper auto-resets after terminal success. Record terminal
                        # predicates from the verified success flag, not from the reset simulator.
                        progress.append([True]*len(goal_names) if success else predicates(env)[1])
                        if not np.isfinite(obs["agent_pos"]).all():
                            raise RuntimeError("Nonfinite simulator state")
                        if ended:
                            break
                    writer.append_data(np.concatenate([obs["pixels"][k] for k in ("image", "image2")], axis=1))
                    writer.close(); writer = None
                    np.savez_compressed(out/f"{name}_trajectory.npz", actions=actions, states=states,
                                        predicates=progress, initial_sim_state=full_state)
                    row.update(success=success, steps=step+1, timeout=not ended, clipped_components=clipped,
                        action_components=7*len(actions), chunk_seconds=inference, generations=generations,
                        max_predicate_fraction=float(np.asarray(progress).mean(axis=1).max()),
                        final_predicate_fraction=float(np.mean(progress[-1])), video=f"{name}.mp4",
                        wall_seconds=time.perf_counter()-begin)
                    print(f"{name}: success={success} steps={step+1} progress={row['max_predicate_fraction']:.2f}", flush=True)
                except Exception:
                    row["error"] = traceback.format_exc()
                    raise
                finally:
                    if writer is not None: writer.close()
                    if env is not None: env.close()
                    report["episodes"].append(row)
                    report["summaries"] = summarize(report["episodes"])
                    report.setdefault("policy_peak_allocated_gb", {})[c["name"]] = torch.cuda.max_memory_allocated()/1e9
                    write_json(out/"report.json", report)
            del policy, pre, post
            gc.collect(); torch.cuda.empty_cache()
        report["status"] = "complete"
    except Exception:
        report["status"] = "failed"; report["error"] = traceback.format_exc()
        raise
    finally:
        if worker is not None: worker.close()
        write_json(out/"report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
