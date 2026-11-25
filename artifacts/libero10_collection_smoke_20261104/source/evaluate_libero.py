"""Declared, paired LIBERO development evaluation of saved policy checkpoints.

Unlike historical runners, loads each checkpoint's actual attention layout.
Inputs contain current observations and optional generated goals, never demo
future frames or demonstration subtask times. Final benchmark starts stay unused.
"""
import argparse
import gc
import json
import math
import random
import shutil
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image

from run_goal_development import (ROOT, GoalWorker, LiberoEnv, _get_suite,
                                 preprocess_observation, imageio, np, torch,
                                 file_hash, write_json, load_smolvla_config,
                                 make_pre_post_processors, GOALS)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from goal_policy import GoalSmolVLAPolicy
from attention_check import AltGoalPolicy
from section_v_prompt import PromptOptions, runtime_prompt, make_section_v_processors
from subtask_controller import build_plan, select_subtask
from metadata_guidance import MetadataGuidedSelector
from libero.libero.envs.bddl_utils import robosuite_parse_problem


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
    if options is not None:
        coverage=config.get("known_training_annotations")
        if coverage is None and (checkpoint.parent/"manifest.json").exists():
            coverage=json.loads((checkpoint.parent/"manifest.json").read_text()).get("known_training_annotations",{})
        policy.pi07_prompt_coverage=coverage or {}
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
    previous_manifest = (json.loads((out/"manifest.json").read_text())
                         if (out/"manifest.json").exists() else None)
    if previous_manifest is not None and not args.resume:
        raise ValueError("Choose a fresh directory; saved runs are never overwritten")
    if args.resume and previous_manifest is None:
        raise ValueError("Resume requires an existing manifest and report")
    spec = json.loads(args.spec.read_text())
    if spec["suite"] not in ("libero_10", "libero_goal"):
        raise ValueError("Unsupported suite")
    phase=spec.get("phase","development")
    if phase not in ("development","training_collection"):
        raise ValueError("Declare development or training_collection; final evaluation is not enabled here")
    allowed={0,1} if phase=="development" else set(range(12,50))
    if not set(spec["init_ids"])<=allowed or not spec["init_ids"]:
        raise ValueError("Development uses starts 0/1; collection uses 12–49. Final starts 2–11 stay reserved.")
    record_training=bool(spec.get("record_training_data",False))
    if record_training and phase!="training_collection":
        raise ValueError("Training data recording requires explicit training_collection phase")
    conditions = spec["conditions"]
    if len({c["name"] for c in conditions}) != len(conditions):
        raise ValueError("Condition names must be unique")
    for c in conditions:
        noise_std=c.get("action_noise_std",0.)
        if not isinstance(noise_std,(int,float)) or not math.isfinite(noise_std) or noise_std<0:
            raise ValueError("Invalid collection action noise")
        if noise_std and phase!="training_collection":raise ValueError("Action perturbations are collection-only")
        beta=c.get("metadata_cfg_beta")
        if beta is not None and (not isinstance(beta,(float,int)) or not math.isfinite(beta) or beta<0):
            raise ValueError("Metadata CFG beta must be finite and nonnegative")
        if beta is not None and not c.get("metadata"):
            raise ValueError("Declare metadata requests for metadata CFG")
        if c.get("subtasks","off") not in ("off","oracle_predicates"):
            raise ValueError("Declare an implemented subtask source")
        if c.get("subtasks","off")!="off" and spec["suite"]!="libero_10":
            raise ValueError("The current oracle is validated for LIBERO-10 only")
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
               "goal_policy.py", "attention_check.py", "libero_runtime.py", "section_v_prompt.py","subtask_controller.py",
               "metadata_guidance.py")
    plans={}
    if any(c.get("subtasks")=="oracle_predicates" for c in conditions):
        for task in spec["task_ids"]:
            plans[task]=build_plan(robosuite_parse_problem(suite.get_task_bddl_file_path(task))["goal_state"])
    manifest = {"purpose": spec.get("purpose","development evaluation; final starts excluded"), "spec": spec, "jobs": jobs,
        "checkpoint_sha256": {c["name"]: file_hash(Path(c["checkpoint"])/"model.safetensors") for c in conditions},
        "source_sha256": {f: file_hash(ROOT/f) for f in sources}, "execute_steps": 8,
        "settling_steps": 10, "control_hz": 20, "world_model_calls_per_refresh": 1,
        "goal_prompt": "current oracle subtask when enabled, otherwise full task",
        "live_subtask_controller": "privileged current BDDL predicates" if plans else None,
        "subtask_update_frequency": "every eight simulator actions; refresh goals on change",
        "world_model_metadata_conditioning": False,
        "metadata_cfg_convention":"v_cond + beta * (v_cond - v_no_metadata); beta=0 is ordinary conditioned inference",
        "metadata_cfg_implementation":"separate positive/negative prefix caches; same task, subtask, images, state and noise",
        "phase":phase,"training_data_recorded":record_training,
        "training_record_format":"per-step uint8 main/wrist before action; state; executed clipped action; current predicate truth",
        "world_model": "/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b/model.pt",
        "timing": "simulator pauses for generation and policy inference",
        "progress_definition": "fraction of BDDL terminal predicates currently true; not a quality label"}
    if any(c["goals"] == "generated" for c in conditions):
        model = Path(manifest["world_model"])
        manifest["world_model_sha256"] = file_hash(model)
        manifest["world_model_config_sha256"] = file_hash(model.with_name("config.yaml"))
        manifest["world_model_stats_sha256"] = file_hash(model.with_name("dataset_stats.json"))
    if previous_manifest is not None:
        for field in ("spec","jobs","checkpoint_sha256"):
            if manifest[field]!=previous_manifest[field]:
                raise ValueError(f"Resume cannot change {field}")
        for field in ("world_model_sha256","world_model_config_sha256","world_model_stats_sha256"):
            if manifest.get(field)!=previous_manifest.get(field):
                raise ValueError(f"Resume world-model file changed: {field}")
        for name,old in previous_manifest["source_sha256"].items():
            # The evaluator may gain orchestration/optional interface features;
            # every such change is snapshotted. Model/input implementations must
            # stay fixed for a paired run.
            if name!="evaluate_libero.py" and manifest["source_sha256"].get(name)!=old:
                raise ValueError(f"Resume dependency changed: {name}")
        report=json.loads((out/"report.json").read_text())
        if report["status"]=="complete":
            print("Already complete; nothing rerun",flush=True)
            return
        attempts=out/"resume_events.json"
        events=json.loads(attempts.read_text()) if attempts.exists() else []
        source_dir=out/f"source_resume_{len(events)+1}"
        events.append({"utc":datetime.now(timezone.utc).isoformat(),"source_directory":source_dir.name,
            "source_sha256":manifest["source_sha256"],"previous_status":report["status"],
            "note":"Only missing/failed episodes are rerun; original manifest and source snapshot retained."})
        write_json(attempts,events)
        failed=[r for r in report["episodes"] if "error" in r]
        report.setdefault("previous_failed_attempts",[]).extend(failed)
        report["episodes"]=[r for r in report["episodes"] if "error" not in r]
        report.pop("error",None)
        report["status"]="running"
    else:
        write_json(out/"manifest.json", manifest)
        source_dir=out/"source"
        report = {"status": "running", "episodes": [], "summaries": {}}
    source_dir.mkdir()
    for f in sources:
        shutil.copy2(ROOT/f, source_dir/f)
    write_json(out/"report.json",report)
    worker = None
    reset_states = {}
    completed=set()
    allowed={(c["name"],j["task_id"],j["initial_state_id"]) for c in conditions for j in jobs}
    for row in report["episodes"]:
        identity=(row["condition"],row["task_id"],row["initial_state_id"])
        if identity in completed or identity not in allowed:
            raise ValueError("Duplicate or unplanned saved episode")
        key=identity[1:]
        name=f"{identity[0]}_task{identity[1]}_init{identity[2]}"
        with np.load(out/f"{name}_trajectory.npz") as saved:
            if saved["actions"].shape!=(row["steps"],7) or not np.isfinite(saved["actions"]).all():
                raise ValueError("Invalid saved trajectory")
            state=saved["initial_sim_state"].copy()
            if np.abs(reset_states.setdefault(key,state)-state).max()>1e-8:
                raise ValueError("Saved conditions have mismatched full simulator resets")
        completed.add(identity)
    try:
        for c in conditions:
            pending=[j for j in jobs if (c["name"],j["task_id"],j["initial_state_id"]) not in completed]
            if not pending:
                continue
            if c["goals"] == "generated" and worker is None:
                worker = GoalWorker(out, [j["task"] for j in jobs]+[s.text for plan in plans.values() for s in plan])
                report["world_model_startup"] = worker.ready
            policy, pre, post, options = load_policy(Path(c["checkpoint"]), c["layout"])
            if c["goals"]!="off" and options is not None and not options.goals:
                raise ValueError("Goal input requires a goal-trained Section V interface")
            if c.get("subtasks")=="oracle_predicates" and (options is None or not options.subtasks):
                raise ValueError("Subtask input requires a checkpoint trained with the Section V subtask interface")
            if c.get("subtasks")=="oracle_predicates" and not policy.pi07_prompt_coverage.get("subtask",0):
                raise ValueError("No subtask annotations were available in this checkpoint's training")
            requests=c.get("metadata",{})
            if set(requests)-{"desired_speed_steps","desired_quality","desired_mistake"}:
                raise ValueError("Unknown metadata request")
            if requests and (options is None or not options.metadata):
                raise ValueError("Metadata request requires a metadata-trained prompt interface")
            for requested,field in (("desired_speed_steps","speed_steps"),("desired_quality","quality"),("desired_mistake","mistake")):
                if requests.get(requested) is not None and not policy.pi07_prompt_coverage.get(field,0):
                    raise ValueError(f"Cannot request {field}: no known training labels recorded for this checkpoint")
            guided=MetadataGuidedSelector(policy,c["metadata_cfg_beta"]) if "metadata_cfg_beta" in c else None
            torch.cuda.reset_peak_memory_stats()
            for job in pending:
                env = writer = None
                row = {"condition": c["name"], **job}
                begin = time.perf_counter()
                try:
                    seed = job["seed"]
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    env = LiberoEnv(suite, job["task_id"], spec["suite"], obs_type="pixels_agent_pos",
                                    episode_index=job["initial_state_id"], num_steps_wait=10)
                    obs, _ = env.reset(seed=seed); policy.reset()
                    if guided is not None:guided.reset()
                    full_state = np.asarray(env._env.env.sim.get_state().flatten()).copy()
                    key = (job["task_id"], job["initial_state_id"])
                    prior = reset_states.setdefault(key, full_state)
                    row["reset_state_max_difference"] = float(np.abs(prior-full_state).max())
                    if row["reset_state_max_difference"] > 1e-8:
                        raise RuntimeError("Paired conditions did not start at the same full simulator state")
                    name = f"{c['name']}_task{job['task_id']}_init{job['initial_state_id']}"
                    writer = imageio.get_writer(str(out/f"{name}.mp4"), fps=2.5, codec="libx264", quality=7)
                    actions, states, progress, generations, inference = [], [], [], [], []
                    training_main=[];training_wrist=[];executed_actions=[]
                    action_rng=np.random.default_rng(700000+seed)
                    subtask_events=[];goal_updates=[];current_subtask=None;last_goal_subtask=None;last_goal_step=-refresh
                    goal_names, initial_progress = predicates(env)
                    row["goal_predicates"] = goal_names
                    progress.append(initial_progress)
                    clipped = 0; ended = success = False
                    goals = [torch.zeros(1, 3, 256, 256) for _ in GOALS]
                    for step in range(spec["max_steps"]):
                        obs["pixels"] = {k: np.ascontiguousarray(v) for k, v in obs["pixels"].items()}
                        if step % 8 == 0:
                            writer.append_data(np.concatenate([obs["pixels"][k] for k in ("image", "image2")], axis=1))
                            if c.get("subtasks")=="oracle_predicates":
                                selection=select_subtask(plans[job["task_id"]],predicates(env)[1])
                                selected=selection.text if selection is not None else None
                                if selected!=current_subtask:
                                    current_subtask=selected
                                    subtask_events.append({"step":step,"text":selected,"source":"simulator_predicate_oracle"})
                        if (c["goals"] == "generated" and step%8==0 and
                            (step-last_goal_step>=refresh or current_subtask!=last_goal_subtask)):
                            prompt=current_subtask or job["task"]
                            goals, timing = worker.predict(obs, prompt, 500000+seed*1000+step, f"{name}_step{step:04d}")
                            generations.append({"step": step,"prompt":prompt, **timing})
                            last_goal_step,last_goal_subtask=step,current_subtask
                        inputs = preprocess_observation(obs)
                        if (c["goals"] == "copy" and step%8==0 and
                            (step-last_goal_step>=refresh or current_subtask!=last_goal_subtask)):
                            # Match generated goals' refresh rate and 224->256
                            # image path; hold the copied picture between refreshes.
                            views=[np.asarray(Image.fromarray(obs["pixels"][k]).resize((224,224),Image.Resampling.BILINEAR))
                                   for k in ("image","image2")]
                            Image.fromarray(np.concatenate(views,axis=1)).save(out/f"{name}_step{step:04d}_goal.png")
                            np.savez_compressed(out/f"{name}_step{step:04d}_current.npz",main=obs["pixels"]["image"],
                                wrist=obs["pixels"]["image2"],state=obs["agent_pos"])
                            goals=[torch.from_numpy(np.asarray(Image.fromarray(v).resize((256,256),Image.Resampling.BILINEAR)).copy())
                                   .permute(2,0,1)[None].float()/255 for v in views]
                            goal_updates.append({"step":step,"source":"copied_current"})
                            last_goal_step,last_goal_subtask=step,current_subtask
                        text = job["task"] if options is None else runtime_prompt(job["task"], options=options,
                            subtask=current_subtask,subtask_source="simulator_predicate_oracle" if current_subtask else None,
                            goal_available=c["goals"] != "off",**requests).text
                        inputs["task"] = [text]
                        if c["layout"] != "plain":
                            for k, value in zip(GOALS, goals):
                                inputs[k] = value
                                inputs[f"{k}_padding_mask"] = torch.tensor([c["goals"] != "off"])
                        if step % 8 == 0:
                            torch.cuda.synchronize(); t = time.perf_counter()
                        with torch.inference_mode():
                            if guided is None:
                                action = post(policy.select_action(pre(inputs)))[0].cpu().numpy()
                            else:
                                negative=inputs.copy()
                                negative["task"]=[runtime_prompt(job["task"],options=options,
                                    subtask=current_subtask,subtask_source="simulator_predicate_oracle" if current_subtask else None,
                                    goal_available=c["goals"]!="off").text]
                                action=post(guided.select_action(pre(inputs),pre(negative)))[0].cpu().numpy()
                        if step % 8 == 0:
                            torch.cuda.synchronize(); inference.append(time.perf_counter()-t)
                        if action.shape != (7,) or not np.isfinite(action).all():
                            raise RuntimeError("Nonfinite or incorrectly shaped action")
                        perturbed=action+(action_rng.normal(0,c.get("action_noise_std",0.),7) if c.get("action_noise_std",0.) else 0)
                        bounded = np.clip(perturbed, -1, 1).astype(np.float32)
                        clipped += int(np.sum(np.abs(perturbed)>1))
                        actions.append(action); states.append(obs["agent_pos"].copy())
                        if record_training:
                            training_main.append(obs["pixels"]["image"].copy());training_wrist.append(obs["pixels"]["image2"].copy())
                            executed_actions.append(bounded.copy())
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
                    if record_training:
                        training_file=out/f"{name}_training.npz"
                        np.savez_compressed(training_file,main=np.asarray(training_main,dtype=np.uint8),
                            wrist=np.asarray(training_wrist,dtype=np.uint8),state=np.asarray(states,dtype=np.float32),
                            action=np.asarray(executed_actions,dtype=np.float32),predicates=np.asarray(progress[:-1],dtype=bool))
                        row["training_data"]={"file":training_file.name,"sha256":file_hash(training_file),
                            "observations":len(actions),"action_noise_std":c.get("action_noise_std",0.),
                            "excludes_post_terminal_reset_observation":True,"fps":20}
                    row.update(success=success, steps=step+1, timeout=not ended, clipped_components=clipped,
                        action_components=7*len(actions), chunk_seconds=inference, generations=generations,
                        subtask_events=subtask_events,
                        goal_updates=goal_updates,
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
    parser.add_argument("--resume",action="store_true")
    main(parser.parse_args())
