"""Train one declared Section V profile from the shared baseline checkpoint.

Use the same saved observation/future/flow schedule for every profile. This is
separate from the original A/B artifacts; the expanded text format needs its
own task-only control. --prepare-only audits inputs without launching training.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import shutil
import time
import traceback

from train_goal_pilot import ROOT, BASE, file_hash, state_hash, flow_inputs, write_json
from train_goal_pilot import np, torch, LeRobotDataset, DataLoader, default_collate
from goal_policy import GoalSmolVLAPolicy, configure_goals
from attention_check import AltGoalPolicy
from libero_runtime import load_smolvla_config
from section_v_data import SectionVIndex, SectionVSamples
from section_v_prompt import PromptOptions, make_section_v_processors
from generated_goal_data import GeneratedGoalCache, MixedGoalSamples
from libero_data import make_index, make_dataset


def save_training_checkpoint(target,policy,pre,post,optimizer,step,manifest,options,coverage,args):
    """Immutable checkpoint plus explicit optimizer/RNG and interface records."""
    if target.exists():raise ValueError(f"Refusing to overwrite checkpoint {target}")
    policy.save_pretrained(target);pre.save_pretrained(target);post.save_pretrained(target)
    write_json(target/"section_v_interface.json",{"prompt_options":asdict(options),"token_limit":128,
        "known_training_annotations":coverage,
        "policy_class":("attention_check.AltGoalPolicy" if args.layout=="alt" else "goal_policy.GoalSmolVLAPolicy"),
        "processor_factory":"section_v_prompt.make_section_v_processors"})
    torch.save({"step":step,"optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state_all() if args.device=="cuda" else None},target/"optimizer.pt")
    write_json(target/"training_manifest.json",manifest)
    write_json(target/"complete.json",{"step":step,"model_sha256":file_hash(target/"model.safetensors")})


def precision(device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device=="cuda" else nullcontext()


@torch.inference_mode()
def validate(policy,pre,samples,batch_size,device):
    policy.eval();total=count=0
    for start in range(0,len(samples),batch_size):
        batch=pre(default_collate([samples[i] for i in range(start,min(start+batch_size,len(samples)))]))
        noise,t=flow_inputs(900000+start,len(batch["action"]))
        with precision(device):
            loss,details=policy(batch,noise=noise.to(device),time=t.to(device))
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite validation loss")
        total+=float(loss)*details["valid_action_components"];count+=details["valid_action_components"]
    return {"flow_mse":total/count,"valid_action_components":count,"observations":len(samples)}


@torch.inference_mode()
def sampled_actions(policy,pre,samples,batch_size,device):
    """Fixed-noise sampled actions complement, but do not replace, flow loss."""
    policy.eval(); sums={8:0.,50:0.};counts={8:0,50:0};per={8:[],50:[]}
    for start in range(0,len(samples),batch_size):
        batch=pre(default_collate([samples[i] for i in range(start,min(start+batch_size,len(samples)))]))
        noise,_=flow_inputs(900000+start,len(batch["action"]))
        prediction=policy.predict_action_chunk(batch,noise=noise.to(device))
        errors=(prediction-batch["action"]).float().square()
        valid=~batch["action_is_pad"].bool()
        for b in range(len(prediction)):
            for horizon in sums:
                values=errors[b,:horizon][valid[b,:horizon]]
                if not values.numel() or not torch.isfinite(values).all():
                    raise RuntimeError("Invalid sampled-action evaluation")
                sums[horizon]+=float(values.sum());counts[horizon]+=values.numel()
                per[horizon].append(float(values.mean()))
    return {f"first_{h}":{"mse":sums[h]/counts[h],"valid_components":counts[h],"per_observation":per[h]}
            for h in sums}


def main(args):
    out=args.output.resolve()
    if (out/"manifest.json").exists():raise RuntimeError("Choose a new output directory; existing runs are preserved")
    if args.steps<1 or args.batch_size<1:raise ValueError("Positive steps and batch size required")
    if args.lr<=0 or args.warmup_steps<1 or args.eval_every<1 or args.checkpoint_every<1:
        raise ValueError("Invalid optimization settings")
    if not args.prepare_only and args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; prepare-only is CPU-safe, or explicitly select --device cpu")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    source=args.schedule_dir.resolve()
    index=make_index(args.dataset,args.dataset_root,args.annotations)
    schedule_manifest=source/"schedule_manifest.json"
    if schedule_manifest.exists():
        provenance=json.loads(schedule_manifest.read_text())
        if (Path(provenance["dataset_root"]).resolve()!=index.root.resolve() or
            provenance["dataset_info_sha256"]!=file_hash(index.root/"meta/info.json")):
            raise ValueError("Schedule belongs to a different dataset/revision")
    elif args.dataset!="libero_10":
        raise ValueError("New datasets require a dataset-identified schedule manifest")
    train_schedule=dict(np.load(source/"train_schedule.npz"))
    validation_schedule=dict(np.load(source/"validation_schedule.npz"))
    examples=args.steps*args.batch_size
    if examples>len(train_schedule["current"]):raise ValueError("Requested budget exceeds the fixed source schedule")
    train_schedule={k:v[:examples] for k,v in train_schedule.items()}
    for schedule in (train_schedule,validation_schedule):
        if args.goal_sampling=="endpoint":
            schedule["future"]=index.segment_end[schedule["current"]]
            schedule["segment_endpoint"]=np.ones(len(schedule["current"]),dtype=bool)
        elif args.goal_sampling=="near":
            rng=np.random.default_rng(args.seed+7+(0 if schedule is train_schedule else 4))
            current=schedule["current"]
            schedule["future"]=np.minimum(current+rng.integers(8,17,len(current)),index.end[current])
            schedule["segment_endpoint"]=np.zeros(len(current),dtype=bool)
    split=json.loads((source/"episode_split.json").read_text())
    for schedule,episodes in ((train_schedule,split["train_episodes"]),(validation_schedule,split["validation_episodes"])):
        for field in ("current","future"):
            if not np.isin(index.episodes[schedule[field]],episodes).all():raise ValueError("Split leakage")
    options=PromptOptions.profile(args.profile,speed_bin_steps=args.speed_bin_steps,
                                  subtask_drop_given_goal=args.subtask_drop_given_goal,
                                  goal_probability=args.goal_probability)
    dataset=make_dataset(index)
    train=SectionVSamples(dataset,index,train_schedule,options,training=True,seed=args.seed)
    heldout=SectionVSamples(dataset,index,validation_schedule,options,training=False,seed=args.seed)
    generated=None
    if args.generated_cache:
        cache=GeneratedGoalCache(args.generated_cache,index,allow_unreviewed=args.allow_unreviewed_generated)
        # Training cache must contain training episodes only, even if only a
        # subset would be selected by this run's mixture draws.
        if not all(int(index.episodes[c]) in split["train_episodes"] for c in cache.entries):
            raise ValueError("Generated training cache contains held-out episodes")
        train=MixedGoalSamples(train,cache,args.generated_probability,args.seed)
        generated={"manifest":str(args.generated_cache.resolve()),"sha256":file_hash(args.generated_cache),
            "requested_probability_given_cached_visible_goal":args.generated_probability,
            "allow_unreviewed":args.allow_unreviewed_generated,"covered_visible_goals":int(train.covered_visible),
            "actual_generated_goals":int(train.generated.sum()),"accepted_cache_entries":len(cache.entries)}
        if not train.generated.any():raise ValueError("Cache mixture selects zero generated goals; inspect coverage/reviews")
    train.write_audit(out/"train_prompts.jsonl");heldout.write_audit(out/"validation_prompts.jsonl")
    for name,schedule in (("train",train.schedule),("validation",heldout.schedule)):
        np.savez_compressed(out/f"{name}_schedule.npz",**schedule)
    write_json(out/"episode_split.json",split)
    device="cpu" if args.prepare_only else args.device
    base=args.base.resolve()
    cfg,_=load_smolvla_config(base,device=device)
    configure_goals(cfg)
    pre,post=make_section_v_processors(cfg,base,max_length=128)
    lengths=pre.token_counts([p.text for p in train.prompts+heldout.prompts])
    if max(lengths)>128:raise ValueError("Prompt exceeds configured 128-token budget")
    coverage={field:sum(p.fields[field]["value"] is not None for p in train.prompts)
              for field in ("subtask","speed_steps","quality","mistake")}
    source_files=("train_section_v.py","section_v_prompt.py","section_v_data.py","goal_policy.py",
                  "goal_data.py","libero_runtime.py","train_goal_pilot.py","attention_check.py","generated_goal_data.py","libero_data.py","rollout_data.py")
    manifest={"profile":args.profile,"prompt_options":asdict(options),"seed":args.seed,
        "output":str(out),"device":args.device,"resume_checkpoint":str(args.resume.resolve()) if args.resume else None,
        "steps":args.steps,"batch_size":args.batch_size,"examples":len(train),
        "base_checkpoint":str(base),"base_checkpoint_sha256":file_hash(base/"model.safetensors"),
        "layout":args.layout,"goal_sampling":args.goal_sampling,
        "dataset":args.dataset,"dataset_root":str(index.root.resolve()),
        "dataset_info_sha256":file_hash(index.root/"meta/info.json"),
        "generated_training_goals":generated,
        "uncertain_subtask_episodes":index.uncertain_subtask_episodes,
        "source_schedule_dir":str(source),"train_schedule_sha256":file_hash(out/"train_schedule.npz"),
        "validation_schedule_sha256":file_hash(out/"validation_schedule.npz"),
        "annotations_sha256":file_hash(args.annotations) if args.annotations else None,
        "source_sha256":{f:file_hash(ROOT/f) for f in source_files},
        "known_training_annotations":coverage,"goal_present_count":int(train.schedule["goal_present"].sum()),
        "max_tokens":max(lengths),"token_limit":128,"silent_truncation":False,
        "optimizer":{"name":"AdamW","lr":args.lr,"betas":[.9,.95],"eps":1e-8,"weight_decay":1e-10,
                     "warmup_steps":args.warmup_steps,"grad_clip":10.},
        "flow_seed_rule":"seed + 100000 + update; same explicit noise/time as original pilot",
        "notes":["All profiles include Control Mode: ee and the same 128-token budget.",
                 "Train a fresh task-only profile control; old A has different text formatting/length.",
                 "Demo subtask labels and durations are offline supervision, not live observations.",
                 "Unknown quality/mistake annotations stay absent; success is not converted to quality.",
                 "Full profile does not establish mixed-quality learning without validated quality variation.",
                 "Metadata CFG (Section VII), joint control and a live subtask planner are not implemented here."]}
    write_json(out/"manifest.json",manifest)
    (out/"source").mkdir()
    for name in source_files:shutil.copy2(ROOT/name,out/"source"/name)
    report={"status":"prepared","optimizer_updates":0,"validation":[],"updates":[]}
    if args.prepare_only:
        pre.save_pretrained(out/"processors");post.save_pretrained(out/"processors")
        write_json(out/"report.json",report)
        print(json.dumps({"status":"prepared","profile":args.profile,"examples":len(train),
                          "annotation_coverage":coverage,"max_tokens":max(lengths)}),flush=True)
        return
    report["status"]="running";write_json(out/"report.json",report)
    try:
        policy_class=AltGoalPolicy if args.layout=="alt" else GoalSmolVLAPolicy
        resume_state=None;start_step=0
        if args.resume:
            checkpoint=args.resume.resolve()
            if not (checkpoint/"complete.json").exists():raise ValueError("Resume checkpoint was not finalized")
            completion=json.loads((checkpoint/"complete.json").read_text())
            if file_hash(checkpoint/"model.safetensors")!=completion["model_sha256"]:
                raise ValueError("Resume checkpoint hash mismatch")
            prior=json.loads((checkpoint/"training_manifest.json").read_text())
            for field in ("profile","prompt_options","seed","batch_size","base_checkpoint_sha256","layout",
                          "goal_sampling","dataset","dataset_root","dataset_info_sha256","annotations_sha256",
                          "optimizer","device","source_sha256"):
                if prior[field]!=manifest[field]:raise ValueError(f"Resume mismatch: {field}")
            if prior.get("generated_training_goals") or manifest.get("generated_training_goals"):
                if prior.get("generated_training_goals")!=manifest.get("generated_training_goals"):
                    raise ValueError("Resume generated-image mixture changed")
            prior_dir=Path(prior["output"])
            for name,new in (("train",train.schedule),("validation",heldout.schedule)):
                old=dict(np.load(prior_dir/f"{name}_schedule.npz"))
                for key,values in old.items():
                    if len(new[key])<len(values) or not np.array_equal(values,new[key][:len(values)]):
                        raise ValueError(f"Resume changes the committed {name} schedule: {key}")
            resume_state=torch.load(checkpoint/"optimizer.pt",map_location="cpu",weights_only=True)
            start_step=int(resume_state["step"])
            if start_step!=completion["step"] or not 0<start_step<args.steps:
                raise ValueError("Resume must extend a completed checkpoint step")
        policy=policy_class.from_pretrained(args.resume or base,config=cfg,strict=True)
        params=[p for p in policy.parameters() if p.requires_grad]
        report["initial_state_sha256"]=state_hash(policy)
        report["trainable_parameters"]=sum(p.numel() for p in params)
        backbone_before=state_hash(policy.model.vlm_with_expert.vlm)
        optimizer=torch.optim.AdamW(params,lr=args.lr,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            torch.set_rng_state(resume_state["torch_rng"])
            if device=="cuda" and resume_state["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
            del resume_state
        report["optimizer_updates"]=start_step
        def evaluation(step):
            return {"step":step,**validate(policy,pre,heldout,args.batch_size,device),
                    "sampled_actions":sampled_actions(policy,pre,heldout,args.batch_size,device)}
        report["validation"].append(evaluation(start_step))
        write_json(out/"report.json",report)
        if device=="cuda":torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        remaining=torch.utils.data.Subset(train,range(start_step*args.batch_size,len(train)))
        # Iterator initialization must not consume the model's global RNG;
        # otherwise restarting mid-run changes subsequent stochastic layers.
        loader_rng=torch.Generator().manual_seed(args.seed)
        for step,raw in enumerate(DataLoader(remaining,batch_size=args.batch_size,shuffle=False,num_workers=0,
                                            generator=loader_rng),start_step+1):
            policy.train();batch=pre(raw)
            noise,t=flow_inputs(args.seed+100000+step,len(raw["task"]))
            for group in optimizer.param_groups:group["lr"]=args.lr*min(step/args.warmup_steps,1.)
            with precision(device):loss,_=policy(batch,noise=noise.to(device),time=t.to(device))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            report["optimizer_updates"]=step
            report["updates"].append({"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm)})
            if step%args.eval_every==0 or step==args.steps:
                report["validation"].append(evaluation(step))
                write_json(out/"report.json",report);print(f"{args.profile} step {step}",flush=True)
            if step%args.checkpoint_every==0 and step<args.steps:
                saved=out/"checkpoints"/f"step_{step:06d}"
                save_training_checkpoint(saved,policy,pre,post,optimizer,step,manifest,options,coverage,args)
                report["latest_checkpoint"]=str(saved)
                write_json(out/"report.json",report)
        report["training_and_validation_seconds"]=time.perf_counter()-begin
        report["frozen_backbone_unchanged"]=backbone_before==state_hash(policy.model.vlm_with_expert.vlm)
        assert report["frozen_backbone_unchanged"]
        if device=="cuda":report["peak_allocated_gb"]=torch.cuda.max_memory_allocated()/1e9
        if options.goals:
            dropped=SectionVSamples(dataset,index,validation_schedule,replace(options,goals=False),training=False,seed=args.seed)
            report["final_goals_dropped"]={**validate(policy,pre,dropped,args.batch_size,device),
                "sampled_actions":sampled_actions(policy,pre,dropped,args.batch_size,device)}
        target=out/"checkpoint"
        save_training_checkpoint(target,policy,pre,post,optimizer,args.steps,manifest,options,coverage,args)
        report["latest_checkpoint"]=str(target)
        final_hash=state_hash(policy)
        del policy,optimizer,params,loss,batch;gc.collect()
        if device=="cuda":torch.cuda.empty_cache()
        reload_cfg,_=load_smolvla_config(target,device=device)
        reloaded=policy_class.from_pretrained(target,config=reload_cfg,strict=True)
        report["checkpoint_roundtrip_exact"]=state_hash(reloaded)==final_hash
        assert report["checkpoint_roundtrip_exact"]
        report["status"]="complete"
    except Exception:
        report["status"]="failed";report["error"]=traceback.format_exc();raise
    finally:write_json(out/"report.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--profile",choices=("task","subtask","goal","subtask_goal","full"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--schedule-dir",type=Path,default=ROOT/"artifacts/goal_training_1000")
    p.add_argument("--annotations",type=Path)
    p.add_argument("--dataset",choices=("libero_10","libero_goal","rollouts"),default="libero_10")
    p.add_argument("--dataset-root",type=Path)
    p.add_argument("--generated-cache",type=Path)
    p.add_argument("--generated-probability",type=float,default=.5)
    p.add_argument("--allow-unreviewed-generated",action="store_true",
                   help="Explicit exploratory condition; unreviewed images are otherwise excluded.")
    p.add_argument("--base",type=Path,default=Path("/media/skr/storage/models/smolvla_libero_90_pi07"))
    p.add_argument("--layout",choices=("alt","codex"),default="alt")
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--warmup-steps",type=int,default=20)
    p.add_argument("--eval-every",type=int,default=100)
    p.add_argument("--checkpoint-every",type=int,default=500)
    p.add_argument("--resume",type=Path,help="Resume a finalized checkpoint into a fresh output directory")
    p.add_argument("--goal-probability",type=float,default=.25)
    p.add_argument("--goal-sampling",choices=("saved","near","endpoint"),default="saved")
    p.add_argument("--steps",type=int,default=1000);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--speed-bin-steps",type=int,default=50)
    p.add_argument("--subtask-drop-given-goal",type=float,default=.30,
                   help="Paper recipe: .30; use 0 in all text arms for a controlled text/image factorial.")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--prepare-only",action="store_true")
    main(p.parse_args())
