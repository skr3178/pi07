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
from libero_runtime import load_smolvla_config
from section_v_data import SectionVIndex, SectionVSamples
from section_v_prompt import PromptOptions, make_section_v_processors


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


def main(args):
    out=args.output.resolve()
    if (out/"manifest.json").exists():raise RuntimeError("Choose a new output directory; existing runs are preserved")
    if args.steps<1 or args.batch_size<1:raise ValueError("Positive steps and batch size required")
    if not args.prepare_only and args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; prepare-only is CPU-safe, or explicitly select --device cpu")
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    source=args.schedule_dir.resolve()
    index=SectionVIndex(ROOT/"data/libero_10_subtask",args.annotations)
    train_schedule=dict(np.load(source/"train_schedule.npz"))
    validation_schedule=dict(np.load(source/"validation_schedule.npz"))
    examples=args.steps*args.batch_size
    if examples>len(train_schedule["current"]):raise ValueError("Requested budget exceeds the fixed source schedule")
    train_schedule={k:v[:examples] for k,v in train_schedule.items()}
    split=json.loads((source/"episode_split.json").read_text())
    for schedule,episodes in ((train_schedule,split["train_episodes"]),(validation_schedule,split["validation_episodes"])):
        for field in ("current","future"):
            if not np.isin(index.episodes[schedule[field]],episodes).all():raise ValueError("Split leakage")
    options=PromptOptions.profile(args.profile,speed_bin_steps=args.speed_bin_steps,
                                  subtask_drop_given_goal=args.subtask_drop_given_goal)
    dataset=LeRobotDataset("lerobot/libero_10_subtask",root=index.root,video_backend="torchcodec",
                          delta_timestamps={"action":[i/index.info["fps"] for i in range(50)]})
    train=SectionVSamples(dataset,index,train_schedule,options,training=True,seed=args.seed)
    heldout=SectionVSamples(dataset,index,validation_schedule,options,training=False,seed=args.seed)
    train.write_audit(out/"train_prompts.jsonl");heldout.write_audit(out/"validation_prompts.jsonl")
    for name,schedule in (("train",train.schedule),("validation",heldout.schedule)):
        np.savez_compressed(out/f"{name}_schedule.npz",**schedule)
    write_json(out/"episode_split.json",split)
    device="cpu" if args.prepare_only else args.device
    cfg,_=load_smolvla_config(BASE,device=device)
    configure_goals(cfg)
    pre,post=make_section_v_processors(cfg,BASE,max_length=128)
    lengths=pre.token_counts([p.text for p in train.prompts+heldout.prompts])
    if max(lengths)>128:raise ValueError("Prompt exceeds configured 128-token budget")
    coverage={field:sum(p.fields[field]["value"] is not None for p in train.prompts)
              for field in ("subtask","speed_steps","quality","mistake")}
    source_files=("train_section_v.py","section_v_prompt.py","section_v_data.py","goal_policy.py",
                  "goal_data.py","libero_runtime.py","train_goal_pilot.py")
    manifest={"profile":args.profile,"prompt_options":asdict(options),"seed":args.seed,
        "steps":args.steps,"batch_size":args.batch_size,"examples":len(train),
        "base_checkpoint":str(BASE),"base_checkpoint_sha256":file_hash(BASE/"model.safetensors"),
        "source_schedule_dir":str(source),"train_schedule_sha256":file_hash(out/"train_schedule.npz"),
        "validation_schedule_sha256":file_hash(out/"validation_schedule.npz"),
        "annotations_sha256":file_hash(args.annotations) if args.annotations else None,
        "source_sha256":{f:file_hash(ROOT/f) for f in source_files},
        "known_training_annotations":coverage,"goal_present_count":int(train.schedule["goal_present"].sum()),
        "max_tokens":max(lengths),"token_limit":128,"silent_truncation":False,
        "optimizer":{"name":"AdamW","lr":1e-4,"betas":[.9,.95],"eps":1e-8,"weight_decay":1e-10,
                     "warmup_steps":20,"grad_clip":10.},
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
        policy=GoalSmolVLAPolicy.from_pretrained(BASE,config=cfg,strict=True)
        params=[p for p in policy.parameters() if p.requires_grad]
        report["initial_state_sha256"]=state_hash(policy)
        report["trainable_parameters"]=sum(p.numel() for p in params)
        backbone_before=state_hash(policy.model.vlm_with_expert.vlm)
        optimizer=torch.optim.AdamW(params,lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        report["validation"].append({"step":0,**validate(policy,pre,heldout,args.batch_size,device)})
        if device=="cuda":torch.cuda.reset_peak_memory_stats()
        begin=time.perf_counter()
        for step,raw in enumerate(DataLoader(train,batch_size=args.batch_size,shuffle=False,num_workers=0),1):
            policy.train();batch=pre(raw)
            noise,t=flow_inputs(args.seed+100000+step,len(raw["task"]))
            for group in optimizer.param_groups:group["lr"]=1e-4*min(step/20,1.)
            with precision(device):loss,_=policy(batch,noise=noise.to(device),time=t.to(device))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            report["optimizer_updates"]=step
            report["updates"].append({"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm)})
            if step%100==0 or step==args.steps:
                report["validation"].append({"step":step,**validate(policy,pre,heldout,args.batch_size,device)})
                write_json(out/"report.json",report);print(f"{args.profile} step {step}",flush=True)
        report["training_and_validation_seconds"]=time.perf_counter()-begin
        report["frozen_backbone_unchanged"]=backbone_before==state_hash(policy.model.vlm_with_expert.vlm)
        assert report["frozen_backbone_unchanged"]
        if device=="cuda":report["peak_allocated_gb"]=torch.cuda.max_memory_allocated()/1e9
        if options.goals:
            dropped=SectionVSamples(dataset,index,validation_schedule,replace(options,goals=False),training=False,seed=args.seed)
            report["final_goals_dropped"]=validate(policy,pre,dropped,args.batch_size,device)
        target=out/"checkpoint"
        policy.save_pretrained(target);pre.save_pretrained(target);post.save_pretrained(target)
        write_json(target/"section_v_interface.json",{"prompt_options":asdict(options),"token_limit":128,
            "policy_class":"goal_policy.GoalSmolVLAPolicy","processor_factory":"section_v_prompt.make_section_v_processors"})
        final_hash=state_hash(policy)
        torch.save({"step":args.steps,"optimizer":optimizer.state_dict()},target/"optimizer.pt")
        del policy,optimizer,params,loss,batch;gc.collect()
        if device=="cuda":torch.cuda.empty_cache()
        reload_cfg,_=load_smolvla_config(target,device=device)
        reloaded=GoalSmolVLAPolicy.from_pretrained(target,config=reload_cfg,strict=True)
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
    p.add_argument("--steps",type=int,default=1000);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--seed",type=int,default=20261029)
    p.add_argument("--speed-bin-steps",type=int,default=50)
    p.add_argument("--subtask-drop-given-goal",type=float,default=.30,
                   help="Paper recipe: .30; use 0 in all text arms for a controlled text/image factorial.")
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    p.add_argument("--prepare-only",action="store_true")
    main(p.parse_args())
