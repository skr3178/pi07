"""Starting-point equivalence and small-data fitting, not a benchmark.

Uses training episodes only. Fit a fixed balanced 32-observation subset with
fresh flow noise at each update and fixed evaluation noise. Logs flow loss,
sampled action error, gradient norms and clipping. No pretrained file changes.
"""
import argparse
import gc
import json
import time
from pathlib import Path

from train_goal_pilot import (ROOT, torch, np, GoalIndex, GoalSamples, LeRobotDataset,
                              default_collate, flow_inputs, file_hash, write_json)
from evaluate_libero import load_policy
from train_section_v import precision, sampled_actions, validate


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.steps<1 or args.samples<10 or args.samples%args.batch_size:
        raise ValueError("Positive steps, at least ten samples, and complete batches required")
    torch.set_num_threads(4)
    if args.device=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    index=GoalIndex(ROOT/"data/libero_10_subtask")
    split=json.loads((ROOT/"artifacts/goal_training_1000/episode_split.json").read_text())
    rng=np.random.default_rng(args.seed);current=[]
    for t in range(10):
        available=np.flatnonzero((index.tasks==t)&np.isin(index.episodes,split["train_episodes"]))
        current.extend(rng.choice(available,args.samples//10+int(t<args.samples%10),replace=False))
    current=np.asarray(current);rng.shuffle(current)
    future=np.minimum(current+rng.integers(8,17,len(current)),index.end[current])
    schedule={"current":current,"future":future,"goal_present":np.ones(len(current),dtype=bool)}
    np.savez_compressed(out/"schedule.npz",**schedule)
    dataset=LeRobotDataset("lerobot/libero_10_subtask",root=index.root,video_backend="torchcodec",
        delta_timestamps={"action":[i/index.info["fps"] for i in range(50)]})
    arm=GoalSamples(dataset,schedule,args.arm)
    samples=[arm[i] for i in range(len(arm))]
    write_json(out/"manifest.json",{"purpose":"training-subset sanity check; no generalization claim",
        "base":str(args.base),"base_sha256":file_hash(args.base/"model.safetensors"),
        "args":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "current":current.tolist(),"episodes":index.episodes[current].tolist(),
        "source_sha256":{f:file_hash(ROOT/f) for f in ("check_policy_training.py","train_section_v.py",
            "evaluate_libero.py","attention_check.py","goal_policy.py","train_goal_pilot.py")}})
    report={"status":"running","runs":[]}
    write_json(out/"report.json",report)
    # Same preprocessed inputs and sampled noise through the two layouts.
    predictions={}
    for layout in ("plain","alt"):
        torch.manual_seed(args.seed)
        policy,pre,post,_=load_policy(args.base,layout,args.device)
        raw=default_collate(samples[:args.batch_size])
        for k in ("observation.images.goal_camera1","observation.images.goal_camera2"):
            raw[k+"_padding_mask"][:]=False
        batch=pre(raw);noise,_=flow_inputs(900000,len(raw["task"]))
        with torch.inference_mode():
            predictions[layout]=policy.predict_action_chunk(batch,noise=noise.to(args.device)).cpu()
        del policy,pre,post,batch;gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
    difference=predictions["plain"]-predictions["alt"]
    report["zero_update_layout_check"]={"max_abs_action_difference":float(difference.abs().max()),
        "rms_action_difference":float(difference.square().mean().sqrt()),
        "first_8_max_abs_action_difference":float(difference[:,:8].abs().max()),
        "first_8_rms_action_difference":float(difference[:,:8].square().mean().sqrt()),
        "observations":args.batch_size,"goal_masks":False}
    write_json(out/"report.json",report)
    print("zero update",report["zero_update_layout_check"],flush=True)
    if args.equivalence_only:
        report["status"]="complete";report["optimizer_updates"]=0
        write_json(out/"report.json",report)
        return
    for lr in args.lrs:
        torch.manual_seed(args.seed)
        policy,pre,post,_=load_policy(args.base,"alt",args.device)
        params=[p for p in policy.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(params,lr=lr,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
        row={"lr":lr,"curve":[],"updates":[]};report["runs"].append(row)
        def assess(step):
            measured={"step":step,**validate(policy,pre,samples,args.batch_size,args.device),
                "sampled_actions":sampled_actions(policy,pre,samples,args.batch_size,args.device)}
            row["curve"].append(measured)
            write_json(out/"report.json",report)
            print(f"lr={lr} step={step} flow={measured['flow_mse']:.5f} "
                  f"action8={measured['sampled_actions']['first_8']['mse']:.5f}",flush=True)
        assess(0);begin=time.perf_counter()
        order=np.random.default_rng(args.seed).permutation(args.samples)
        for step in range(1,args.steps+1):
            offset=((step-1)*args.batch_size)%args.samples
            raw=default_collate([samples[i] for i in order[offset:offset+args.batch_size]])
            batch=pre(raw);noise,t=flow_inputs(args.seed+100000+step,len(raw["task"]))
            for group in optimizer.param_groups:group["lr"]=lr*min(step/args.warmup_steps,1.)
            policy.train()
            with precision(args.device):loss,_=policy(batch,noise=noise.to(args.device),time=t.to(args.device))
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite fit loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            row["updates"].append({"step":step,"loss":float(loss.detach()),"gradient_norm":float(norm)})
            if step%args.eval_every==0 or step==args.steps:assess(step)
        row["seconds"]=time.perf_counter()-begin
        row["clipped_gradient_fraction"]=float(np.mean([r["gradient_norm"]>10 for r in row["updates"]]))
        if args.arm=="B":
            row["training_subset_goal_ablations"]={}
            for mode in ("copy","wrong_task","off"):
                changed=[]
                for i,sample in enumerate(samples):
                    item=sample.copy()
                    partner=next(j for j in range(len(samples)) if index.tasks[current[j]]!=index.tasks[current[i]])
                    for cam,goal in zip(("image","image2"),("goal_camera1","goal_camera2")):
                        key=f"observation.images.{goal}"
                        if mode=="copy":item[key]=sample[f"observation.images.{cam}"]
                        elif mode=="wrong_task":item[key]=samples[partner][key]
                        else:item[key+"_padding_mask"]=torch.tensor(False)
                    changed.append(item)
                row["training_subset_goal_ablations"][mode]=sampled_actions(policy,pre,changed,args.batch_size,args.device)
        del policy,pre,post,optimizer,params,loss,batch;gc.collect()
        if args.device=="cuda":torch.cuda.empty_cache()
    report["status"]="complete";write_json(out/"report.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base",type=Path,default=Path("/media/skr/storage/models/smolvla_libero_90_pi07"))
    p.add_argument("--steps",type=int,default=200);p.add_argument("--samples",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=8);p.add_argument("--eval-every",type=int,default=50)
    p.add_argument("--lrs",type=float,nargs="+",default=[1e-5,1e-4]);p.add_argument("--warmup-steps",type=int,default=20)
    p.add_argument("--seed",type=int,default=20261104);p.add_argument("--arm",choices=("A","B"),default="A")
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    p.add_argument("--equivalence-only",action="store_true")
    main(p.parse_args())
