"""Fixed-observation action diagnostics for a saved Section V checkpoint.

Real futures are offline-only controls. Validation indices and normalization
are taken from the training record, never resampled after seeing outcomes.
"""
import argparse
import json
from pathlib import Path
import time
from train_goal_pilot import ROOT,np,torch,default_collate,flow_inputs,file_hash,write_json
from libero_data import make_index,make_dataset
from section_v_data import SectionVSamples
from evaluate_libero import load_policy
from goal_policy import GOALS
from generated_goal_data import GeneratedGoalCache


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    run=args.run.resolve();manifest=json.loads((run/"manifest.json").read_text())
    training=json.loads((run/"report.json").read_text())
    if training["status"]!="complete":raise ValueError("Use a finished training run")
    schedule=dict(np.load(run/"validation_schedule.npz"))
    index=make_index(manifest["dataset"],Path(manifest["dataset_root"]),args.annotations)
    if (file_hash(index.root/"meta/info.json")!=manifest["dataset_info_sha256"] or
        (file_hash(args.annotations) if args.annotations else None)!=manifest["annotations_sha256"]):
        raise ValueError("Dataset or annotations differ from training")
    split=json.loads((run/"episode_split.json").read_text())
    assert np.isin(index.episodes[schedule["current"]],split["validation_episodes"]).all()
    policy,pre,_,options=load_policy(run/"checkpoint",manifest["layout"],device=args.device)
    dataset=make_dataset(index);samples=SectionVSamples(dataset,index,schedule,options,training=False,seed=manifest["seed"])
    decoded=[samples[i] for i in range(len(samples))]
    task_ids=index.tasks[schedule["current"]]
    wrong=[int(np.flatnonzero(task_ids!=task_ids[i])[0]) for i in range(len(samples))]
    modes=["off"] if not options.goals else ["saved_real","near_real","endpoint_real","copy_current","wrong_task","off"]
    cache=GeneratedGoalCache(args.generated_cache,index) if args.generated_cache else None
    if cache:
        if not options.goals:raise ValueError("Checkpoint has no trained goal interface")
        if not all(int(c) in cache.entries for c in schedule["current"]):
            raise ValueError("Generated diagnostic needs accepted coverage of every validation observation")
        modes.append("generated")
    record={"training_run":str(run),"checkpoint_sha256":file_hash(run/"checkpoint/model.safetensors"),
        "validation_schedule_sha256":file_hash(run/"validation_schedule.npz"),"modes":modes,
        "device":args.device,"batch_size":args.batch_size,"noise_rule":"900000 + batch start",
        "units":"checkpoint-normalized 7-D action error, ignoring padded timesteps",
        "text":"same trained profile fields in every image condition; offline subtask and duration annotations",
        "copied_image_processing":"original 256px current images; live copied control additionally uses 224px resize",
        "source_sha256":{f:file_hash(ROOT/f) for f in ("evaluate_section_v_offline.py","section_v_data.py","libero_data.py","metadata_guidance.py")}}
    write_json(out/"manifest.json",record);report={"status":"running","conditions":{}}
    for mode in modes:
        rows=[];predictions=[];started=time.perf_counter()
        for start in range(0,len(decoded),args.batch_size):
            inputs=[]
            for i in range(start,min(start+args.batch_size,len(decoded))):
                sample=decoded[i].copy();current=int(schedule["current"][i])
                if mode in ("near_real","endpoint_real"):
                    future=(min(current+16,int(index.end[current])) if mode=="near_real" else int(index.segment_end[current]))
                    frame=dataset[future]
                    values=[frame[f"observation.images.{k}"] for k in ("image","image2")]
                elif mode=="copy_current":values=[sample[f"observation.images.{k}"] for k in ("image","image2")]
                elif mode=="wrong_task":values=[decoded[wrong[i]][g] for g in GOALS]
                elif mode=="generated":values=cache.images(current,sample)
                else:values=[sample[g] for g in GOALS]
                for key,value in zip(GOALS,values):
                    sample[key]=value;sample[f"{key}_padding_mask"]=torch.tensor(mode!="off")
                inputs.append(sample)
            batch=pre(default_collate(inputs));noise,_=flow_inputs(900000+start,len(inputs))
            with torch.inference_mode():pred=policy.predict_action_chunk(batch,noise=noise.to(args.device))
            assert torch.isfinite(pred).all();error=(pred-batch["action"]).float().square().cpu()
            valid=~batch["action_is_pad"].cpu();predictions.extend(pred.cpu().numpy())
            for b in range(len(inputs)):
                i=start+b;current=int(schedule["current"][i]);row={"index":current,"episode":int(index.episodes[current]),"task_id":int(task_ids[i])}
                for horizon in (8,50):
                    values=error[b,:horizon][valid[b,:horizon]]
                    row[f"sum_{horizon}"]=float(values.sum());row[f"count_{horizon}"]=values.numel()
                rows.append(row)
        metrics={f"mse_{h}":sum(r[f"sum_{h}"] for r in rows)/sum(r[f"count_{h}"] for r in rows) for h in (8,50)}
        metrics.update(observations=len(rows),wall_seconds=time.perf_counter()-started)
        report["conditions"][mode]={**metrics,"samples":rows}
        np.savez_compressed(out/f"{mode}.npz",actions=np.asarray(predictions),current=schedule["current"])
        write_json(out/"report.json",report);print(mode,metrics,flush=True)
    report["status"]="complete";write_json(out/"report.json",report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--annotations",type=Path);p.add_argument("--generated-cache",type=Path)
    p.add_argument("--device",choices=("cuda","cpu"),default="cuda");p.add_argument("--batch-size",type=int,default=8)
    main(p.parse_args())
