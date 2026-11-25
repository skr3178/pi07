"""Generate a local, observation-verified goal cache and contact sheets.

The generator receives only current images/state and a declared instruction.
Real endpoints are decoded afterwards solely for visual review. Ratings begin
as unreviewed; raw pixel distances are not semantic-success labels.
"""
import argparse
import json
from pathlib import Path
import shutil
import traceback

from run_goal_development import ROOT, GoalWorker, np, torch, file_hash, write_json
from train_goal_pilot import LeRobotDataset
from section_v_data import SectionVIndex
from generated_goal_data import tensor_rgb,observation_digest,generation_prompt
from libero_data import make_index,make_dataset
from PIL import Image,ImageDraw


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    index=make_index(args.dataset,args.dataset_root)
    schedule=dict(np.load(args.schedule))
    eligible=np.asarray(schedule.get("goal_present",np.ones(len(schedule["current"]),dtype=bool)),dtype=bool)
    candidates=np.unique(schedule["current"][eligible])
    if args.limit<10:raise ValueError("Use at least ten entries for task coverage")
    rng=np.random.default_rng(args.seed);chosen=[]
    for task in range(10):
        pool=candidates[index.tasks[candidates]==task];rng.shuffle(pool)
        chosen.extend(pool[:args.limit//10+int(task<args.limit%10)])
    chosen=[int(x) for x in chosen]
    dataset=make_dataset(index,chunk_size=None)
    prompts=[generation_prompt(index,c,args.prompt_mode) for c in chosen]
    model=Path("/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b/model.pt")
    manifest={"status":"running","dataset_root":str(index.root.resolve()),
        "dataset_info_sha256":file_hash(index.root/"meta/info.json"),
        "schedule":str(args.schedule.resolve()),"schedule_sha256":file_hash(args.schedule),
        "prompt_mode":args.prompt_mode,"seed":args.seed,"entries":[],
        "selection":"balanced task sample of unique observations with visible goals in the supplied schedule",
        "eligible_unique_observations":len(candidates),
        "world_model_sha256":file_hash(model),"denoising_steps":10,"calls_per_prediction":1,
        "world_model_config_sha256":file_hash(model.with_name("config.yaml")),
        "world_model_stats_sha256":file_hash(model.with_name("dataset_stats.json")),
        "review_definition":"Visual review of instruction-consistent near-term progress, both cameras and physical plausibility; not endpoint success or pixel MSE",
        "source_sha256":{f:file_hash(ROOT/f) for f in ("prepare_generated_goals.py","generated_goal_data.py",
            "section_v_data.py","goal_data.py","imagewam_goal_worker.py","run_goal_development.py","libero_data.py")}}
    (out/"source").mkdir()
    for name in manifest["source_sha256"]:shutil.copy2(ROOT/name,out/"source"/name)
    write_json(out/"manifest.json",manifest)
    worker=None;rows=[]
    try:
        worker=GoalWorker(out,prompts);manifest["startup"]=worker.ready
        for i,(current,prompt) in enumerate(zip(chosen,prompts)):
            sample=dataset[current]
            views=[tensor_rgb(sample[f"observation.images.{k}"]) for k in ("image","image2")]
            obs={"pixels":dict(zip(("image","image2"),views)),"agent_pos":np.asarray(sample["observation.state"])}
            name=f"index{current}"
            _,timing=worker.predict(obs,prompt,args.seed+i,name)
            target=out/f"{name}_goal.png"
            entry={"current_index":current,"episode_index":int(index.episodes[current]),
                "frame_index":int(index.frames[current]),"task_id":int(index.tasks[current]),
                "prompt":prompt,"seed":args.seed+i,"observation_sha256":observation_digest(sample),
                "image":target.name,"image_sha256":file_hash(target),"review":"unreviewed",**timing}
            manifest["entries"].append(entry)
            # Only the reviewer gets the real endpoint; it is not a worker input.
            endpoint=int(index.segment_end[current]);future=dataset[endpoint]
            near_index=min(current+16,int(index.end[current]));near=dataset[near_index]
            predicted=Image.open(target).convert("RGB")
            reference=Image.fromarray(np.concatenate([tensor_rgb(future[f"observation.images.{k}"])
                                                     for k in ("image","image2")],axis=1))
            near_image=Image.fromarray(np.concatenate([tensor_rgb(near[f"observation.images.{k}"])
                                                      for k in ("image","image2")],axis=1))
            current_image=Image.fromarray(np.concatenate(views,axis=1))
            row=Image.new("RGB",(1792,264),"white");draw=ImageDraw.Draw(row)
            draw.text((6,4),f"task {entry['task_id']} ep {entry['episode_index']} frame {entry['frame_index']}: {prompt}",fill="black")
            for j,(label,image) in enumerate((("CURRENT",current_image),("GENERATED +16",predicted),
                                             (f"REAL NEAR +{near_index-current}",near_image),(f"REAL ENDPOINT +{endpoint-current}",reference))):
                draw.text((448*j+6,22),label,fill="black");row.paste(image.resize((448,224)),(448*j,40))
            rows.append(row)
            write_json(out/"manifest.json",manifest)
            print(f"generated {i+1}/{len(chosen)} index={current}",flush=True)
        for page in range(0,len(rows),5):
            group=rows[page:page+5];grid=Image.new("RGB",(1792,264*len(group)),"white")
            for j,row in enumerate(group):grid.paste(row,(0,264*j))
            grid.save(out/f"review_{page//5:02d}.png")
        manifest["status"]="complete"
    except Exception:
        manifest["status"]="failed";manifest["error"]=traceback.format_exc();raise
    finally:
        if worker is not None:worker.close()
        write_json(out/"manifest.json",manifest)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    p.add_argument("--schedule",type=Path,default=ROOT/"artifacts/goal_training_1000/train_schedule.npz")
    p.add_argument("--limit",type=int,default=20);p.add_argument("--seed",type=int,default=20261104)
    p.add_argument("--prompt-mode",choices=("task","subtask"),default="subtask")
    p.add_argument("--dataset",choices=("libero_10","libero_goal"),default="libero_10")
    p.add_argument("--dataset-root",type=Path)
    main(p.parse_args())
