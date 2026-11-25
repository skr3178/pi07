"""Export recorded simulator transitions to a lossless local training dataset.

No success-to-quality conversion. Review templates start with missing labels.
"""
import argparse
import json
from pathlib import Path
import shutil
from train_goal_pilot import ROOT,np,file_hash,write_json
from section_v_data import SectionVIndex
from subtask_controller import build_plan,select_subtask


def main(args):
    run=args.run.resolve();report=json.loads((run/"report.json").read_text())
    manifest=json.loads((run/"manifest.json").read_text())
    if report["status"]!="complete" or manifest.get("phase")!="training_collection":
        raise ValueError("Export a completed, explicitly declared training collection")
    canonical=SectionVIndex(ROOT/"data/libero_10_subtask")
    task_ids={text:i for i,text in canonical.task_names.items()}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/"meta").mkdir()
    episodes=[];offset=0
    for ep,row in enumerate(report["episodes"]):
        if "error" in row:raise ValueError("Collection contains failed executions")
        recorded=row["training_data"];source=run/recorded["file"]
        if file_hash(source)!=recorded["sha256"]:raise ValueError("Collection data hash mismatch")
        if row["initial_state_id"] in range(2,12):raise ValueError("Reserved final state in training data")
        folder=out/"episodes"/f"episode_{ep:05d}";folder.mkdir(parents=True)
        with np.load(source) as arrays:
            n=len(arrays["action"])
            if n!=row["steps"]:raise ValueError("Wrong action count")
            assert arrays["main"].shape==arrays["wrist"].shape==(n,256,256,3)
            assert arrays["state"].shape==(n,8) and arrays["action"].shape==(n,7)
            assert np.isfinite(arrays["action"]).all() and np.isfinite(arrays["state"]).all()
            assert np.max(np.abs(arrays["action"]))<=1
            plan=build_plan(row["goal_predicates"])
            texts=[]
            for truth in arrays["predicates"]:
                selected=select_subtask(plan,[bool(x) for x in truth]);texts.append(selected.text if selected else "")
            for key in ("main","wrist","state","action","predicates"):
                np.save(folder/f"{key}.npy",arrays[key],allow_pickle=False)
            np.save(folder/"subtask_text.npy",np.asarray(texts),allow_pickle=False)
        hashes={p.name:file_hash(p) for p in folder.glob("*.npy")}
        episodes.append({"episode_index":ep,"start":offset,"end_exclusive":offset+n,
            "task_id":int(task_ids[row["task"]]),"task":row["task"],"condition":row["condition"],
            "simulator_task_id":row["task_id"],"initial_state_id":row["initial_state_id"],
            "success":row["success"],"max_predicate_fraction":row["max_predicate_fraction"],
            "source_recording":str(source),"source_sha256":recorded["sha256"],"files_sha256":hashes,
            "folder":str(folder.relative_to(out)),"subtask_source":"current simulator predicate oracle; not demonstration time"})
        offset+=n
    write_json(out/"meta/episodes.json",episodes)
    write_json(out/"meta/info.json",{"format":"pi07_lossless_rollouts_v1","fps":20,"total_frames":offset,
        "total_episodes":len(episodes),"task_names":canonical.task_names,
        "episode_metadata_sha256":file_hash(out/"meta/episodes.json"),
        "state_dimension":8,"action_dimension":7,"images":[256,256,3],
        "actions":"actual executed, clipped end-effector commands; includes declared collection perturbations",
        "time":"one observation before each action at the simulator's 20 Hz control rate",
        "terminal":"post-action terminal/reset images excluded",
        "source_manifest":str(run/"manifest.json"),"source_manifest_sha256":file_hash(run/"manifest.json"),
        "source_sha256":{name:file_hash(ROOT/name) for name in ("prepare_rollout_dataset.py","subtask_controller.py")}})
    (out/"annotations_template.jsonl").write_text("".join(json.dumps({"episode_index":e["episode_index"]})+"\n" for e in episodes))
    shutil.copy2(run/"manifest.json",out/"collection_manifest.json")
    print(json.dumps({"episodes":len(episodes),"frames":offset,"output":str(out),"quality_or_mistake_labels_assigned":0}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
