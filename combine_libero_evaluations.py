"""Join completed paired conditions without copying or rewriting original runs.

The analyzer verifies every full simulator reset across the joined conditions.
Original manifests and source snapshots remain linked from the joined record.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    runs=[p.resolve() for p in args.runs]
    records=[(p,json.loads((p/"manifest.json").read_text()),json.loads((p/"report.json").read_text())) for p in runs]
    first=records[0][1];manifest=copy.deepcopy(first)
    manifest["spec"]["conditions"]=[];manifest["checkpoint_sha256"]={}
    manifest.pop("source_sha256",None)
    manifest["combined_sources"]=[];manifest["combiner_sha256"]=sha(Path(__file__))
    report={"status":"complete","episodes":[]};names=set()
    for root,source,result in records:
        if result["status"]!="complete":raise ValueError(f"Unfinished run: {root}")
        for field in ("jobs","execute_steps","settling_steps","control_hz"):
            if source[field]!=first[field]:raise ValueError(f"Unpaired {field}")
        for field in ("suite","task_ids","init_ids","max_steps","goal_refresh_steps"):
            if source["spec"].get(field)!=first["spec"].get(field):raise ValueError(f"Different {field}")
        for name,value in first["source_sha256"].items():
            if name!="evaluate_libero.py" and source["source_sha256"].get(name)!=value:
                raise ValueError(f"Model/input dependency changed: {name}")
        for condition in source["spec"]["conditions"]:
            name=condition["name"]
            if name in names:raise ValueError(f"Duplicate condition {name}")
            names.add(name);manifest["spec"]["conditions"].append(condition)
            manifest["checkpoint_sha256"][name]=source["checkpoint_sha256"][name]
        manifest["combined_sources"].append({"directory":str(root),"manifest_sha256":sha(root/"manifest.json"),
            "report_sha256":sha(root/"report.json"),"source_sha256":source["source_sha256"]})
        report["episodes"].extend(result["episodes"])
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for i,(root,_,result) in enumerate(records):
        (out/f"original_run_{i}").symlink_to(root,target_is_directory=True)
        for row in result["episodes"]:
            prefix=f"{row['condition']}_task{row['task_id']}_init{row['initial_state_id']}"
            for path in root.glob(prefix+"_*"):
                if path.is_file():(out/path.name).symlink_to(path)
    manifest["source_version_note"]="Evaluator orchestration versions differ; original and resume source snapshots remain in each linked run. Shared model/input dependency hashes match."
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    timing={str(root):json.loads((root/"timing_context.json").read_text())
            for root,_,_ in records if (root/"timing_context.json").exists()}
    if timing:(out/"timing_context.json").write_text(json.dumps(timing,indent=2)+"\n")
    print(json.dumps({"output":str(out),"conditions":sorted(names),"episodes":len(report["episodes"])}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--runs",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True);main(p.parse_args())
