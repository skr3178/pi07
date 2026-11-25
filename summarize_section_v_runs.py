"""Verify matched training inputs and summarize measured prompt-profile results."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(args):
    records=[];reference=None;schedules=None
    for path in args.runs:
        path=path.resolve();m=json.loads((path/"manifest.json").read_text());r=json.loads((path/"report.json").read_text())
        saved={name:dict(np.load(path/f"{name}_schedule.npz")) for name in ("train","validation")}
        if reference is None:reference=m;schedules=saved
        else:
            for key in ("base_checkpoint_sha256","steps","batch_size","seed","optimizer","layout","dataset_info_sha256","token_limit","flow_seed_rule"):
                if m[key]!=reference[key]:raise ValueError(f"Unmatched {key}")
            for key in ("annotations_sha256","dataset_action_transform"):
                if m.get(key)!=reference.get(key):raise ValueError(f"Unmatched {key}")
            for name in saved:
                for key in ("current","future","segment_endpoint"):
                    if not np.array_equal(saved[name][key],schedules[name][key]):raise ValueError(f"Different {name} {key}")
        values=r.get("validation",[])
        record={"run":str(path),"profile":m["profile"],"status":r["status"],"updates":r["optimizer_updates"],
            "goal_present_training_examples":m["goal_present_count"],"prompt_options":m["prompt_options"],
            "first_8_curve":[{"step":v["step"],"mse":v["sampled_actions"]["first_8"]["mse"]} for v in values]}
        if r["status"]=="complete":
            assert r["checkpoint_roundtrip_exact"] and r["frozen_backbone_unchanged"]
            record["final_first_8_mse"]=values[-1]["sampled_actions"]["first_8"]["mse"]
            if "final_goals_dropped" in r:record["goals_off_first_8_mse"]=r["final_goals_dropped"]["sampled_actions"]["first_8"]["mse"]
        diagnostic=path/"offline_actions/report.json"
        if diagnostic.exists():
            d=json.loads(diagnostic.read_text())
            if d["status"]=="complete":
                record["offline_conditions"]={k:v["mse_8"] for k,v in d["conditions"].items()}
                if "copy_current" in d["conditions"]:
                    a=d["conditions"]["copy_current"]["samples"];b=d["conditions"]["saved_real"]["samples"]
                    task_differences={}
                    for x,y in zip(a,b):
                        assert (x["index"],x["episode"])==(y["index"],y["episode"])
                        task_differences.setdefault(x["task_id"],[]).append(y["sum_8"]/y["count_8"]-x["sum_8"]/x["count_8"])
                    means=[np.mean(v) for v in task_differences.values()]
                    draws=np.random.default_rng(20261104).choice(means,(10000,len(means)),replace=True).mean(axis=1)
                    record["real_minus_copy"]={"equal_task_mean_difference":float(np.mean(means)),
                        "task_bootstrap_95_interval":np.quantile(draws,[.025,.975]).tolist(),
                        "positive_means_real_goal_has_larger_error":True}
        records.append(record)
    args.output.mkdir(parents=True,exist_ok=True)
    varying_text_dropout=len({r["prompt_options"]["subtask_drop_given_goal"] for r in records})>1
    interpretation=("Conditional subtask dropout differs across profiles, so full vs combined does not isolate metadata."
                    if varying_text_dropout else "Conditional subtask dropout settings match across these profiles.")
    report={"status":"complete" if all(r["status"]=="complete" for r in records) else "partial",
        "dataset":reference["dataset"],
        "matched_base_optimizer_budget_current_and_future_indices":True,"runs":records,
        "caveats":["One training seed; fixed held-out demo observations, not simulator success.",
            "Real future goals are offline-only; no online access to demonstration futures.",
            interpretation,
            "Goal usefulness is not guaranteed; data task order, annotation provenance and pretraining overlap limit the claim."]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Section V matched training measurements","",f"Dataset: {reference['dataset']}. Same starting weights, observation/future schedule, optimizer, {reference['steps']}-update budget and 128-token interface verified.","",
        "| Profile | Updates/status | Final first-eight MSE | Goals off MSE |","| --- | --- | --- | --- |"]
    for r in records:
        final=r.get("final_first_8_mse");off=r.get("goals_off_first_8_mse")
        lines.append(f"| {r['profile']} | {r['updates']} / {r['status']} | {final:.4f} | {off:.4f} |" if final is not None and off is not None else
                     f"| {r['profile']} | {r['updates']} / {r['status']} | {f'{final:.4f}' if final is not None else 'pending'} | — |")
    lines += ["","Normalized action error, lower is better. These values do not measure live task success.","",
              interpretation,"",
              "[report.json](report.json) contains learning curves, image-content controls and paired task-bootstrap intervals where available."]
    (args.output/"README.md").write_text("\n".join(lines)+"\n");print(json.dumps({r["profile"]:r.get("final_first_8_mse",r["status"]) for r in records}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--runs",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
