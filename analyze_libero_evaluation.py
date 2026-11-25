"""Audit saved evaluation trajectories and report paired development outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image


def main(args):
    root=args.run.resolve();report=json.loads((root/"report.json").read_text())
    manifest=json.loads((root/"manifest.json").read_text())
    if report["status"]!="complete":raise ValueError("Analyze a completed run")
    conditions=[c["name"] for c in manifest["spec"]["conditions"]]
    jobs=manifest["jobs"];by_key={};resets={};rows=[];generations=copies=reset_comparisons=0;max_input_difference=0.
    for row in report["episodes"]:
        if "error" in row:raise ValueError("Evaluation contains errors")
        key=(row["task_id"],row["initial_state_id"])
        if (row["condition"],key) in by_key:raise ValueError("Duplicate episode")
        by_key[row["condition"],key]=row
        name=f"{row['condition']}_task{key[0]}_init{key[1]}"
        with np.load(root/f"{name}_trajectory.npz") as trajectory:
            actions=trajectory["actions"];states=trajectory["states"]
            progress=trajectory["predicates"];initial=trajectory["initial_sim_state"]
            assert actions.shape==(row["steps"],7) and states.shape==(row["steps"],8)
            assert np.isfinite(actions).all() and np.isfinite(states).all()
            assert len(progress)==row["steps"]+1
            reset_comparisons+=int(key in resets)
            prior=resets.setdefault(key,initial.copy())
            assert np.max(np.abs(prior-initial))<=1e-8
            for generation in row["generations"]:
                step=generation["step"]
                with np.load(root/f"{name}_step{step:04d}_current.npz") as current:
                    error=float(np.max(np.abs(current["state"]-states[step])))
                    max_input_difference=max(max_input_difference,error)
                    assert error==0
                    assert current["main"].shape==current["wrist"].shape==(256,256,3)
                with Image.open(root/f"{name}_step{step:04d}_goal.png") as image:
                    assert image.size==(448,224) and image.mode=="RGB"
                generations+=1
            for update in row.get("goal_updates",[]):
                step=update["step"]
                assert update["source"]=="copied_current"
                with np.load(root/f"{name}_step{step:04d}_current.npz") as current:
                    assert np.array_equal(current["state"],states[step])
                    expected=np.concatenate([np.asarray(Image.fromarray(current[k]).resize((224,224),Image.Resampling.BILINEAR))
                                             for k in ("main","wrist")],axis=1)
                assert np.array_equal(np.asarray(Image.open(root/f"{name}_step{step:04d}_goal.png")),expected)
                copies+=1
            rows.append({"condition":row["condition"],"task_id":key[0],"initial_state_id":key[1],
                "task":row["task"],"success":row["success"],"steps":row["steps"],
                "initial_progress":float(progress[0].mean()),
                "max_progress":float(progress.mean(axis=1).max()),
                "max_progress_gain":float(progress.mean(axis=1).max()-progress[0].mean())})
    assert len(rows)==len(conditions)*len(jobs)
    paired=[];rng=np.random.default_rng(20261104)
    tasks=sorted({j["task_id"] for j in jobs});ids=manifest["spec"]["init_ids"]
    baseline="A" if "A" in conditions else conditions[0]
    comparisons=[(baseline,condition) for condition in conditions if condition!=baseline]
    for generated in manifest["spec"]["conditions"]:
        if generated["goals"]!="generated":continue
        for copied in manifest["spec"]["conditions"]:
            if copied["goals"]=="copy" and all(generated.get(k)==copied.get(k)
                for k in ("checkpoint","layout","subtasks","metadata","metadata_cfg_beta")):
                comparisons.append((copied["name"],generated["name"]))
    for pair in manifest["spec"].get("paired_comparisons",[]):
        if len(pair)!=2 or not set(pair)<=set(conditions):raise ValueError("Unknown declared comparison")
        comparisons.append(tuple(pair))
    comparisons=list(dict.fromkeys(comparisons))
    for baseline,condition in comparisons:
        differences=[];wins=losses=0
        for task in tasks:
            per=[]
            for init in ids:
                a=by_key[baseline,(task,init)]["success"]
                b=by_key[condition,(task,init)]["success"]
                wins+=int(b and not a);losses+=int(a and not b);per.append(int(b)-int(a))
            differences.append(np.mean(per))
        draws=rng.choice(differences,(10000,len(tasks)),replace=True).mean(axis=1)
        paired.append({"condition":condition,"reference":baseline,"success_rate_difference":float(np.mean(differences)),
            "task_bootstrap_95_interval":np.quantile(draws,[.025,.975]).tolist(),"paired_wins":wins,"paired_losses":losses})
    summary={}
    for condition in conditions:
        selected=[r for r in rows if r["condition"]==condition]
        summary[condition]={"successes":sum(r["success"] for r in selected),"episodes":len(selected),
            "mean_max_progress":float(np.mean([r["max_progress"] for r in selected])),
            "mean_max_progress_gain":float(np.mean([r["max_progress_gain"] for r in selected]))}
    final=manifest["spec"].get("phase")=="final"
    analysis={"status":"passed","phase":"final" if final else manifest["spec"].get("phase","development"),
        "summary":summary,"paired":paired,"episodes":rows,
        "generated_pairs_audited":generations,"generation_state_max_difference":max_input_difference,
        "copied_pairs_audited":copies,"cross_condition_reset_comparisons":reset_comparisons,
        "reset_state_max_difference_limit":1e-8,
        "limitations":[("Reserved final starts after protocol freeze; pretraining and physical demo independence remain unverified." if final else
                         "Development starts only; no final benchmark or unseen-pretraining claim."),
            "One training seed; task bootstrap describes these task/start outcomes, not training variability. A zero-width interval from identical observed outcomes does not prove population equivalence.",
            "Predicate progress includes already-satisfied conditions; gain subtracts initial progress.",
            "Input audit verifies recorded states and shapes, not semantic correctness of generated goals."]}
    (root/"analysis.json").write_text(json.dumps(analysis,indent=2)+"\n")
    lines=[f"# {manifest['spec']['suite']} paired simulator evaluation", "", "Completed 2026-11-04.","",
        f"Same task/start pairs, seeds, 50-action predictions with eight executed, {manifest['spec']['max_steps']}-step limits.",
        "Generated goals use one ImageWAM call from current main/wrist images and state every 16 simulator actions.",
        "Simulation pauses during inference. No demonstration future is supplied online.","",
        "| Condition | Successes | Mean maximum predicate fraction | Mean gain over initial state |",
        "| --- | --- | --- | --- |"]
    for name,r in summary.items():
        lines.append(f"| {name} | {r['successes']}/{r['episodes']} | {r['mean_max_progress']:.3f} | {r['mean_max_progress_gain']:.3f} |")
    lines += ["",f"Verified {len(rows)} episodes and finite actions; {reset_comparisons} cross-condition full simulator reset comparisons. "
        f"Audited {generations} generated pairs and {copies} copied-current pairs; maximum generator/trajectory state difference {max_input_difference}.","",
        "See [analysis.json](analysis.json) for per-task outcomes, paired wins/losses and task-bootstrap intervals; "
        "[manifest.json](manifest.json) for checkpoint and source hashes; [report.json](report.json) for timing and memory.","",
        f"These are {'frozen final' if final else 'development'} measurements from one training seed. They do not establish the paper's overall efficacy. "
        "The state/shape audit does not judge goal-image semantics. Initial predicate satisfaction is reported separately."]
    if manifest["spec"]["max_steps"]<100:
        lines += ["", "**Short runtime smoke only. These truncated episodes do not measure task performance.**"]
    if (root/"timing_context.json").exists():
        lines += ["", "GPU diagnostics overlapped part of this evaluation; see [timing context](timing_context.json). "
                  "Overlapping wall/inference times are not an isolated throughput comparison."]
    (root/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"summary":summary,"paired":paired,"generated_pairs_audited":generations}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    main(p.parse_args())
