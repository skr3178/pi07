"""Freeze evaluated inference conditions before using reserved simulator starts."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
RUNTIME=("evaluate_libero.py","run_goal_development.py","imagewam_goal_worker.py",
         "goal_policy.py","attention_check.py","libero_runtime.py","section_v_prompt.py",
         "subtask_controller.py","metadata_guidance.py","train_goal_pilot.py","freeze_protocol.py")
CONDITION_FIELDS=("checkpoint","layout","goals","subtasks","metadata","metadata_cfg_beta")
MODEL_FILES=("model.safetensors","config.json","policy_preprocessor.json","policy_postprocessor.json")


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):h.update(block)
    return h.hexdigest()


def spec_digest(spec):return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def validate_reserved_spec(spec):
    if spec.get("phase")!="final" or spec.get("suite") not in ("libero_10","libero_goal"):
        raise ValueError("Final protocol needs an explicit supported final suite")
    ids=spec["init_ids"];tasks=spec["task_ids"]
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(range(2,12)):
        raise ValueError("Final starts must be unique IDs within 2–11")
    if not tasks or len(tasks)!=len(set(tasks)) or not set(tasks)<=set(range(10)):
        raise ValueError("Invalid final task IDs")
    if spec.get("record_training_data") or any(c.get("action_noise_std",0) for c in spec["conditions"]):
        raise ValueError("Final evaluation cannot collect training data or perturb actions")


def condition_identity(condition):
    value={k:condition.get(k) for k in CONDITION_FIELDS}
    value["checkpoint"]=str(Path(value["checkpoint"]).resolve())
    value["subtasks"]=value["subtasks"] or "off"
    value["metadata"]=value["metadata"] or {}
    return value


def inventory(spec):
    paths={ROOT/name for name in RUNTIME}
    # Include local inference dependencies, including editable vendor sources.
    for tree in (ROOT/"vendor/lerobot/src/lerobot/policies/smolvla",
                 ROOT/"vendor/lerobot/src/lerobot/processor",
                 ROOT/"vendor/lerobot/src/lerobot/envs",
                 ROOT/"vendor/lerobot-libero/libero"):
        paths.update(tree.rglob("*.py"))
    config=ROOT/"artifacts/lerobot_libero_setup/libero_config/config.yaml"
    paths.add(config)
    env_config=json.loads(config.read_text())  # local config is JSON-compatible YAML
    for key in ("bddl_files","init_states"):
        directory=Path(env_config[key])/spec["suite"]
        files=[p for p in directory.rglob("*") if p.is_file()]
        if not files:raise ValueError(f"No simulator {key} files found for final suite")
        paths.update(files)
    paths.update(p for p in Path(env_config["assets"]).rglob("*") if p.is_file())
    for condition in spec["conditions"]:
        checkpoint=Path(condition["checkpoint"]).resolve()
        for name in MODEL_FILES:
            path=checkpoint/name
            if not path.is_file():raise FileNotFoundError(path)
            paths.add(path)
        paths.update(checkpoint.glob("*.json"));paths.update(checkpoint.glob("*.safetensors"))
    if any(c["goals"]=="generated" for c in spec["conditions"]):
        model=Path("/media/skr/storage/models/imagewam/imagewam_release/libero/flux2_klein_4b")
        paths.update(model/name for name in ("model.pt","config.yaml","dataset_stats.json"))
        paths.update((ROOT.parent/"ImageWAM/src/imagewam").rglob("*.py"))
        paths.update((ROOT.parent/"ImageWAM/third_party/flux2/src").rglob("*.py"))
    return {str(p.resolve()):digest(p) for p in sorted(paths)}


def validate_frozen_protocol(spec,path):
    validate_reserved_spec(spec)
    saved=json.loads(Path(path).read_text())
    if saved.get("format")!="pi07_frozen_final_v1" or saved["spec_sha256"]!=spec_digest(spec):
        raise ValueError("Final settings differ from the frozen protocol")
    for file,wanted in saved["files_sha256"].items():
        if digest(file)!=wanted:raise ValueError(f"Frozen file changed: {file}")
    return saved


def main(args):
    spec=json.loads(args.spec.read_text());validate_reserved_spec(spec)
    target=args.output.resolve()
    if target.exists():raise ValueError("Choose a new immutable protocol directory")
    if Path(spec["frozen_protocol"]).resolve()!=target/"protocol.json":
        raise ValueError("Spec must point to this protocol.json before freezing")
    seen=[];evidence={}
    for run in args.development:
        run=run.resolve();m=json.loads((run/"manifest.json").read_text())
        r=json.loads((run/"report.json").read_text());a=json.loads((run/"analysis.json").read_text())
        if r["status"]!="complete" or a["status"]!="passed":raise ValueError("Development run is incomplete or unaudited")
        if m["spec"]["suite"]!=spec["suite"]:continue
        if any(m["spec"].get(k,16)!=spec.get(k,16) for k in ("max_steps","goal_refresh_steps")):
            continue
        seen.extend(condition_identity(c) for c in m["spec"]["conditions"])
        evidence.update({str(run/name):digest(run/name) for name in ("manifest.json","report.json","analysis.json")})
    for c in spec["conditions"]:
        if condition_identity(c) not in seen:raise ValueError(f"Final condition has no matching audited development run: {c['name']}")
    files=inventory(spec);files.update(evidence)
    record={"format":"pi07_frozen_final_v1","frozen_utc":datetime.now(timezone.utc).isoformat(),
        "spec":spec,"spec_sha256":spec_digest(spec),"files_sha256":files,
        "development_evidence":list(evidence),
        "limitations":["One training seed; final starts are disjoint from development/collection IDs, not certified independent of pretrained or demo data.",
                       "This freeze protects exact inputs and settings, not model efficacy or annotation correctness."]}
    target.mkdir(parents=True)
    (target/"protocol.json").write_text(json.dumps(record,indent=2)+"\n")
    (target/"spec.json").write_text(json.dumps(spec,indent=2)+"\n")
    print(json.dumps({"protocol":str(target/"protocol.json"),"episodes_per_condition":len(spec["task_ids"])*len(spec["init_ids"]),"files":len(files)}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--spec",type=Path,required=True)
    p.add_argument("--development",type=Path,nargs="+",required=True);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
