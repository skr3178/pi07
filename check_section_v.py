"""Quantitative Section V prompt/data/tokenizer gates, with optional CPU model smoke."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME","/media/skr/storage/hf_home")
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("HF_DATASETS_CACHE",str(ROOT / "artifacts/lerobot_libero_setup/datasets_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

import numpy as np
import torch
from torch.utils.data import default_collate
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from goal_policy import GOALS, GoalSmolVLAPolicy, configure_goals
from libero_runtime import load_smolvla_config
from section_v_data import SectionVIndex, SectionVSamples
from section_v_prompt import Evidence, PromptContext, PromptOptions, build_prompt, runtime_prompt, make_section_v_processors, speed_bin


def reject(function):
    try: function()
    except (ValueError,TypeError): return 1
    raise AssertionError("Invalid input was accepted")


def main(args,report):
    torch.set_num_threads(4); torch.manual_seed(20261030)
    options=PromptOptions()
    synthetic=PromptContext("move the cup",Evidence("grasp the handle","synthetic_test_fixture"),
        Evidence(300,"synthetic_test_fixture"),Evidence(5,"synthetic_test_fixture"),Evidence(False,"synthetic_test_fixture"))
    prompts=[build_prompt(synthetic,options,training=True,seed=31,sample_key=i) for i in range(10000)]
    goals=sum(p.goal_present for p in prompts)
    dropped_subtasks=sum(p.goal_present and p.fields["subtask"]["status"]=="dropped" for p in prompts)
    all_meta=sum(p.decisions["all_metadata_drop"] for p in prompts)
    measurements={"goal_present":{"count":goals,"n":10000,"expected_probability":.25},
        "subtask_dropped_given_goal":{"count":dropped_subtasks,"n":goals,"expected_probability":.30},
        "all_metadata_dropped":{"count":all_meta,"n":10000,"expected_probability":.15}}
    for field in ("speed_steps","quality","mistake"):
        measurements[field+"_present"]={"count":sum(p.fields[field]["status"]=="present" for p in prompts),
                                        "n":10000,"expected_probability":.8075}
    for value in measurements.values():
        n,p=value["n"],value["expected_probability"]
        value["rate"]=value["count"]/n
        value["standard_deviations"]=abs(value["count"]-n*p)/math.sqrt(n*p*(1-p))
        assert value["standard_deviations"]<=3,value
    assert all("Control Mode: ee." in p.text for p in prompts)
    assert all(p.fields["subtask"]["status"]=="present" for p in prompts if not p.goal_present)
    assert prompts[17]==build_prompt(synthetic,options,training=True,seed=31,sample_key=17)
    for profile in ("task","subtask","goal","subtask_goal","full"):
        other=build_prompt(synthetic,PromptOptions.profile(profile),training=True,seed=31,sample_key=17)
        if profile in ("goal","subtask_goal","full"):
            assert other.goal_present==prompts[17].goal_present
    assert speed_bin(1750,500)==2000 and speed_bin(2249,500)==2000
    invalid=sum((reject(lambda:PromptContext("x",quality=Evidence(6,"test"))),
        reject(lambda:PromptContext("x",mistake=Evidence("false","test"))),
        reject(lambda:PromptContext("x",control_mode="joint")),
        reject(lambda:Evidence(5)), reject(lambda:Evidence(5,"test",1.1)),
        reject(lambda:runtime_prompt("task",subtask="do something"))))
    runtime=runtime_prompt("move the cup",desired_quality=5,desired_mistake=False)
    assert not runtime.goal_present and "Quality: 5." in runtime.text and "Mistake: false." in runtime.text
    assert "Subtask:" not in runtime.text
    report["synthetic_dropout_audit"]={"samples":10000,"measurements":measurements,
        "control_mode_present":10000,"invalid_requests_rejected":invalid,
        "note":"Synthetic labels test code only; they are not attached to training data."}

    index=SectionVIndex(ROOT / "data/libero_10_subtask")
    train=index.split()["train_episodes"]
    schedule=index.schedule(train,10000,seed=42)
    # Construct audit without decoding; dataset is only accessed in __getitem__.
    sampled=SectionVSamples(None,index,schedule,options,training=True)
    assert all(p.fields["quality"]["status"]==p.fields["mistake"]["status"]=="unavailable" for p in sampled.prompts)
    assert all("Quality:" not in p.text and "Mistake:" not in p.text for p in sampled.prompts)
    for ep in index.repaired_artifact_episodes:
        start,_=index.bounds[ep]
        assert index.subtasks[start]==index.subtasks[start+1]!=15
    assert len(index.uncertain_subtask_episodes)==3
    for ep in index.uncertain_subtask_episodes:
        start,end=index.bounds[ep]
        for current in (start, end-1):
            context=index.context(current)
            assert context.subtask.value is None
            assert "Subtask:" not in build_prompt(context,options,training=False).text
    crossing={k:v.copy() for k,v in schedule.items()}
    crossing["current"][0]=index.bounds[0][1]-1;crossing["future"][0]=index.bounds[1][0]
    reject(lambda:SectionVSamples(None,index,crossing,options,training=True))
    # Verify provenance ingestion and interval boundaries with a clearly synthetic fixture.
    with tempfile.TemporaryDirectory(dir=ROOT / "artifacts") as tmp:
        path=Path(tmp)/"annotations.jsonl"
        row={"episode_index":0,"quality":{"value":3,"source":"synthetic_test_fixture"},
             "mistakes":[{"start_frame":2,"end_frame_exclusive":5,
                          "label":{"value":True,"source":"synthetic_test_fixture"}}]}
        path.write_text(json.dumps(row)+"\n");index.load_annotations(path)
        start=index.bounds[0][0]
        assert index.context(start+1).mistake.value is None
        assert index.context(start+2).mistake.value is True
        assert index.context(start+5).mistake.value is None
        row["mistakes"].append(row["mistakes"][0])
        path.write_text(json.dumps(row)+"\n")
        reject(lambda:index.load_annotations(path))
        index.annotations={}
    report["data"]={"examples_audited":10000,"cross_episode_goals":0,"unknown_quality":10000,
        "uncertain_subtask_episodes":index.uncertain_subtask_episodes,
        "unknown_subtask_samples":sum(p.fields["subtask"]["value"] is None for p in sampled.prompts),
        "unknown_mistake":10000,"subtask_ids":sorted(int(v) for v in np.unique(index.subtasks)),
        "repaired_one_frame_artifacts":len(index.repaired_artifact_episodes),
        "speed_bin_steps":options.speed_bin_steps,"speed_units":"retained dataset frames; physical timing unverified"}
    sampled.write_audit(args.output/"sample_audit.jsonl")

    cfg,_=load_smolvla_config("/media/skr/storage/models/smolvla_libero",device="cpu")
    configure_goals(cfg)
    pre,post=make_section_v_processors(cfg,"/media/skr/storage/models/smolvla_libero")
    # Check every task/subtask pairing, including the full metadata schema.
    examples=[]
    for task in range(10):
        for sid in np.unique(index.subtasks[index.tasks==task]):
            i=int(np.flatnonzero((index.tasks==task)&(index.subtasks==sid))[0])
            c=index.context(i)
            c=PromptContext(c.task,c.subtask,c.speed_steps,synthetic.quality,synthetic.mistake)
            examples.append(build_prompt(c,options,training=False).text)
    counts=pre.token_counts(examples)
    assert max(counts)<=128
    reject(lambda:pre({"task":"long instruction "*300}))
    dataset=LeRobotDataset("lerobot/libero_10_subtask",root=index.root,video_backend="torchcodec",
                          delta_timestamps={"action":[i/index.info["fps"] for i in range(50)]})
    actual=SectionVSamples(dataset,index,schedule,options,training=True)
    chosen=[int(np.flatnonzero(index.tasks[schedule["current"]]==task)[0]) for task in range(10)]
    for i in chosen:
        item=actual[i]
        for key in GOALS:
            assert bool(item[key+"_padding_mask"])==actual.prompts[i].goal_present
        processed=pre(default_collate([item]))
        assert processed["observation.language.tokens"].shape==(1,128)
        assert torch.isfinite(processed["action"]).all()
    # True goal-present full prompt for the model smoke; no invented data labels.
    smoke_schedule={k:v[:1].copy() for k,v in schedule.items()}
    smoke_data=SectionVSamples(dataset,index,smoke_schedule,options,training=False)
    batch=pre(default_collate([smoke_data[0]]))
    pre.save_pretrained(args.output/"processors");post.save_pretrained(args.output/"processors")
    reload_pre,_=make_section_v_processors(cfg,args.output/"processors")
    reloaded=reload_pre(default_collate([smoke_data[0]]))
    assert torch.equal(batch["observation.language.tokens"],reloaded["observation.language.tokens"])
    report["tokenizer"]={"example_prompts":len(examples),"max_tokens":max(counts),
        "prompts_over_old_48_token_limit":sum(n>48 for n in counts),"configured_limit":128,
        "silent_truncation":False,"decoded_and_preprocessed_examples":len(chosen),"saved_processor_roundtrip":True}
    (args.output/"prompt_examples.json").write_text(json.dumps({"synthetic_full_fields":examples,
        "actual_data_examples":[actual.audit(i) for i in chosen],"runtime_request":asdict(runtime)},indent=2)+"\n")
    if args.model_smoke:
        start=time.perf_counter()
        policy=GoalSmolVLAPolicy.from_pretrained("/media/skr/storage/models/smolvla_libero",config=cfg,strict=True)
        policy.train()
        noise=torch.randn(1,50,32)
        loss,details=policy(batch,noise=noise,time=torch.tensor([.5]))
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in policy.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        report["model_smoke"]={"device":"cpu","loss":float(loss.detach()),
            "finite_gradient_tensors":len(grads),"seconds":time.perf_counter()-start,
            "optimizer_updates":0,"note":"Forward/backward wiring check; no performance claim."}
    report["status"]="passed"


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=ROOT/"artifacts/section_v_implementation")
    p.add_argument("--model-smoke",action="store_true")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"report.json").exists():raise RuntimeError("Preserve existing check; choose a new output directory")
    report={"status":"running"}
    try:main(args,report)
    except Exception:
        report["status"]="failed";report["error"]=traceback.format_exc();raise
    finally:
        (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report),flush=True)
