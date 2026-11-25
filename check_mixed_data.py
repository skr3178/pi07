"""Audit the curated mixed-quality study before fitting metadata policies."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from libero_data import make_index,make_dataset
from section_v_data import SectionVSamples
from section_v_prompt import PromptOptions
from train_goal_pilot import write_json


def main(args):
    torch.set_num_threads(4)
    index=make_index("mixed",args.dataset,args.annotations);dataset=make_dataset(index)
    split=json.loads((args.schedule/"episode_split.json").read_text())
    train=dict(np.load(args.schedule/"train_schedule.npz"))
    validation=dict(np.load(args.schedule/"validation_schedule.npz"))
    assert not set(split["train_episodes"])&set(split["validation_episodes"])
    for schedule,side in ((train,"train_episodes"),(validation,"validation_episodes")):
        assert np.isin(index.episodes[schedule["current"]],split[side]).all()
        assert np.array_equal(index.episodes[schedule["current"]],index.episodes[schedule["future"]])
    no_meta=SectionVSamples(dataset,index,train,PromptOptions.profile("subtask_goal"),training=True,seed=20261104)
    full=SectionVSamples(dataset,index,train,PromptOptions.profile("full"),training=True,seed=20261104)
    assert np.array_equal(no_meta.schedule["goal_present"],full.schedule["goal_present"])
    visible={k:{} for k in ("quality","mistake","speed_steps")}
    for a,b in zip(no_meta.prompts,full.prompts):
        assert a.fields["subtask"]==b.fields["subtask"]
        for field,counts in visible.items():
            value=b.fields[field]
            if value["status"]=="present":
                key=str(value.get("bin",value["value"]));counts[key]=counts.get(key,0)+1
    assert set(visible["mistake"])=={"True","False"}
    checked=padded=0
    for episode in split["train_episodes"]+split["validation_episodes"]:
        start,end=index.bounds[episode]
        for c in (start,(start+end)//2,end-1):
            row=dataset[c];which=int(c>=index.frame_offset)
            local=c-(index.frame_offset if which else 0)
            original=dataset.datasets[which][local]
            for key in ("action","action_is_pad","observation.state","observation.images.image","observation.images.image2"):
                assert torch.equal(row[key],original[key]),key
            assert row["action"].shape==(50,7) and row["observation.state"].shape==(8,)
            assert torch.isfinite(row["action"]).all() and torch.isfinite(row["observation.state"]).all()
            assert torch.equal(row["action_is_pad"],torch.tensor(c+np.arange(50)>=end))
            assert index.context(c).quality.value in range(1,6)
            checked+=1;padded+=int(row["action_is_pad"].sum())
    report={"status":"passed","observations_checked":checked,"padded_positions":padded,
        "paired_profiles":"same goals and subtask dropout; only metadata presence differs",
        "visible_training_metadata":visible,"training_episodes":split["train_episodes"],
        "validation_episodes":split["validation_episodes"],
        "rollout_fraction":float(np.mean(train["current"]>=index.frame_offset)),
        "limits":"Provisional assistant labels; sparse mistakes; demo physical starts/time spacing and pretraining overlap unverified."}
    write_json(args.output,report);print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("dataset","annotations","schedule","output"):p.add_argument("--"+name,type=Path,required=True)
    main(p.parse_args())
