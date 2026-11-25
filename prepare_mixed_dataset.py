"""Declare a small, reviewable metadata study without copying demonstration images."""
import argparse
from pathlib import Path
import json
from train_goal_pilot import ROOT,file_hash,write_json
from section_v_data import SectionVIndex
from rollout_data import RolloutIndex


def main(args):
    if not 0<args.rollout_probability<1:raise ValueError("Use a genuine two-source mixture")
    demo=SectionVIndex(ROOT/"data/libero_10_subtask");rollout=RolloutIndex(args.rollouts)
    split=json.loads(args.demo_split.read_text());selected=[]
    # One training and one held-out demonstration per collected task. Their
    # quality labels still require review; this is not an automatic expert tag.
    for task in sorted(set(rollout.tasks)):
        for side in ("train_episodes","validation_episodes"):
            eligible=[ep for ep in split[side] if demo.tasks[demo.bounds[ep][0]]==task]
            selected.append(eligible[0])
    out=args.output.resolve();(out/"meta").mkdir(parents=True,exist_ok=False)
    info={"format":"pi07_demo_rollout_mixture_v1","fps":None,"components":
        [{"root":str(x.root.resolve()),"info_sha256":file_hash(x.root/"meta/info.json")} for x in (demo,rollout)],
        "demo_episode_subset":selected,"rollout_probability":args.rollout_probability,
        "demo_split":str(args.demo_split.resolve()),"demo_split_sha256":file_hash(args.demo_split),
        "demo_episode_offset":0,"rollout_episode_offset":len(demo.bounds),
        "action_interface":"same 7-D end-effector command representation; original normalizer retained",
        "time_caveat":"Demo retained frames have nominal fps=10, physical spacing unverified; collected actions are 20 Hz. Goal horizons are sampled separately. Speed is explicitly recorded steps, not shared physical seconds.",
        "source_sha256":{f:file_hash(ROOT/f) for f in ("prepare_mixed_dataset.py","mixed_data.py")}}
    write_json(out/"meta/info.json",info)
    print(json.dumps({"output":str(out),"demo_episodes":selected,"rollout_episodes":len(rollout.records),"quality_labels_assigned":0}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rollouts",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--demo-split",type=Path,default=ROOT/"artifacts/goal_training_1000/episode_split.json")
    p.add_argument("--rollout-probability",type=float,default=.5);main(p.parse_args())
