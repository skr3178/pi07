"""Check recorded action alignment, lossless views, padding and grouped splits."""
import argparse
from pathlib import Path
import numpy as np
from train_goal_pilot import write_json
from rollout_data import RolloutIndex,RolloutDataset


def main(args):
    index=RolloutIndex(args.dataset);dataset=RolloutDataset(index)
    checked=padded=0
    for row in index.records:
        with np.load(row["source_recording"]) as original:
            for frame in sorted({0,(len(original["action"])-1)//2,len(original["action"])-1}):
                sample=dataset[row["start"]+frame];n=len(original["action"]);ids=frame+np.arange(50)
                assert np.array_equal(sample["action"].numpy(),original["action"][np.minimum(ids,n-1)])
                assert np.array_equal(sample["action_is_pad"].numpy(),ids>=n)
                assert np.array_equal(sample["observation.state"].numpy(),original["state"][frame])
                for source,key in (("main","image"),("wrist","image2")):
                    actual=(sample[f"observation.images.{key}"].permute(1,2,0).numpy()*255).round().astype(np.uint8)
                    assert np.array_equal(actual,original[source][frame])
                context=index.context(row["start"]+frame)
                assert context.quality.value is None and context.mistake.value is None
                checked+=1;padded+=int((ids>=n).sum())
    grouped=None
    if all(len({r["initial_state_id"] for r in index.records if r["task_id"]==t})>=2 for t in set(index.tasks)):
        split=index.split();groups=[]
        for side in ("train_episodes","validation_episodes"):
            groups.append({(index.records[e]["task_id"],index.records[e]["initial_state_id"]) for e in split[side]})
        assert not groups[0]&groups[1];grouped=True
    report={"status":"passed","observations_checked":checked,"padded_action_positions_checked":padded,
        "cameras_state_executed_actions_exact":True,"unknown_metadata_preserved":True,
        "physical_start_groups_disjoint":grouped,"dataset":str(args.dataset.resolve())}
    args.output.parent.mkdir(parents=True,exist_ok=True);write_json(args.output,report);print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
