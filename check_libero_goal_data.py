"""Verify the local Goal adapter against raw actions/state and episode bounds."""
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq

from train_goal_pilot import ROOT,torch,np,default_collate,load_smolvla_config,write_json
from libero_data import make_index,make_dataset
from section_v_data import SectionVSamples
from section_v_prompt import PromptOptions,make_section_v_processors
from goal_policy import configure_goals
from PIL import Image,ImageDraw


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    index=make_index("libero_goal");dataset=make_dataset(index)
    raw=pq.read_table(index.root/"data",columns=["index","action","observation.state"]).to_pandas().sort_values("index")
    actions=np.stack(raw["action"]);states=np.stack(raw["observation.state"])
    assert set(np.unique(actions[:,-1]))=={0.,1.}
    canonical=actions.copy();canonical[:,-1]=1-2*actions[:,-1]
    delta=np.diff(states[:,6]-states[:,7])
    same=index.episodes[1:]==index.episodes[:-1]
    motion={}
    for command in (-1,1):
        mask=same&(canonical[:-1,-1]==command)&(np.abs(delta)>.0001)
        mean=float(delta[mask].mean())
        assert command*mean<0, "Canonical gripper command disagrees with finger motion"
        motion[str(command)]={"transitions":int(mask.sum()),"mean_width_change":mean,
                              "closing_fraction":float((delta[mask]<0).mean())}
    current=[]
    for task in range(10):
        episode=int(index.episodes[np.flatnonzero(index.tasks==task)[0]])
        start,end=index.bounds[episode]
        current.extend([start,(start+end)//2,end-1])
    current=np.asarray(current);schedule={"current":current,"future":index.end[current],"goal_present":np.ones(len(current),bool)}
    samples=SectionVSamples(dataset,index,schedule,PromptOptions.profile("full"),training=False)
    cfg,_=load_smolvla_config("/media/skr/storage/models/smolvla_libero_90_pi07",device="cpu")
    configure_goals(cfg);pre,_=make_section_v_processors(cfg,"/media/skr/storage/models/smolvla_libero_90_pi07")
    max_normalized_state=max_normalized_action=0.;padding_count=0;rows=[]
    for i,c in enumerate(current):
        sample=samples[i];end=int(index.end[c]);wanted=np.minimum(c+np.arange(50),end)
        assert np.array_equal(sample["observation.state"].numpy(),states[c])
        assert np.array_equal(sample["action"].numpy(),canonical[wanted])
        assert np.array_equal(sample["action_is_pad"].numpy(),c+np.arange(50)>end)
        padding_count+=int(sample["action_is_pad"].sum())
        for camera in ("image","image2"):
            assert sample[f"observation.images.{camera}"].shape==(3,256,256)
        assert index.context(int(c)).subtask.value is None
        assert "Subtask:" not in sample["task"]
        batch=pre(default_collate([sample]))
        assert torch.isfinite(batch["observation.state"]).all() and torch.isfinite(batch["action"]).all()
        max_normalized_state=max(max_normalized_state,float(batch["observation.state"].abs().max()))
        max_normalized_action=max(max_normalized_action,float(batch["action"].abs().max()))
        if i%3==0:
            images=[(sample[f"observation.images.{camera}"].permute(1,2,0)*255).round().byte().numpy() for camera in ("image","image2")]
            row=Image.new("RGB",(512,282),"white");ImageDraw.Draw(row).text((4,4),index.context(int(c)).task,fill="black")
            row.paste(Image.fromarray(np.concatenate(images,axis=1)),(0,26));rows.append(row)
    sheet=Image.new("RGB",(1024,282*5),"white")
    for i,row in enumerate(rows):sheet.paste(row,(512*(i%2),282*(i//2)))
    sheet.save(args.output/"starting_views.png")
    report={"status":"passed","episodes":len(index.bounds),"frames":len(index.episodes),"tasks":10,
        "decoded_examples":len(current),"raw_arm_actions_state_exact":True,"padding_masks_exact":True,
        "gripper_conversion":"simulator_action = 1 - 2 * raw_FastWAM_action",
        "canonical_gripper_motion_check":motion,
        "padded_action_steps_checked":padding_count,"current_cameras_shape":[3,256,256],
        "maximum_abs_normalized_state":max_normalized_state,"maximum_abs_normalized_action":max_normalized_action,
        "subtask_annotations":"absent; not inferred from task names",
        "limits":"Checks numerical schema, sampled alignment and aggregate gripper motion, not full simulator replay or success."}
    write_json(args.output/"report.json",report);print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
