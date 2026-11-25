"""Check generated-goal routing and provenance on real decoded observations.

Colored images below are explicitly synthetic fixtures, never training data.
"""
import argparse
import copy
import json
import tempfile
from pathlib import Path

from train_goal_pilot import ROOT, np, torch, LeRobotDataset, file_hash, write_json
from section_v_data import SectionVIndex,SectionVSamples
from section_v_prompt import PromptOptions
from generated_goal_data import GeneratedGoalCache,MixedGoalSamples,observation_digest,generation_prompt
from PIL import Image


def rejects(fn):
    try:fn()
    except ValueError:return
    raise AssertionError("Expected rejection")


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    index=SectionVIndex(ROOT/"data/libero_10_subtask")
    schedule=index.schedule(index.split()["train_episodes"],10,20261104,goal_probability=1.)
    dataset=LeRobotDataset("lerobot/libero_10_subtask",root=index.root,video_backend="torchcodec",
        delta_timestamps={"action":[i/index.info["fps"] for i in range(50)]})
    options=PromptOptions.profile("full",goal_probability=1.)
    samples=SectionVSamples(dataset,index,schedule,options,training=True)
    raw=samples[0];current=int(schedule["current"][0])
    with tempfile.TemporaryDirectory(dir=args.output) as tmp:
        root=Path(tmp);path=root/"manifest.json";image=root/"fixture.png"
        rgb=np.zeros((224,448,3),dtype=np.uint8);rgb[:,:224,0]=255;rgb[:,224:,2]=255
        Image.fromarray(rgb).save(image)
        entry={"current_index":current,"episode_index":int(index.episodes[current]),
            "frame_index":int(index.frames[current]),"prompt":generation_prompt(index,current,"subtask"),
            "image":image.name,"image_sha256":file_hash(image),"observation_sha256":observation_digest(raw),
            "review":"unreviewed"}
        manifest={"status":"complete","dataset_root":str(index.root.resolve()),
            "dataset_info_sha256":file_hash(index.root/"meta/info.json"),"prompt_mode":"subtask","entries":[entry]}
        write_json(path,manifest)
        assert not GeneratedGoalCache(path,index).entries
        cache=GeneratedGoalCache(path,index,allow_unreviewed=True)
        mixed=MixedGoalSamples(samples,cache,1.,31)
        replaced=mixed[0]
        assert torch.equal(raw["action"],replaced["action"])
        assert torch.equal(raw["observation.images.image"],replaced["observation.images.image"])
        assert replaced["observation.images.goal_camera1"][0].eq(1).all()
        assert replaced["observation.images.goal_camera1"][2].eq(0).all()
        assert replaced["observation.images.goal_camera2"][2].eq(1).all()
        assert mixed.audit(0)["goal_source"]=="generated"
        assert not MixedGoalSamples(samples,cache,0.,31).generated.any()
        off=SectionVSamples(dataset,index,schedule,PromptOptions.profile("task"),training=True)
        assert not MixedGoalSamples(off,cache,1.,31).generated.any()
        rejected=copy.deepcopy(manifest);rejected["entries"][0]["review"]="rejected";write_json(path,rejected)
        assert not GeneratedGoalCache(path,index,allow_unreviewed=True).entries
        duplicate=copy.deepcopy(manifest);duplicate["entries"].append(copy.deepcopy(entry));write_json(path,duplicate)
        rejects(lambda:GeneratedGoalCache(path,index))
        write_json(path,manifest)
        changed=raw.copy();changed["observation.state"]=raw["observation.state"]+1
        rejects(lambda:cache.images(current,changed))
        # Index, prompt, episode and corrupted bytes all reject independently.
        for key,value in (("episode_index",-1),("frame_index",-1),("prompt","wrong instruction"),("image_sha256","bad")):
            bad=copy.deepcopy(manifest);bad["entries"][0][key]=value;write_json(path,bad)
            rejects(lambda:GeneratedGoalCache(path,index,allow_unreviewed=True))
        report={"status":"passed","real_observations":1,"schedule_samples":10,"fixture_images_used_for_training":0,
            "unreviewed_excluded_by_default":True,"rejected_excluded_even_in_exploratory_mode":True,
            "duplicate_unreviewed_indices_rejected":True,"left_right_camera_routing":True,
            "actions_and_current_images_unchanged":True,"hidden_goals_never_replaced":True,
            "zero_mixture_selects_none":True,"bad_provenance_cases_rejected":5}
    write_json(args.output/"report.json",report);print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
