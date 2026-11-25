"""Canonical policy inputs for the two local LIBERO datasets.

LIBERO-Goal has task instructions but no subtask annotations. Missing phases
stay missing. Its wrist camera is renamed and both 512px views are resized to
256px, matching the live policy interface. FastWAM gripper targets (0 close,
1 open) are converted to simulator commands (+1 close, -1 open). The six arm
commands and eight state values are untouched.
"""
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch

from section_v_data import SectionVIndex
from section_v_prompt import Evidence,PromptContext
from goal_data import GoalIndex

ROOT=Path(__file__).resolve().parent
GOAL_ROOT=Path("/media/skr/storage/datasets/LIBERO-fastwam/lerobot_v30/libero_goal_no_noops_lerobot")


class LiberoGoalIndex(SectionVIndex):
    def __init__(self,root,annotations=None):
        self.root=Path(root);self.info=json.loads((self.root/"meta/info.json").read_text())
        self.table=pq.read_table(self.root/"data",columns=["index","episode_index","frame_index","task_index"])
        data=self.table.to_pandas().sort_values("index")
        if not np.array_equal(data["index"],np.arange(len(data))):raise ValueError("Noncontiguous dataset index")
        self.episodes=data.episode_index.to_numpy();self.tasks=data.task_index.to_numpy()
        self.frames=data.frame_index.to_numpy();self.subtasks=np.full(len(data),-1,dtype=np.int64)
        self.end=np.empty(len(data),dtype=np.int64);self.bounds={}
        for episode,rows in data.groupby("episode_index",sort=True):
            indices=rows["index"].to_numpy()
            if not np.array_equal(indices,np.arange(indices[0],indices[-1]+1)):raise ValueError("Noncontiguous episode")
            self.bounds[int(episode)]=(int(indices[0]),int(indices[-1])+1)
            self.end[indices]=indices[-1]
        self.segment_end=self.end.copy()  # task endpoint; NOT an annotated intermediate boundary
        self.repaired_artifact_episodes=[];self.uncertain_subtask_episodes={}
        names=pq.read_table(self.root/"meta/tasks.parquet").to_pydict()
        self.task_names=dict(zip(names["task_index"],names["__index_level_0__"]))
        self.annotations={}
        if annotations is not None:self.load_annotations(annotations)

    def context(self,current):
        if not 0<=current<len(self.episodes):raise IndexError(current)
        episode=int(self.episodes[current]);task=int(self.tasks[current]);start,end=self.bounds[episode]
        row=self.annotations.get(episode,{});mistake=Evidence()
        for a,b,label in row.get("mistakes",[]):
            if a<=self.frames[current]<b:mistake=label;break
        return PromptContext(self.task_names[task],
            subtask=Evidence(None,"LIBERO-Goal release has no subtask annotations"),
            speed_steps=Evidence(end-start,"retained-frame count; no-op-filtered physical duration unverified",1.),
            quality=row.get("quality",Evidence()),mistake=mistake)

    def schedule(self,episodes,n,seed,goal_probability=.25):
        # Same mixture as the existing sampler, using this release's nominal fps.
        rng=np.random.default_rng(seed)
        allowed=np.flatnonzero(np.isin(self.episodes,episodes))
        current=rng.choice(allowed,n,replace=True);present=rng.random(n)<goal_probability
        endpoint=rng.random(n)<.25
        maximum=np.minimum(round(4*self.info["fps"]),self.end[current]-current)
        future=current+(rng.random(n)*(maximum+1)).astype(np.int64)
        future[endpoint]=self.segment_end[current[endpoint]]
        return {"current":current,"future":future,"goal_present":present,"segment_endpoint":endpoint}


class CanonicalGoalDataset:
    def __init__(self,dataset):self.dataset=dataset
    def __len__(self):return len(self.dataset)
    def __getitem__(self,i):
        sample=self.dataset[i].copy()
        if "action" in sample:
            action=sample["action"].clone()
            gripper=action[..., -1]
            if not torch.isfinite(action).all() or not ((gripper==0)|(gripper==1)).all():
                raise ValueError("Expected FastWAM binary gripper targets: 0 close, 1 open")
            action[..., -1]=1-2*gripper
            sample["action"]=action
        for source,target in (("observation.images.image","observation.images.image"),
                              ("observation.images.wrist_image","observation.images.image2")):
            image=sample[source]
            if image.shape[0]!=3 and image.shape[-1]==3:image=image.permute(2,0,1)
            if image.ndim!=3 or image.shape[0]!=3:raise ValueError("Expected RGB video")
            pixels=(image.detach().cpu().permute(1,2,0).clamp(0,1)*255).round().byte().numpy()
            pixels=np.asarray(Image.fromarray(pixels).resize((256,256),Image.Resampling.BILINEAR)).copy()
            sample[target]=torch.from_numpy(pixels).permute(2,0,1).float()/255
        sample.pop("observation.images.wrist_image")
        return sample


def make_index(kind,root=None,annotations=None):
    if kind=="mixed":
        from mixed_data import MixedIndex
        if root is None:raise ValueError("Mixed datasets require an explicit root")
        return MixedIndex(root,annotations)
    if kind=="libero_10":return SectionVIndex(root or ROOT/"data/libero_10_subtask",annotations)
    if kind=="libero_goal":return LiberoGoalIndex(root or GOAL_ROOT,annotations)
    if kind=="rollouts":
        from rollout_data import RolloutIndex
        if root is None:raise ValueError("Rollout datasets require an explicit root")
        return RolloutIndex(root,annotations)
    raise ValueError(kind)


def make_dataset(index,chunk_size=50):
    from mixed_data import MixedIndex,MixedDataset
    if isinstance(index,MixedIndex):return MixedDataset(index,chunk_size)
    from rollout_data import RolloutIndex,RolloutDataset
    if isinstance(index,RolloutIndex):return RolloutDataset(index,chunk_size)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    goal=isinstance(index,LiberoGoalIndex)
    dataset=LeRobotDataset("local/libero_goal_no_noops_lerobot" if goal else "lerobot/libero_10_subtask",
        root=index.root,video_backend="torchcodec",
        delta_timestamps={"action":[i/index.info["fps"] for i in range(chunk_size)]} if chunk_size else None)
    return CanonicalGoalDataset(dataset) if goal else dataset
