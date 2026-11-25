"""Explicit demonstration/rollout mixture with shared task and action interfaces."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from section_v_data import SectionVIndex
from section_v_prompt import Evidence
from rollout_data import RolloutIndex,RolloutDataset


class MixedIndex(SectionVIndex):
    def __init__(self,root,annotations=None):
        self.root=Path(root).resolve();self.info=json.loads((self.root/"meta/info.json").read_text())
        if self.info.get("format")!="pi07_demo_rollout_mixture_v1":raise ValueError("Wrong mixed format")
        components=[]
        for item,cls in zip(self.info["components"],(SectionVIndex,RolloutIndex)):
            path=Path(item["root"])
            if hashlib.sha256((path/"meta/info.json").read_bytes()).hexdigest()!=item["info_sha256"]:
                raise ValueError("Mixed-data component changed")
            components.append(cls(path))
        self.components=components;demo,rollout=components
        if demo.task_names!=rollout.task_names:raise ValueError("Mixed tasks must map by identical full instructions")
        self.task_names=demo.task_names;self.frame_offset=len(demo.episodes);self.episode_offset=len(demo.bounds)
        self.episodes=np.r_[demo.episodes,rollout.episodes+self.episode_offset]
        self.tasks=np.r_[demo.tasks,rollout.tasks];self.frames=np.r_[demo.frames,rollout.frames]
        self.end=np.r_[demo.end,rollout.end+self.frame_offset]
        self.segment_end=np.r_[demo.segment_end,rollout.segment_end+self.frame_offset]
        self.subtasks=np.r_[demo.subtasks,rollout.subtasks]
        self.bounds=dict(demo.bounds)
        self.bounds.update({ep+self.episode_offset:(a+self.frame_offset,b+self.frame_offset) for ep,(a,b) in rollout.bounds.items()})
        self.uncertain_subtask_episodes=demo.uncertain_subtask_episodes
        self.repaired_artifact_episodes=demo.repaired_artifact_episodes
        self.annotations={}
        if annotations is not None:self.load_annotations(annotations)

    def context(self,current):
        which=int(current>=self.frame_offset);local=current-(self.frame_offset if which else 0)
        context=self.components[which].context(local);episode=int(self.episodes[current])
        row=self.annotations.get(episode,{});mistake=Evidence()
        for a,b,label in row.get("mistakes",[]):
            if a<=self.frames[current]<b:mistake=label;break
        return replace(context,quality=row.get("quality",Evidence()),mistake=mistake)

    def split(self,seed=20261104,validation_per_task=1):
        path=Path(self.info["demo_split"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.info["demo_split_sha256"]:raise ValueError("Demo split changed")
        demo=json.loads(path.read_text());rollout=self.components[1].split(seed,validation_per_task)
        selected=set(self.info["demo_episode_subset"])
        return {"seed":seed,"train_episodes":[ep for ep in demo["train_episodes"] if ep in selected]+
                [ep+self.episode_offset for ep in rollout["train_episodes"]],
            "validation_episodes":[ep for ep in demo["validation_episodes"] if ep in selected]+
                [ep+self.episode_offset for ep in rollout["validation_episodes"]],
            "split_unit":"frozen original demo episodes plus simulator-start-grouped rollout episodes",
            "physical_initial_state_grouping_verified":False,
            "rollout_split":rollout,"limitation":"Demo physical starts and overlap with rollout starts/pretraining are unverified."}

    def schedule(self,episodes,n,seed,goal_probability=.25):
        rng=np.random.default_rng(seed);current=np.empty(n,dtype=np.int64)
        allowed=np.flatnonzero(np.isin(self.episodes,episodes))
        pools=[allowed[allowed<self.frame_offset],allowed[allowed>=self.frame_offset]]
        use_rollout=rng.random(n)<self.info["rollout_probability"]
        if not len(pools[0]):use_rollout[:]=True
        if not len(pools[1]):use_rollout[:]=False
        for which,pool in enumerate(pools):
            mask=use_rollout==bool(which)
            if mask.any():current[mask]=rng.choice(pool,int(mask.sum()),replace=True)
        present=rng.random(n)<goal_probability;endpoint=rng.random(n)<.25
        limits=np.where(current<self.frame_offset,40,80)
        maximum=np.minimum(limits,self.end[current]-current)
        future=current+(rng.random(n)*(maximum+1)).astype(np.int64);future[endpoint]=self.segment_end[current[endpoint]]
        return {"current":current,"future":future,"goal_present":present,"segment_endpoint":endpoint,
                "rollout_sample":current>=self.frame_offset}


class MixedDataset:
    def __init__(self,index,chunk_size=50):
        from libero_data import make_dataset
        self.index=index;self.datasets=[make_dataset(index.components[0],chunk_size),RolloutDataset(index.components[1],chunk_size)]
    def __len__(self):return len(self.index.episodes)
    def __getitem__(self,i):
        which=int(i>=self.index.frame_offset)
        return self.datasets[which][i-(self.index.frame_offset if which else 0)]
