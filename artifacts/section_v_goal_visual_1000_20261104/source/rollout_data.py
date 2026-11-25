"""Memory-mapped collection data with physical-initial-state grouped splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from section_v_data import SectionVIndex
from section_v_prompt import Evidence,PromptContext


class RolloutIndex(SectionVIndex):
    def __init__(self,root,annotations=None):
        self.root=Path(root).resolve();self.info=json.loads((self.root/"meta/info.json").read_text())
        if self.info.get("format")!="pi07_lossless_rollouts_v1":raise ValueError("Wrong rollout data format")
        meta=self.root/"meta/episodes.json"
        if hashlib.sha256(meta.read_bytes()).hexdigest()!=self.info["episode_metadata_sha256"]:
            raise ValueError("Rollout episode metadata changed")
        self.records=json.loads(meta.read_text());n=self.info["total_frames"]
        self.episodes=np.empty(n,dtype=np.int64);self.tasks=self.episodes.copy();self.frames=self.episodes.copy()
        self.end=self.episodes.copy();self.segment_end=self.episodes.copy();self.subtasks=self.episodes.copy()
        self.phase_text=np.empty(n,dtype=object);self.bounds={};self.task_names={int(k):v for k,v in self.info["task_names"].items()}
        names={};expected=0
        for row in self.records:
            ep,a,b=row["episode_index"],row["start"],row["end_exclusive"]
            if ep!=len(self.bounds) or a!=expected or b<=a:raise ValueError("Noncontiguous rollout episodes")
            expected=b;self.bounds[ep]=(a,b);self.episodes[a:b]=ep;self.tasks[a:b]=row["task_id"]
            self.frames[a:b]=np.arange(b-a);self.end[a:b]=b-1
            folder=(self.root/row["folder"]).resolve()
            if not folder.is_relative_to(self.root):raise ValueError("Rollout folder outside dataset")
            # Hash all inputs once per index creation; images thereafter use mmap.
            for name,digest in row["files_sha256"].items():
                with (folder/name).open("rb") as f:
                    if hashlib.file_digest(f,"sha256").hexdigest()!=digest:raise ValueError(f"Changed rollout array: {name}")
            texts=np.load(folder/"subtask_text.npy",allow_pickle=False)
            if len(texts)!=b-a:raise ValueError("Wrong phase count")
            self.phase_text[a:b]=texts
            self.subtasks[a:b]=[names.setdefault(t,len(names)) if t else -1 for t in texts]
            starts=np.r_[0,np.flatnonzero(texts[1:]!=texts[:-1])+1];ends=np.r_[starts[1:],len(texts)]
            for left,right in zip(starts,ends):self.segment_end[a+left:a+right]=a+right-1
        if expected!=n:raise ValueError("Wrong frame count")
        self.repaired_artifact_episodes=[];self.uncertain_subtask_episodes={};self.annotations={}
        if annotations is not None:self.load_annotations(annotations)

    def context(self,current):
        ep=int(self.episodes[current]);task=int(self.tasks[current]);a,b=self.bounds[ep]
        row=self.annotations.get(ep,{});mistake=Evidence()
        for left,right,label in row.get("mistakes",[]):
            if left<=self.frames[current]<right:mistake=label;break
        return PromptContext(self.task_names[task],
            subtask=Evidence(self.phase_text[current] or None,"current simulator predicate oracle"),
            speed_steps=Evidence(b-a,"executed actions at 20 Hz, excluding settling and inference pauses",1.),
            quality=row.get("quality",Evidence()),mistake=mistake)

    def split(self,seed=20261104,validation_per_task=1):
        # Every condition from a physical start stays on the same side.
        rng=np.random.default_rng(seed);train=[];val=[];groups={}
        for task in sorted(set(self.tasks)):
            records=[r for r in self.records if r["task_id"]==task]
            starts=np.array(sorted({r["initial_state_id"] for r in records}))
            if len(starts)<=validation_per_task:raise ValueError("Need separate train and validation start groups")
            rng.shuffle(starts);heldout=set(starts[:validation_per_task]);groups[int(task)]=sorted(int(x) for x in heldout)
            for r in records:(val if r["initial_state_id"] in heldout else train).append(r["episode_index"])
        return {"seed":seed,"train_episodes":sorted(train),"validation_episodes":sorted(val),
            "split_unit":"task and simulator initial-state ID, across every collection condition",
            "validation_initial_states_by_task":groups,"physical_initial_state_grouping_verified":True,
            "limitation":"No certified independence from base-model pretraining or source demonstrations."}

    def schedule(self,episodes,n,seed,goal_probability=.25):
        rng=np.random.default_rng(seed);allowed=np.flatnonzero(np.isin(self.episodes,episodes))
        current=rng.choice(allowed,n,replace=True);present=rng.random(n)<goal_probability;endpoint=rng.random(n)<.25
        maximum=np.minimum(80,self.end[current]-current)
        future=current+(rng.random(n)*(maximum+1)).astype(np.int64);future[endpoint]=self.segment_end[current[endpoint]]
        return {"current":current,"future":future,"goal_present":present,"segment_endpoint":endpoint}


class RolloutDataset:
    def __init__(self,index,chunk_size=50):self.index,self.chunk_size=index,chunk_size;self.arrays={}
    def __len__(self):return len(self.index.episodes)
    def __getitem__(self,current):
        ep=int(self.index.episodes[current]);frame=int(self.index.frames[current]);row=self.index.records[ep]
        if ep not in self.arrays:
            folder=self.index.root/row["folder"]
            self.arrays[ep]={k:np.load(folder/f"{k}.npy",mmap_mode="r",allow_pickle=False) for k in ("main","wrist","state","action")}
        data=self.arrays[ep]
        sample={"observation.state":torch.from_numpy(data["state"][frame].copy()),"task":row["task"]}
        for source,target in (("main","image"),("wrist","image2")):
            sample[f"observation.images.{target}"]=torch.from_numpy(data[source][frame].copy()).permute(2,0,1).float()/255
        if self.chunk_size:
            ids=frame+np.arange(self.chunk_size);sample["action_is_pad"]=torch.from_numpy(ids>=len(data["action"]))
            sample["action"]=torch.from_numpy(data["action"][np.minimum(ids,len(data["action"])-1)].copy())
        else:sample["action"]=torch.from_numpy(data["action"][frame].copy())
        return sample
