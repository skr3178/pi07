"""Audited cached ImageWAM goals; no silent replacement of real-image targets."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from goal_policy import GOALS


def tensor_rgb(tensor):
    return (tensor.detach().cpu().permute(1,2,0).clamp(0,1)*255).round().byte().numpy()


def observation_digest(sample):
    digest=hashlib.sha256()
    for key in ("observation.images.image","observation.images.image2"):
        digest.update(tensor_rgb(sample[key]).tobytes())
    digest.update(np.asarray(sample["observation.state"],dtype="<f4").tobytes())
    return digest.hexdigest()


def generation_prompt(index,current,mode):
    context=index.context(current)
    if mode=="task":return context.task
    if mode=="subtask":return context.subtask.value or context.task
    raise ValueError("Unknown world-model prompt mode")


class GeneratedGoalCache:
    def __init__(self,path,index,*,allow_unreviewed=False):
        self.path=Path(path).resolve();self.root=self.path.parent
        manifest=json.loads(self.path.read_text())
        if manifest.get("status")!="complete":raise ValueError("Incomplete goal cache")
        if Path(manifest["dataset_root"]).resolve()!=index.root.resolve():
            raise ValueError("Goal cache belongs to a different dataset")
        info_hash=hashlib.sha256((index.root/"meta/info.json").read_bytes()).hexdigest()
        if manifest["dataset_info_sha256"]!=info_hash:raise ValueError("Dataset metadata changed")
        self.entries={}
        for entry in manifest["entries"]:
            current=entry["current_index"]
            if type(current) is not int or not 0<=current<len(index.episodes) or current in self.entries:
                raise ValueError("Invalid or duplicate cache index")
            if (entry["episode_index"]!=int(index.episodes[current]) or
                entry["frame_index"]!=int(index.frames[current])):
                raise ValueError("Cache episode/frame mismatch")
            if entry["prompt"]!=generation_prompt(index,current,manifest["prompt_mode"]):
                raise ValueError("Cached prompt no longer matches current annotations")
            if entry.get("review")!="accepted" and not allow_unreviewed:
                continue
            image=(self.root/entry["image"]).resolve()
            if not image.is_relative_to(self.root):raise ValueError("Cache image lies outside cache directory")
            if hashlib.sha256(image.read_bytes()).hexdigest()!=entry["image_sha256"]:
                raise ValueError("Cached image hash mismatch")
            with Image.open(image) as im:
                if im.size!=(448,224) or im.mode!="RGB":raise ValueError("Incorrect multiview image format")
            self.entries[current]=entry

    def images(self,current,sample):
        entry=self.entries[current]
        if observation_digest(sample)!=entry["observation_sha256"]:
            raise ValueError("Cached goal was generated from a different observation/state")
        rgb=np.asarray(Image.open(self.root/entry["image"]).convert("RGB"))
        output=[]
        for view in (rgb[:,:224],rgb[:,224:]):
            arr=np.asarray(Image.fromarray(view).resize((256,256),Image.Resampling.BILINEAR)).copy()
            output.append(torch.from_numpy(arr).permute(2,0,1).float()/255)
        return output


class MixedGoalSamples:
    """Keep all action targets. Replace a declared subset of visible real goals."""
    def __init__(self,samples,cache,probability,seed):
        if not 0<=probability<=1:raise ValueError("Invalid generated-goal mixture")
        self.samples,self.cache=samples,cache
        self.prompts,self.schedule=samples.prompts,samples.schedule
        rng=np.random.default_rng(seed+291)
        self.generated=np.array([p.goal_present and int(c) in cache.entries and draw<probability
            for p,c,draw in zip(self.prompts,self.schedule["current"],rng.random(len(samples)))])
        self.covered_visible=sum(p.goal_present and int(c) in cache.entries
                                 for p,c in zip(self.prompts,self.schedule["current"]))

    def __len__(self):return len(self.samples)

    def __getitem__(self,i):
        sample=self.samples[i]
        if self.generated[i]:
            for key,image in zip(GOALS,self.cache.images(int(self.schedule["current"][i]),sample)):
                sample[key]=image
        return sample

    def audit(self,i):
        row=self.samples.audit(i)
        row["goal_source"]="generated" if self.generated[i] else "real_future" if self.prompts[i].goal_present else "absent"
        if self.generated[i]:
            entry=self.cache.entries[int(self.schedule["current"][i])]
            row["generated_image_sha256"]=entry["image_sha256"]
            row["generation_prompt"]=entry["prompt"]
            row["generation_review"]=entry["review"]
        return row

    def write_audit(self,path):
        with Path(path).open("w") as f:
            for i in range(len(self)):f.write(json.dumps(self.audit(i))+"\n")
