"""Offline Section V annotations and samples; no live subtask-time lookup."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from goal_data import GoalIndex, GoalSamples
from section_v_prompt import Evidence, PromptContext, build_prompt


# Audited meanings in dataset.md. Confidence was not quantitatively assessed;
# keep it unknown rather than inventing a numerical confidence score.
SUBTASKS = {
    0:"pick up the book and place it in the back compartment of the caddy",
    1:"close the microwave", 2:"close the drawer",
    3:"put the chocolate pudding to the right of the plate",
    4:"put the moka pot on the stove", 5:"put the first moka pot on the stove",
    6:"put the white mug on the plate", 7:"put the yellow and white mug on the right plate",
    8:"put the black bowl in the bottom drawer", 9:"put the alphabet soup in the basket",
    10:"pick up the book and place it in the back compartment of the caddy",
    11:"put the butter in the basket", 12:"put the cream cheese box in the basket",
    13:"put the tomato sauce in the basket", 14:"put the yellow and white mug in the microwave",
    15:"turn on the stove",
}


def evidence_from_json(value):
    if not isinstance(value,dict) or set(value)-{"value","source","confidence"}:
        raise ValueError("Each annotation must contain value, source and optional confidence")
    if not {"value","source"} <= set(value):
        raise ValueError("Missing annotation value or source")
    return Evidence(**value)


class SectionVIndex(GoalIndex):
    def __init__(self, root, annotations=None):
        super().__init__(root)
        table = self.table.to_pandas().sort_values("index")
        self.subtasks = table.subtask_index.to_numpy().copy()
        for episode in self.repaired_artifact_episodes:
            start,_ = self.bounds[episode]
            self.subtasks[start] = self.subtasks[start+1]
        names = pq.read_table(self.root / "meta/tasks.parquet").to_pydict()
        self.task_names = dict(zip(names["task_index"],names["__index_level_0__"]))
        unknown = set(np.unique(self.subtasks))-set(SUBTASKS)
        if unknown:
            raise ValueError(f"Unmapped subtask IDs: {unknown}")
        if np.any((self.subtasks==0)&(self.tasks!=9)):
            raise ValueError("Subtask ID 0 alias is only validated for the book task")
        self.annotations = {}
        if annotations is not None:
            self.load_annotations(annotations)

    def load_annotations(self, path):
        """JSONL: episode_index, optional quality and nonoverlapping mistakes.

        Mistakes have start_frame, end_frame_exclusive and label. A success bit
        alone is deliberately not accepted as a quality or mistake annotation.
        """
        loaded = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if not isinstance(row,dict) or set(row)-{"episode_index","quality","mistakes"}:
                raise ValueError("Unknown annotation fields")
            ep = row.get("episode_index")
            if type(ep) is not int or ep not in self.bounds or ep in loaded:
                raise ValueError("Unknown/duplicate episode index")
            quality = evidence_from_json(row["quality"]) if "quality" in row else Evidence()
            PromptContext("validate",quality=quality)
            start,end = self.bounds[ep]
            if not np.array_equal(self.frames[start:end],np.arange(end-start)):
                raise ValueError("Annotations require contiguous zero-based episode frame indices")
            intervals = []
            for item in row.get("mistakes",[]):
                if set(item)!={"start_frame","end_frame_exclusive","label"}:
                    raise ValueError("Invalid mistake interval schema")
                a,b=item["start_frame"],item["end_frame_exclusive"]
                if type(a) is not int or type(b) is not int or not 0<=a<b<=end-start:
                    raise ValueError("Mistake interval lies outside the episode")
                label=evidence_from_json(item["label"])
                PromptContext("validate",mistake=label)
                intervals.append((a,b,label))
            intervals.sort(key=lambda item:item[0])
            if any(left[1]>right[0] for left,right in zip(intervals,intervals[1:])):
                raise ValueError("Overlapping mistake intervals")
            loaded[ep]={"quality":quality,"mistakes":intervals}
        self.annotations=loaded

    def context(self, current):
        if not 0<=current<len(self.episodes):
            raise IndexError(current)
        episode,task,subtask = int(self.episodes[current]),int(self.tasks[current]),int(self.subtasks[current])
        start,end=self.bounds[episode]
        row=self.annotations.get(episode,{})
        mistake=Evidence()
        for a,b,label in row.get("mistakes",[]):
            if a<=self.frames[current]<b:
                mistake=label;break
        source=("dataset.md: task co-occurrence and sampled video inspection" if subtask in (9,11,12,13)
                else "dataset.md: sampled segment start/end inspection")
        text=SUBTASKS[subtask]
        if task==7 and subtask==6:
            text="put the white mug on the left plate"
        return PromptContext(self.task_names[task],
            subtask=Evidence(text,source),
            speed_steps=Evidence(end-start,"dataset episode retained-frame count; not verified physical duration",1.),
            quality=row.get("quality",Evidence()),mistake=mistake)


class SectionVSamples:
    """Drop-in map-style training dataset with deterministic prompt masks.

    Pass the resulting item to make_section_v_processors and GoalSmolVLAPolicy.
    Annotation provenance stays in audit records, outside model input tensors.
    """
    def __init__(self, dataset, index, schedule, options, *, training, seed=20261030):
        self.index,self.options=index,options
        self.schedule={k:np.array(v,copy=True) for k,v in schedule.items()}
        current,future=self.schedule["current"],self.schedule["future"]
        if len(current)!=len(future) or np.any(current<0) or np.any(future>=len(index.episodes)):
            raise ValueError("Invalid sample indices")
        if np.any(future<current) or np.any(index.episodes[current]!=index.episodes[future]):
            raise ValueError("Goals must stay in the same episode and not precede the observation")
        self.prompts=[build_prompt(index.context(int(c)),options,training=training,seed=seed,sample_key=i)
                      for i,c in enumerate(current)]
        self.schedule["goal_present"]=np.array([p.goal_present for p in self.prompts],dtype=bool)
        self.samples=GoalSamples(dataset,self.schedule,"B")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,i):
        sample=self.samples[i]
        sample["task"]=self.prompts[i].text
        return sample

    def audit(self,i):
        current=int(self.schedule["current"][i])
        return {"sample":i,"index":current,"episode_index":int(self.index.episodes[current]),
                "frame_index":int(self.index.frames[current]),"task_id":int(self.index.tasks[current]),
                "subtask_id":int(self.index.subtasks[current]),"future_index":int(self.schedule["future"][i]),
                **asdict(self.prompts[i])}

    def write_audit(self,path):
        with Path(path).open("w") as f:
            for i in range(len(self)):
                f.write(json.dumps(self.audit(i))+"\n")
