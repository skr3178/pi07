"""Freeze split and paired observations before training any prompt profile."""
import argparse
import json
from pathlib import Path

from train_goal_pilot import np,file_hash,write_json
from libero_data import make_index


def main(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if args.steps<1 or args.batch_size<1 or args.validation_examples<10:raise ValueError("Invalid sample count")
    index=make_index(args.dataset,args.dataset_root)
    split=index.split(args.seed,args.validation_per_task)
    train=index.schedule(split["train_episodes"],args.steps*args.batch_size,args.seed,goal_probability=1.)
    parts=[]
    tasks=sorted({int(index.tasks[index.bounds[e][0]]) for e in split["validation_episodes"]})
    for position,task in enumerate(tasks):
        eps=[e for e in split["validation_episodes"] if index.tasks[index.bounds[e][0]]==task]
        parts.append(index.schedule(eps,args.validation_examples//len(tasks)+int(position<args.validation_examples%len(tasks)),
                                    args.seed+1000+task,goal_probability=1.))
    validation={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order=np.random.default_rng(args.seed+2000).permutation(args.validation_examples)
    validation={k:v[order] for k,v in validation.items()}
    for name,schedule in (("train",train),("validation",validation)):
        assert np.all(index.episodes[schedule["current"]]==index.episodes[schedule["future"]])
        np.savez_compressed(out/f"{name}_schedule.npz",**schedule)
    write_json(out/"episode_split.json",split)
    write_json(out/"schedule_manifest.json",{"dataset":args.dataset,"dataset_root":str(index.root.resolve()),
        "dataset_info_sha256":file_hash(index.root/"meta/info.json"),"seed":args.seed,
        "train_examples":len(train["current"]),"validation_examples":args.validation_examples,
        "validation_task_counts":{int(t):int((index.tasks[validation['current']]==t).sum()) for t in tasks},
        "validation_episode_count":len(np.unique(index.episodes[validation["current"]])),
        "future_horizon_unit":"20 Hz simulator actions" if args.dataset=="rollouts" else "retained dataset frames; physical elapsed time not certified",
        "endpoint_definition":("episode end, no subtask labels" if args.dataset=="libero_goal" else
                               "current-predicate oracle phase end" if args.dataset=="rollouts" else "annotated segment end"),
        "source_sha256":file_hash(Path(__file__))})
    print(json.dumps({"dataset":args.dataset,"training_episodes":len(split['train_episodes']),
                      "validation_episodes":len(split['validation_episodes'])}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",choices=("libero_10","libero_goal","rollouts","mixed"),required=True)
    p.add_argument("--dataset-root",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--steps",type=int,default=1000);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--validation-per-task",type=int,default=5);p.add_argument("--validation-examples",type=int,default=100)
    p.add_argument("--seed",type=int,default=20261104)
    main(p.parse_args())
