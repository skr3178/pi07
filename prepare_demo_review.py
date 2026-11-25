"""Review the explicit demonstration subset used in the metadata toy study."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from train_goal_pilot import torch
from libero_data import make_index,make_dataset
from generated_goal_data import tensor_rgb


def main(args):
    torch.set_num_threads(4);info=json.loads((args.mixture/"meta/info.json").read_text())
    index=make_index("libero_10",Path(info["components"][0]["root"]));dataset=make_dataset(index,chunk_size=None)
    args.output.mkdir(parents=True,exist_ok=False);catalog=[]
    for ep in info["demo_episode_subset"]:
        start,end=index.bounds[ep];ids=np.unique(np.linspace(start,end-1,15).round().astype(int))
        grid=Image.new("RGB",(1536,48+280*((len(ids)+2)//3)),"white");draw=ImageDraw.Draw(grid)
        task=index.task_names[int(index.tasks[start])]
        draw.text((5,5),f"Demonstration episode {ep}, {end-start} retained frames; no quality label assigned",fill="black")
        draw.text((5,23),task,fill="black")
        for i,current in enumerate(ids):
            sample=dataset[int(current)];x,y=(i%3)*512,48+(i//3)*280
            draw.text((x+4,y+3),f"frame {current-start}, phase {index.subtasks[current]}",fill="black")
            grid.paste(Image.fromarray(np.concatenate([tensor_rgb(sample[f"observation.images.{k}"]) for k in ("image","image2")],axis=1)),(x,y+24))
        name=f"episode_{ep:03d}.png";grid.save(args.output/name)
        catalog.append({"episode_index":ep,"frames":(ids-start).tolist(),"sheet":name,"task":task,"review":"unreviewed"})
    (args.output/"catalog.json").write_text(json.dumps(catalog,indent=2)+"\n");print(f"Prepared {len(catalog)} demonstrations")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--mixture",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
