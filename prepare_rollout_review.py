"""Prepare trace contact sheets and missing-label review templates."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def main(args):
    root=args.run.resolve();report=json.loads((root/"report.json").read_text())
    if report["status"]!="complete":raise ValueError("Review a completed collection")
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);catalog=[]
    for ep,row in enumerate(report["episodes"]):
        if "error" in row:raise ValueError("Failed collection entry")
        with np.load(root/row["training_data"]["file"]) as data:
            ids=np.unique(np.linspace(0,len(data["action"])-1,args.frames).round().astype(int))
            grid=Image.new("RGB",(1536,48+280*((len(ids)+2)//3)),"white");draw=ImageDraw.Draw(grid)
            draw.text((5,5),f"Episode {ep}: {row['condition']} task {row['task_id']} start {row['initial_state_id']}; success={row['success']} steps={row['steps']}",fill="black")
            draw.text((5,23),row['task'],fill="black")
            for i,frame in enumerate(ids):
                x,y=(i%3)*512,48+(i//3)*280
                draw.text((x+4,y+3),f"step {frame}, predicate truth {data['predicates'][frame].astype(int).tolist()}",fill="black")
                grid.paste(Image.fromarray(np.concatenate([data['main'][frame],data['wrist'][frame]],axis=1)),(x,y+24))
            image=f"episode_{ep:03d}.png";grid.save(out/image)
        catalog.append({"episode_index":ep,"sheet":image,"frames":ids.tolist(),"task":row["task"],
            "condition":row["condition"],"initial_state_id":row["initial_state_id"],"success":row["success"],
            "raw_recording":str(root/row["training_data"]["file"]),"review_status":"unreviewed"})
    (out/"catalog.json").write_text(json.dumps(catalog,indent=2)+"\n")
    (out/"README.md").write_text("# Mixed-behavior review\n\nThese sheets sample the whole trace. They support a provisional overall quality judgment, not precise mistake boundaries. Inspect raw frames/video when assigning a mistake interval. Success and predicate counts remain separate measurements.\n\nProvisional quality rubric: 1 = no meaningful progress with persistent failure; 2 = attempted/partial progress with major errors; 3 = substantial coherent progress but unresolved failure or major recovery; 4 = completion with minor corrections or inefficiency; 5 = clean, effective completion with no evident errors. Record a rationale, reviewer and evidence for every assigned label; do not automatically convert success to a rating. Unclear labels remain missing.\n\nAssistant visual ratings must be identified as such. They are not independent human annotations and do not establish the paper's metadata effect.\n")
    print(json.dumps({"episodes":len(catalog),"output":str(out),"assigned_labels":0}))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frames",type=int,default=15);main(p.parse_args())
