"""Numerical CFG checks on a real decoded observation; no behavior claim."""
import argparse
from pathlib import Path
from train_goal_pilot import ROOT,np,torch,write_json,default_collate,flow_inputs,file_hash
from train_section_v import make_index,make_dataset,SectionVSamples,PromptOptions
from evaluate_libero import load_policy
from section_v_prompt import runtime_prompt
from metadata_guidance import guided_action_chunk


def main(args):
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    policy,pre,_,options=load_policy(args.checkpoint,"alt",device=args.device)
    index=make_index("libero_10",None,None);dataset=make_dataset(index)
    schedule=dict(np.load(ROOT/"artifacts/section_v_resume_schedule_20261104/validation_schedule.npz"))
    samples=SectionVSamples(dataset,index,schedule,options,training=False)
    sample=samples[0];context=index.context(int(schedule["current"][0]))
    shared=dict(options=options,subtask=context.subtask.value,subtask_source="offline dataset check",goal_available=True)
    pos=sample.copy();neg=sample.copy()
    pos["task"]=runtime_prompt(context.task,desired_speed_steps=context.speed_steps.value,**shared).text
    neg["task"]=runtime_prompt(context.task,**shared).text
    a=pre(default_collate([pos]));b=pre(default_collate([neg]));noise,_=flow_inputs(417,1);noise=noise.to(args.device)
    with torch.inference_mode():ordinary=policy.predict_action_chunk(a,noise=noise.clone())
    zero=guided_action_chunk(policy,a,b,beta=0,noise=noise,force_two_branches=True)
    same=guided_action_chunk(policy,a,a,beta=1.7,noise=noise,force_two_branches=True)
    guided=guided_action_chunk(policy,a,b,beta=1.3,noise=noise)
    zero_error=float((ordinary-zero).abs().max());same_error=float((ordinary-same).abs().max())
    assert zero_error<1e-6 and same_error<1e-6 and torch.isfinite(guided).all()
    corrupted=b.copy();corrupted["observation.state"]=b["observation.state"]+1
    try:guided_action_chunk(policy,a,corrupted,beta=1.3,noise=noise)
    except ValueError:pass
    else:raise AssertionError("Changed robot state was not rejected")
    report={"status":"passed","device":args.device,"checkpoint":str(args.checkpoint.resolve()),
        "checkpoint_sha256":file_hash(args.checkpoint/"model.safetensors"),
        "beta_convention":"v_cond + beta * (v_cond - v_no_metadata); zero is ordinary conditioned inference",
        "zero_guidance_max_error":zero_error,"identical_branch_max_error":same_error,
        "beta_1_3_action_max_change":float((ordinary-guided).abs().max()),"non_metadata_change_rejected":True,
        "conditioned_text":pos["task"],"unconditioned_text":neg["task"],
        "source_sha256":{f:file_hash(ROOT/f) for f in ("metadata_guidance.py","check_metadata_guidance.py")},
        "scope":"Numerical sampler/interface check with a two-update smoke checkpoint; not behavior efficacy."}
    write_json(args.output/"report.json",report);print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,default=ROOT/"artifacts/section_v_resume_full_20261104/checkpoint")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    main(p.parse_args())
