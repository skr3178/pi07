"""Validate oracle task coverage and prerequisite handling using actual BDDL."""
import argparse
import json
from pathlib import Path

from run_goal_development import ROOT,_get_suite,write_json
from libero.libero.envs.bddl_utils import robosuite_parse_problem
from subtask_controller import build_plan,select_subtask


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    suite=_get_suite("libero_10");report={"status":"running","tasks":[]}
    for i in range(10):
        parsed=robosuite_parse_problem(suite.get_task_bddl_file_path(i))
        predicates=parsed["goal_state"];plan=build_plan(predicates)
        truth=[False]*len(predicates);visited=[]
        for step in plan:
            selected=select_subtask(plan,truth)
            assert selected==step
            visited.append(step.text);truth[step.predicate_index]=True
        assert select_subtask(plan,truth) is None
        # If a previously satisfied condition becomes false, it is requested
        # again; elapsed simulation time cannot advance the oracle.
        truth[plan[0].predicate_index]=False
        assert select_subtask(plan,truth)==plan[0]
        if any(p[0].lower()=="close" for p in predicates):
            assert plan[-1].predicate[0].lower()=="close"
        report["tasks"].append({"task_id":i,"task":suite.get_task(i).language,"ordered_subtasks":visited})
    report["status"]="passed"
    report["scope"]="BDDL coverage and synthetic predicate transitions; live controller rollout still required"
    write_json(args.output/"report.json",report);print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
