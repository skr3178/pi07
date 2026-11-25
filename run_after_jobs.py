"""Run an argument-list command after recorded local jobs succeed.

Use under background_job.py. Failed dependencies stop the chain; this never
restarts a failed job or treats an elapsed timeout as completion.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main(args,command):
    if not command:raise ValueError("A command is required after --")
    started=time.monotonic();last=None
    while True:
        states=[]
        for path in args.wait_for:
            job=json.loads(path.read_text());state=job["status"]
            if state=="failed" or (state=="complete" and job.get("exit_code")!=0):
                raise RuntimeError(f"Dependency failed: {path}")
            if state not in ("queued","starting","running","complete"):
                raise RuntimeError(f"Unknown dependency status {state}: {path}")
            states.append(state)
        if states!=last:print("Dependency states:",dict(zip(map(str,args.wait_for),states)),flush=True);last=states
        if all(state=="complete" for state in states):break
        if time.monotonic()-started>args.timeout:raise TimeoutError("Dependencies did not complete")
        time.sleep(15)
    subprocess.run(command,check=True)


if __name__=="__main__":
    argv=sys.argv[1:];separator=argv.index("--")
    p=argparse.ArgumentParser();p.add_argument("--wait-for",type=Path,nargs="+",required=True)
    p.add_argument("--timeout",type=float,default=86400)
    main(p.parse_args(argv[:separator]),argv[separator+1:])
