"""Local durable job launcher with PID identity and exit-status records.

Commands are argument lists, never shell strings. The monitor has its own
session so a chat/tool interruption need not kill an expensive experiment.
No automatic restart: callers must inspect both the job and experiment report.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def timestamp():return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:return Path(f"/proc/{pid}/stat").read_text().rsplit(")",1)[1].split()[19]
    except (OSError,IndexError):return None


def write(path,value):
    temporary=path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value,indent=2)+"\n");temporary.replace(path)


def worker(path):
    job=json.loads(path.read_text())
    job.update(monitor_pid=os.getpid(),monitor_identity=process_identity(os.getpid()),status="starting")
    write(path,job)
    try:
        with Path(job["log"]).open("a",buffering=1) as log:
            process=subprocess.Popen(job["command"],cwd=job["cwd"],stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT)
            job.update(pid=process.pid,pid_identity=process_identity(process.pid),status="running",started_utc=timestamp())
            write(path,job);code=process.wait()
        job.update(exit_code=code,status="complete" if code==0 else "failed",finished_utc=timestamp())
    except Exception:
        job.update(status="failed",error=traceback.format_exc(),finished_utc=timestamp())
    write(path,job)


def main(args):
    directory=args.job_dir.resolve();path=directory/"job.json"
    if args.mode=="start":
        command=args.command
        if command and command[0]=="--":command=command[1:]
        if not command:raise ValueError("Supply a command after --")
        directory.mkdir(parents=True,exist_ok=False)
        log=args.log.resolve() if args.log else directory/"run.log"
        log.parent.mkdir(parents=True,exist_ok=True)
        job={"status":"queued","command":command,"cwd":str(args.cwd.resolve()),"log":str(log),"created_utc":timestamp()}
        write(path,job)
        with (directory/"monitor.log").open("a") as monitor_log:
            monitor=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),"worker","--job-dir",str(directory)],
                cwd=args.cwd,stdin=subprocess.DEVNULL,stdout=monitor_log,stderr=subprocess.STDOUT,
                start_new_session=True,close_fds=True)
        print(json.dumps({"monitor_pid":monitor.pid,"job":str(path),"log":str(log)}))
    elif args.mode=="worker":worker(path)
    else:
        job=json.loads(path.read_text())
        for prefix in ("pid","monitor_pid"):
            identity="pid_identity" if prefix=="pid" else "monitor_identity"
            job[prefix+"_live"]=bool(job.get(prefix) and job.get(identity) and process_identity(job[prefix])==job[identity])
        print(json.dumps(job,indent=2))


if __name__=="__main__":
    arguments=sys.argv[1:];command=[]
    if "--" in arguments:
        split=arguments.index("--");command=arguments[split+1:];arguments=arguments[:split]
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("start","status","worker"))
    p.add_argument("--job-dir",type=Path,required=True);p.add_argument("--cwd",type=Path,default=Path.cwd())
    p.add_argument("--log",type=Path)
    args=p.parse_args(arguments);args.command=command
    main(args)
