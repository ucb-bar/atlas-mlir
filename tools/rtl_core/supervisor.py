"""Parameterized owned-process resource guard for post-build RTL replay.

The compile probe's original guard remains frozen because its source bytes are
part of accepted build receipts. This guard uses the same descendant discovery.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from probe import helper, owned_processes
sys.dont_write_bytecode=True


def source_records(support):
    probe_module=sys.modules[owned_processes.__module__]
    if helper.__module__!=owned_processes.__module__:
        raise ValueError("Resource guard helpers came from different modules")
    return [support.identity(Path(__file__).absolute()),support.identity(probe_module.__file__)]


def bounded(command,cwd,env,log_path,support,*,affinity,rss_limit,wall_seconds):
    if set(os.sched_getaffinity(0))!=set(affinity):
        raise RuntimeError("Resource guard requires its declared outer CPU affinity")
    if rss_limit<=0 or wall_seconds<=0:
        raise ValueError("Resource limits must be positive")
    started,peak=time.monotonic(),0
    with support.safe_path(log_path).open("w") as log:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        usage={"command":command,"pid":process.pid,"cpu_affinity":sorted(affinity),
               "rss_limit_bytes":rss_limit,"wall_limit_seconds":wall_seconds,"rss_poll_seconds":0.5}
        support.write_json(log_path.with_suffix(".resources.json"),usage)
        try:
            while process.poll() is None:
                rss=0
                for pid in owned_processes(process.pid)|{os.getpid()}:
                    try:status=Path(f"/proc/{pid}/status").read_text().splitlines()
                    except FileNotFoundError:continue
                    rss+=sum(int(line.split()[1])*1024 for line in status if line.startswith("VmRSS:"))
                peak=max(peak,rss)
                if rss>rss_limit or time.monotonic()-started>wall_seconds:
                    raise RuntimeError("Owned job exceeded declared aggregate RSS or wall bound")
                time.sleep(0.5)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            usage.update(returncode=process.returncode,wall_seconds=time.monotonic()-started,peak_aggregate_rss_bytes=peak)
            support.write_json(log_path.with_suffix(".resources.json"),usage)
    return usage


def main():
    def interrupted(number,frame):raise SystemExit(f"Owned resource guard received signal {number}")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-helper",type=Path,required=True)
    parser.add_argument("--log",type=Path,required=True)
    parser.add_argument("--cpu",type=int,required=True)
    parser.add_argument("--rss-limit-bytes",type=int,required=True)
    parser.add_argument("--wall-seconds",type=int,required=True)
    parser.add_argument("command",nargs=argparse.REMAINDER)
    args=parser.parse_args();support=helper(args.support_helper)
    command=args.command[1:] if args.command[:1]==["--"] else args.command
    if len(command)<2:raise ValueError("Expected an explicit interpreter and script")
    records=source_records(support)+[support.identity(args.support_helper)]
    records += [support.identity(command[0]),support.identity(command[1])]
    result={"status":"failed","records":records}
    try:
        result["execution"]=bounded(command,Path.cwd(),os.environ.copy(),support.safe_path(args.log),support,
                                    affinity={args.cpu},rss_limit=args.rss_limit_bytes,wall_seconds=args.wall_seconds)
        if result["execution"]["returncode"]:raise RuntimeError("Guarded program failed")
        result["status"]="passed"
    finally:
        for record in records:support.identity(record["path"],record["sha256"])
        support.write_json(support.safe_path(args.log).with_suffix(".supervision.json"),result)


if __name__=="__main__":main()
