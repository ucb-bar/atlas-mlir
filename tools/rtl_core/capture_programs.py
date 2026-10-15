#!/usr/bin/env python3
"""Bound three selected native one-program model runs from frozen producers."""
import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from probe import helper, owned_processes
sys.dont_write_bytecode=True


def executable(path):
    """Admit each symlink target lexically before following its components."""
    original=Path(path).absolute();pending=original;seen=set();links=[]
    while True:
        if ".." in pending.parts or any(bad in part.lower() for part in pending.parts for bad in ("hammer","vlsi")):
            raise ValueError("Unadmitted interpreter target")
        current=Path(pending.anchor)
        for index,part in enumerate(pending.parts[1:],1):
            current/=part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                if current in seen:raise ValueError("Interpreter link loop")
                seen.add(current);target=Path(os.readlink(current))
                if ".." in target.parts or any(bad in component.lower() for component in target.parts for bad in ("hammer","vlsi")):
                    raise ValueError("Unadmitted interpreter link target")
                links.append({"path":str(current),"target":str(target)})
                pending=(target if target.is_absolute() else current.parent/target).joinpath(*pending.parts[index+1:])
                break
        else:return original,current,links


def run(args):
    support=helper(args.support_helper)
    session=support.safe_path(args.session_root);out=support.safe_path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    provider=support.safe_path(args.provider_root)
    records=[support.identity(Path(__file__).absolute())]
    frozen=json.loads(support.safe_path(provider/"snapshots.json").read_text())
    records += [support.identity(value["frozen"]["path"],value["frozen"]["sha256"]) for value in frozen["records"]]
    python,physical,links=executable(args.python)
    records.append(support.identity(physical))
    cases=[("resnet",session/"merlin/out/artifacts/recaptures/resnet50/atlas-sota/resnet-layer1-conv1-native-v2"),
           ("tinyllama",session/"tinyllama/merlin/out/artifacts/compare/atlas-sota/tinyllama-native-t1-v1"),
           ("smolvla",session/"tinyllama/merlin/out/artifacts/compare/atlas-sota/smolvla-native-s1-v1")]
    receipt={"schema_version":1,"status":"running","records":records,"interpreter_links":links,"jobs":[]}
    try:
        for name,source in cases:
            records += [support.identity(source/file) for file in ("operands.npz","manifest.json")]
            command=[str(python),str(provider/"run_model_program_contraction.py"),"--model-root",str(session/"npu-model"),
                     "--input",str(source/"operands.npz"),"--manifest",str(source/"manifest.json"),"--recipe","tensor-pow2",
                     "--repeats","1","--out",str(out/name)]
            env=os.environ.copy();env.pop("PYTHONPATH",None)
            env.update({"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONDONTWRITEBYTECODE":"1"})
            start=time.monotonic();peak=0
            with (out/(name+".log")).open("w") as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                                         preexec_fn=lambda:os.sched_setaffinity(0,{0}))
                item={"id":name,"pid":process.pid,"command":command};receipt["jobs"].append(item)
                support.write_json(out/"receipt.json",receipt)
                try:
                    while process.poll() is None:
                        rss=0
                        for pid in owned_processes(process.pid)|{os.getpid()}:
                            try:status=Path(f"/proc/{pid}/status").read_text().splitlines()
                            except FileNotFoundError:continue
                            rss+=sum(int(line.split()[1])*1024 for line in status if line.startswith("VmRSS:"))
                        peak=max(peak,rss)
                        if rss>2*1024**3 or time.monotonic()-start>600:raise RuntimeError("Native model job resource limit")
                        time.sleep(0.25)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    item.update(returncode=process.returncode,wall_seconds=time.monotonic()-start,peak_aggregate_rss_bytes=peak)
            if process.returncode:raise RuntimeError("Selected model program failed: "+name)
            records.append(support.identity(out/name/"receipt.json"))
            support.write_json(out/"receipt.json",receipt)
        receipt["status"]="passed_model_only"
    except BaseException as error:
        receipt["status"]="failed";receipt["error"]=f"{type(error).__name__}: {error}";raise
    finally:
        try:
            for record in records:support.identity(record["path"],record["sha256"])
            for link in links:
                if os.readlink(link["path"])!=link["target"]:raise ValueError("Interpreter link changed")
            receipt["source_artifacts_preserved"]=True
        except Exception as error:
            receipt["status"]="failed";receipt["source_artifacts_preserved"]=False;receipt["bookend_error"]=str(error)
        support.write_json(out/"receipt.json",receipt)
        if not receipt.get("source_artifacts_preserved"):raise RuntimeError("Native model source bookends failed")


if __name__=="__main__":
    def interrupted(number,frame):raise SystemExit(f"Owned model supervisor received signal {number}")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("session-root","provider-root","support-helper","python","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
