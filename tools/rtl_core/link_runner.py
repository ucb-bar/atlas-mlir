#!/usr/bin/env python3
"""Compile only the checked harness and relink authenticated current-core objects."""
import argparse
import json
import os
import sys
from pathlib import Path
from probe import helper, bounded
sys.dont_write_bytecode=True


def run(args):
    support=helper(args.support_helper)
    source=support.safe_path(args.source);own=support.safe_path(Path(__file__).absolute().parent)
    support.identity(source/"receipt.json",args.source_sha256)
    old=json.loads((source/"receipt.json").read_text())
    if old["status"]!="passed_build_only" or not old["source_artifacts_preserved"]:
        raise ValueError("Prior source-bound build did not pass")
    out=support.safe_path(args.out);out.mkdir(parents=True,exist_ok=False)
    records=[]
    receipt={"schema_version":1,"status":"preparing","records":records,"command":sys.argv,
             "claim":"Checked harness object compile and relink against authenticated generated current-core objects",
             "environment":old["environment"],"historical_source_mappings":[]}
    def bind(path,sha=None):
        record=support.identity(path,sha);records.append(record);return record
    try:
        bind(args.support_helper)
        bind(source/"receipt.json",args.source_sha256)
        historical=next(record for record in old["records"] if record["path"].endswith("/rtl-core-runner-v1/raw_runner.cpp"))
        bind(historical["path"],historical["sha256"])
        for record in old["records"]:
            if record["path"]==str(own/"raw_runner.cpp"):
                if record["sha256"]!=historical["sha256"]:raise ValueError("Historical harness epoch differs")
                receipt["historical_source_mappings"].append({"original":record,"frozen":historical})
            else:bind(record["path"],record["sha256"])
        for name in ("link_runner.py","probe.py","raw_runner.cpp"):
            record=bind(own/name)
            support.copy_source(record["path"],out/name,record["sha256"]);bind(out/name,record["sha256"])
        reused=[bind(source/"obj_dir"/name) for name in ("verilated.o","verilated_threads.o","VAtlasCore__ALL.a")]
        receipt["reused_objects"]=reused
        compiler=next(record for record in records if record["path"].endswith("/x86_64-conda-linux-gnu-c++"))
        runtime=support.safe_path(old["environment"]["VERILATOR_ROOT"])
        compile=[compiler["path"],"-O0","-std=c++17","-faligned-new","-fcf-protection=none",
                 "-DVM_COVERAGE=0","-DVM_SC=0","-DVM_TRACE=0","-DVM_TRACE_FST=0","-DVM_TRACE_VCD=0",
                 "-I"+str(source/"obj_dir"),"-I"+str(runtime/"include"),"-I"+str(runtime/"include/vltstd"),
                 "-MMD","-MF",str(out/"raw_runner.d"),"-c",str(out/"raw_runner.cpp"),"-o",str(out/"raw_runner.o")]
        link=[compiler["path"],str(out/"raw_runner.o")]+[record["path"] for record in reused]+[
             "-pthread","-lpthread","-latomic","-o",str(out/"raw_runner")]
        env=os.environ.copy();env.update(old["environment"])
        for stage,command in (("compile",compile),("link",link)):
            receipt[stage]=bounded(command,out,env,out/(stage+".log"),support)
            bind(out/(stage+".log"));bind(out/(stage+".resources.json"))
            if receipt[stage]["returncode"]:raise RuntimeError("Checked harness "+stage+" failed")
            if stage=="compile":bind(out/"raw_runner.o")
        dependencies=support.safe_path(out/"raw_runner.d").read_text().replace("\\\n"," ").partition(":")[2].split()
        for value in dependencies:
            dependency=support.safe_path(value)
            if not any(record["path"]==str(dependency) for record in records):
                raise ValueError("Checked harness used a source outside the prior pinned closure")
        bind(out/"raw_runner.d")
        receipt["executable"]=bind(out/"raw_runner")
        receipt["status"]="passed_build_only"
    except BaseException as error:
        receipt["status"]="failed";receipt["error"]=f"{type(error).__name__}: {error}";raise
    finally:
        try:
            for record in records:support.identity(record["path"],record["sha256"])
            receipt["source_artifacts_preserved"]=True
        except Exception as error:
            receipt["status"]="failed";receipt["source_artifacts_preserved"]=False
            receipt["bookend_error"]=f"{type(error).__name__}: {error}"
        support.write_json(out/"receipt.json",receipt)
        if not receipt.get("source_artifacts_preserved"):raise RuntimeError("Checked link bookends failed")


if __name__=="__main__":
    import signal
    def interrupted(number,frame):raise SystemExit(f"Owned linker received signal {number}")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","support-helper","out"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-sha256",required=True)
    run(parser.parse_args())
