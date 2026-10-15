#!/usr/bin/env python3
"""Replay only make with explicit generated C++ and used runtime source pins."""
import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
sys.dont_write_bytecode=True
HELPER_SHA="2b089e369bee6f8d540c588e089b96395d82a18b2ec22edc95983e6a1764b6c4"


def lexical(value):
    path=Path(value)
    if ".." in path.parts or any(bad in part.lower() for part in path.parts for bad in ("hammer","vlsi")):
        raise ValueError("Required dependency has an unadmitted lexical component")


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def run(args):
    lexical(args.support_helper)
    helper=args.support_helper.absolute()
    lexical(helper)
    current=Path(helper.anchor)
    for part in helper.parts[1:]:
        current/=part
        if stat.S_ISLNK(os.lstat(current).st_mode):raise ValueError("Helper symlink refused")
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError("Helper differs")
    support=load(helper,"rebuild_support")
    source=support.safe_path(args.source);out=support.safe_path(args.out)
    support.identity(source/"receipt.json",args.source_sha256)
    old=json.loads((source/"receipt.json").read_text())
    if old["status"]!="passed_build_only" or not old["source_artifacts_preserved"]:
        raise ValueError("Prior build is unavailable")
    out.mkdir(parents=True,exist_ok=False);(out/"obj_dir").mkdir()
    records=[]
    receipt={"schema_version":1,"status":"preparing","records":records,"command":sys.argv,
             "claim":"Make replay with generated C++ and exact used Verilator runtime sources pinned before/after",
             "environment":old["environment"],"prior_build":args.source_sha256}
    def bind(path,sha=None):
        value=support.identity(path,sha);records.append(value);return value
    try:
        bind(source/"receipt.json",args.source_sha256)
        for record in old["records"]:bind(record["path"],record["sha256"])
        own=bind(Path(__file__).absolute())
        support.copy_source(own["path"],out/"rebuild_runner.py",own["sha256"]);bind(out/"rebuild_runner.py",own["sha256"])
        original_obj=source/"obj_dir"
        names=sorted(entry.name for entry in os.scandir(original_obj))
        for name in names:lexical(name)
        runtime=support.safe_path(old["environment"]["VERILATOR_ROOT"])
        used=set()
        generator_inputs={Path(old["generate"]["command"][0]),source/"AtlasCore.sv"}
        if not generator_inputs <= {Path(record["path"]) for record in records}:
            raise ValueError("Generator inputs were not authenticated by the prior build")
        depfiles=[name for name in names if name.endswith(".d")]
        if not depfiles:raise ValueError("No explicit generated dependency files")
        for name in depfiles:
            bind(original_obj/name)
            content=support.safe_path(original_obj/name).read_text().replace("\\\n"," ")
            for value in content.partition(":")[2].split():
                lexical(value) # No metadata access to this dependency before admission.
                path=Path(value)
                path=path if path.is_absolute() else original_obj/path
                if path.is_relative_to(runtime):used.add(path)
                elif path.is_relative_to(original_obj) or path==source/"raw_runner.cpp" or path in generator_inputs:pass
                else:raise ValueError("Unexpected required dependency root")
        used.add(runtime/"include/verilated.mk")
        receipt["runtime_source_closure"]=[bind(path) for path in sorted(used)]
        receipt["dependency_file_count"]=len(depfiles)
        generated=[]
        for name in names:
            if not name.endswith((".cpp",".h",".mk")):continue
            record=bind(original_obj/name)
            target=out/"obj_dir"/name
            support.copy_source(record["path"],target,record["sha256"]);generated.append(bind(target,record["sha256"]))
        receipt["generated_sources"]=generated
        probe_record=next(record for record in records if record["path"].endswith("/rtl-core-probe-v1/probe.py"))
        probe=load(support.safe_path(probe_record["path"]),"frozen_rebuild_supervisor")
        command=old["compile"]["command"].copy()
        command[command.index("-C")+1]=str(out/"obj_dir")
        command.append("OBJCACHE=")
        receipt["build_command"]=command
        support.write_json(out/"receipt.json",receipt)
        if args.prepare_only:
            receipt["status"]="prepared"
        else:
            env=os.environ.copy();env.update(old["environment"])
            receipt["compile"]=probe.bounded(command,out,env,out/"compile.log",support)
            bind(out/"compile.log");bind(out/"compile.resources.json")
            if receipt["compile"]["returncode"]:raise RuntimeError("Pinned make replay failed")
            for entry in os.scandir(out/"obj_dir"):
                lexical(entry.name)
                if not entry.name.endswith(".d"):continue
                content=support.safe_path(out/"obj_dir"/entry.name).read_text().replace("\\\n"," ")
                for value in content.partition(":")[2].split():
                    lexical(value)
                    dependency=Path(value)
                    dependency=dependency if dependency.is_absolute() else out/"obj_dir"/dependency
                    if dependency.is_relative_to(runtime) and dependency not in used:
                        raise ValueError("Replayed compiler used an unpinned runtime source")
                    if not (dependency.is_relative_to(runtime) or dependency.is_relative_to(out/"obj_dir") or dependency==source/"raw_runner.cpp"):
                        raise ValueError("Replayed compiler used an unexpected source root")
            receipt["executable"]=bind(out/"obj_dir/raw_runner")
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
        if not receipt.get("source_artifacts_preserved"):raise RuntimeError("Make replay bookends failed")


if __name__=="__main__":
    import signal
    def interrupted(number,frame):raise SystemExit(f"Owned replay received signal {number}")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","support-helper","out"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-sha256",required=True)
    parser.add_argument("--prepare-only",action="store_true")
    run(parser.parse_args())
