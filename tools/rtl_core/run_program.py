#!/usr/bin/env python3
"""Run authenticated raw instruction images on the freshly compiled AtlasCore."""
import argparse
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

sys.dont_write_bytecode=True
HELPER_SHA="80c9ec02d2e6b323cd3c2629750bac934ff02988561ea9c1468ac2f29e694505"


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def run(args):
    import hashlib
    path=args.support_helper.absolute()
    if ".." in path.parts or any(bad in part.lower() for part in path.parts for bad in ("hammer","vlsi")):
        raise ValueError("Unadmitted helper path")
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current/=part
        if stat.S_ISLNK(os.lstat(current).st_mode):raise ValueError("Helper symlink refused")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=HELPER_SHA:raise ValueError("Helper differs")
    support=module(path,"raw_rtl_support")
    out=support.safe_path(args.out);out.mkdir(parents=True,exist_ok=False)
    records=[]
    receipt={"schema_version":1,"status":"preparing","command":sys.argv,"records":records,
             "claim":"Fresh current AtlasCore raw-image functional check; explicit synthetic TL response policy",
             "native_quality_claim":False,"platform_performance_claim":False,
             "public_idle_only":True,"vpu_busy_observed":False,"quiet_cycles":32,"runs":[]}
    def bind(path,sha=None):
        record=support.identity(path,sha);records.append(record);return record
    try:
        bind(path,HELPER_SHA)
        runner=support.safe_path(args.runner_root)
        bind(runner/"receipt.json",args.runner_sha256)
        build=json.loads((runner/"receipt.json").read_text())
        if build["status"]!="passed_build_only" or not build["source_artifacts_preserved"]:
            raise ValueError("Raw runner build did not pass")
        for record in build["records"]:bind(record["path"],record["sha256"])
        executable=build["executable"];bind(executable["path"],executable["sha256"])
        probe_record=next(record for record in build["records"] if record["path"].endswith("/rtl-core-probe-v1/probe.py"))
        probe=module(support.safe_path(probe_record["path"]),"raw_frozen_supervisor")
        own=support.safe_path(Path(__file__).absolute().parent)
        for name in ("run_program.py","prepare_program.py"):
            record=bind(own/name)
            support.copy_source(record["path"],out/name,record["sha256"]);bind(out/name,record["sha256"])
        if args.cpu is not None:
            expected_probe=bind(own/"probe.py",probe_record["sha256"])
            record=bind(own/"supervisor.py")
            support.copy_source(record["path"],out/"supervisor.py",record["sha256"]);bind(out/"supervisor.py",record["sha256"])
            cpu_guard=module(out/"supervisor.py","frozen_single_cpu_supervisor")
            guard_sources=cpu_guard.source_records(support)
            if guard_sources[1]!=expected_probe:
                raise ValueError("CPU guard imported a probe outside the admitted source root")
            records.extend(guard_sources)
            receipt["single_cpu_budget"]={"cpu":args.cpu,"rss_limit_bytes":2*1024**3,"wall_seconds_per_run":args.run_wall_seconds}
        preparation=module(out/"prepare_program.py","frozen_raw_preparation")
        data,inputs=preparation.prepare(support,args.input,args.input_sha256,args.oracle,out,args.controls,
                                        reuse_model_arithmetic=args.reuse_model_arithmetic,model_root=args.model_root)
        records.extend(inputs)
        receipt["arithmetic_evidence"]=data["arithmetic_evidence"]
        support.write_json(out/"input.json",data);bind(out/"input.json")
        if args.prepare_only:
            receipt["status"]="prepared"
        else:
            env=os.environ.copy();env.update(build["environment"])
            for case in data["cases"]:
                for repeat in range(2 if case["id"]=="positive" else 1):
                    stem=case["id"]+"-"+str(repeat)
                    actual=out/(stem+"-actual.bin");trace=out/(stem+"-trace.jsonl");summary=out/(stem+"-summary.json")
                    memory=data["memory"]
                    command=[executable["path"],case["program"],data["initial"],str(actual),str(trace),str(42+repeat),
                             str(args.max_cycles),str(memory["dram_base"]),str(memory["input_bytes"]),
                             str(memory["output_offset"]),str(memory["output_bytes"]),str(summary)]
                    if args.cpu is None:
                        execution=probe.bounded(command,out,env,out/(stem+".log"),support)
                    else:
                        execution=cpu_guard.bounded(command,out,env,out/(stem+".log"),support,
                                                    affinity={args.cpu},rss_limit=2*1024**3,wall_seconds=args.run_wall_seconds)
                    item={"id":case["id"],"repeat":repeat,"execution":execution}
                    receipt["runs"].append(item)
                    support.write_json(out/"receipt.json",receipt)
                    for artifact in (actual,trace,summary,out/(stem+".log"),out/(stem+".resources.json")):
                        if support.safe_path(artifact).exists():bind(artifact)
                    if case["id"]=="wrong_halt":
                        if execution["returncode"]!=1 or "CSR status is not halted with ECALL reason2" not in (out/(stem+".log")).read_text():
                            raise AssertionError("Expected explicit CSR halt refusal did not occur")
                        item["physical_replay"]=preparation.replay(data,case,trace,None,case["program"])
                        item["status"]="expected_CSR_halt_refusal"
                        support.write_json(out/"receipt.json",receipt)
                        continue
                    if execution["returncode"]:raise RuntimeError("Raw RTL process failed: "+stem)
                    verdict=preparation.replay(data,case,trace,actual,case["program"])
                    result=json.loads(summary.read_text())
                    if any(result[key]!=value for key,value in data["requested_dma"].items()) or result["responses"]!=verdict["requests"]:
                        raise AssertionError("RTL summary traffic differs from independently replayed events")
                    if result.get("vpu_busy_observed") is not False or not result["guards_preserved"]:
                        raise AssertionError("Incorrect completion claim or guard result")
                    if result.get("csr_status")!=5 or result.get("halt_reason")!="ecall":
                        raise AssertionError("Summary did not confirm physical ECALL status")
                    item["physical_replay"]=verdict;item["summary"]=result
                    support.write_json(out/"receipt.json",receipt)
            positives=[support.safe_path(out/("positive-"+str(i)+"-actual.bin")).read_bytes() for i in range(2)]
            if positives[0]!=positives[1]:raise AssertionError("Initialized repeats differ")
            if args.controls and support.safe_path(out/"wrong_oracle-0-actual.bin").read_bytes()!=positives[0]:
                raise AssertionError("Wrong-oracle control changed physical execution")
            receipt["status"]="passed_raw_rtl_functional"
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
        if not receipt.get("source_artifacts_preserved"):raise RuntimeError("Raw run bookends failed")


if __name__=="__main__":
    import signal
    def interrupted(number,frame):raise SystemExit(f"Owned runner received signal {number}")
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root","input","oracle","support-helper","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("runner-sha256","input-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--controls",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--max-cycles",type=int,default=100000)
    parser.add_argument("--cpu",type=int,help="Explicit single CPU for post-build replay; preserves the compile supervisor epoch")
    parser.add_argument("--run-wall-seconds",type=int,default=600,help="Per-simulator wall bound for --cpu replay (default: 600)")
    parser.add_argument("--reuse-model-arithmetic",action="store_true",help="Reuse an authenticated complete-model actual payload; incompatible with controls")
    parser.add_argument("--model-root",type=Path,help="Explicit source root required for arithmetic reuse bookends")
    args=parser.parse_args()
    if args.run_wall_seconds<=0 or (args.run_wall_seconds!=600 and args.cpu is None):
        parser.error("A positive custom per-run wall bound requires --cpu")
    if args.reuse_model_arithmetic and (args.controls or args.model_root is None):
        parser.error("Arithmetic reuse requires --model-root and cannot enable controls")
    run(args)
