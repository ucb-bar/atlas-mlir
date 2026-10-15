"""Authenticate retained raw images and independently rebuild their arithmetic."""
import importlib.util
import json
from pathlib import Path
import numpy as np


def prepare(support, source, receipt_sha, oracle_path, out, controls):
    source, out = support.safe_path(source), support.safe_path(out)
    records = [support.identity(source / "receipt.json", receipt_sha)]
    original = json.loads((source / "receipt.json").read_text())
    if original["status"] != "passed_model_only":
        raise ValueError("Source model program did not pass")
    recipe = original["recipe"]
    if recipe not in ("atlas_mxu0_fp8_unit_bf16acc_v1", "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1"):
        raise ValueError("Unsupported source recipe")
    for name in ("initial-dram.bin", "program.bin", "encoded.npz", "expected-bf16-tiles.bin"):
        record = support.identity(source / name, original["artifacts"][name]); records.append(record)
        support.copy_source(record["path"], out / name, record["sha256"])
        records.append(support.identity(out / name, record["sha256"]))
    oracle = support.identity(oracle_path, original["code_before"]["atlas_tile_arithmetic.py"])
    records.append(oracle)
    support.copy_source(oracle["path"], out / "oracle.py", oracle["sha256"])
    records.append(support.identity(out / "oracle.py", oracle["sha256"]))
    spec = importlib.util.spec_from_file_location("raw_program_oracle", out / "oracle.py")
    arithmetic = importlib.util.module_from_spec(spec); spec.loader.exec_module(arithmetic)
    m,k,n = original["shape"]
    if any(value <= 0 or value % 32 for value in (m,k,n)):
        raise ValueError("Only complete physical tiles are admitted")
    with np.load(out / "encoded.npz", allow_pickle=False) as bundle:
        if set(bundle.files) != {"A","B"}:
            raise ValueError("Unexpected encoded operands")
        a,b = bundle["A"].copy(),bundle["B"].copy()
    if a.dtype != np.uint8 or b.dtype != np.uint8 or a.shape != (m,k) or b.shape != (k,n):
        raise ValueError("Encoded layout differs")
    inputs = b"".join(a[row:row+32,inner:inner+32].tobytes() for row in range(0,m,32) for inner in range(0,k,32))
    inputs += b"".join(b[inner:inner+32,col:col+32].tobytes() for col in range(0,n,32) for inner in range(0,k,32))
    memory = original["memory"]
    if memory["dram_base"] != 1<<32 or memory["input_bytes"] != len(inputs) or memory["output_offset"] != len(inputs)+256:
        raise ValueError("Unsupported non-aliasing memory contract")
    initial = (out / "initial-dram.bin").read_bytes()
    expected = (out / "expected-bf16-tiles.bin").read_bytes()
    requested={"load_bytes":(m//32)*(n//32)*(2*k//32*1024),"store_bytes":m*n*2}
    if len(expected)!=m*n*2 or original["requested_dma"]!=requested:
        raise ValueError("Shape-derived output or DMA traffic differs")
    if initial != inputs+b"\xa5"*256+b"\xcd"*len(expected)+b"\x5a"*256:
        raise ValueError("Encoded inputs/guards/output sentinel differ")
    if memory["output_bytes"] != len(expected) or memory["total_dram_bytes"] != len(initial):
        raise ValueError("Memory extents differ")
    exponent = original["output_rescale_exponent"]
    def evaluate(kind):
        physical=[]
        for row in range(0,m,32):
            for col in range(0,n,32):
                left,right = a[row:row+32],b[:,col:col+32]
                if kind == "omitted_B_transpose":
                    right = np.concatenate([right[inner:inner+32].T for inner in range(0,k,32)])
                if kind == "reset_continuations":
                    left,right=left[:,-32:],right[-32:]
                raw=[[arithmetic.mxu0_dot(left[r],right[:,c]) for c in range(32)] for r in range(32)]
                if recipe == "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1":
                    raw=[[arithmetic.rescale_bf16(value,exponent) for value in values] for values in raw]
                physical.append(arithmetic.output_halves(raw))
        return b"".join(physical)
    rebuilt=evaluate("positive")
    if rebuilt != expected:
        raise ValueError("Independent physical-word arithmetic differs from source expected bytes")
    program=(out / "program.bin").read_bytes()
    if len(program)%4 or len(program)//4 != original["instruction_words"] or program[-4:] != b"\x73\0\0\0":
        raise ValueError("Malformed raw instruction image")
    cases=[{"id":"positive","program":str(out/"program.bin"),"expected":expected.hex(),"actual_oracle":expected.hex()}]
    if controls:
        for name in ("omitted_B_transpose","reset_continuations","aliased_output_tiles","wrong_oracle"):
            if name not in original["executed_negative_controls"]:
                raise ValueError("A requested raw-image control is unavailable")
            control=original["executed_negative_controls"][name]
            directory=source/("control-"+name)
            record=support.identity(directory/"program.bin",control["program_sha256"]);records.append(record)
            destination=out/(name+".bin")
            support.copy_source(record["path"],destination,record["sha256"])
            records.append(support.identity(destination,record["sha256"]))
            source_actual=support.identity(directory/"actual-bf16-tiles-0.bin",control["actual_sha256"])
            records.append(source_actual)
            predicted = (rebuilt[-2048:]+b"\xcd"*(len(expected)-2048) if name=="aliased_output_tiles"
                         else rebuilt if name=="wrong_oracle" else evaluate(name))
            if support.safe_path(source_actual["path"]).read_bytes()!=predicted:
                raise ValueError("Independent mutated arithmetic differs from source model execution")
            wrong = bytes([expected[0]^1])+expected[1:] if name=="wrong_oracle" else expected
            if predicted==wrong:
                raise ValueError("Negative control lacks an output witness")
            cases.append({"id":name,"program":str(destination),"expected":wrong.hex(),"actual_oracle":predicted.hex()})
        halt_path=out/"wrong_halt.bin"
        halt_path.write_bytes(program[:-4]+b"\x73\x00\x10\x00")
        records.append(support.identity(halt_path))
        cases.append({"id":"wrong_halt","program":str(halt_path),"expected":expected.hex(),
                      "actual_oracle":expected.hex(),"mutation":{"offset":len(program)-4,"from":0x73,"to":0x00100073}})
    return {"schema_version":1,"memory":memory,"initial":str(out/"initial-dram.bin"),
            "requested_dma":original["requested_dma"],"cases":cases},records


def replay(data, case, trace_path, actual_path, program_path):
    """Reconstruct output bytes from physical DMA events, independent of summary."""
    import struct
    memory=data["memory"]
    output=bytearray(b"\xcd"*memory["output_bytes"])
    initial=Path(data["initial"]).read_bytes()
    outstanding={};requests=[];acks=[];host=[];states=[];order=[];writes=[];responses=0;status_a=[];status_d=[]
    for line in Path(trace_path).read_text().splitlines():
        event=json.loads(line);kind=event["event"];cycle=event["cycle"]
        if kind=="host_a":host.append(event)
        elif kind=="host_d":acks.append(event)
        elif kind=="status_a":status_a.append(event)
        elif kind=="status_d":status_d.append(event)
        elif kind=="state":states.append(event)
        elif kind=="dma_a":
            address=event["address"];offset=address-memory["dram_base"]
            if address%32 or event["source"] in outstanding or len(event["bytes"])!=32:
                raise AssertionError("Invalid physical DMA request")
            if event["opcode"]==0:
                index=offset-memory["output_offset"]
                if index<0 or index+32>len(output):raise AssertionError("Physical DMA store crossed guards")
                output[index:index+32]=bytes(event["bytes"]);writes.append(index)
            elif event["opcode"]==4:
                if offset<0 or offset+32>memory["input_bytes"]:raise AssertionError("Unmapped physical DMA load")
                if bytes(event["bytes"])!=initial[offset:offset+32]:raise AssertionError("Physical DMA load data differs")
            else:raise AssertionError("Unexpected physical DMA opcode")
            outstanding[event["source"]]=event;requests.append(event);order.append(event["source"])
        elif kind=="dma_d":
            if not order or order.pop(0)!=event["source"]:raise AssertionError("Declared FIFO response policy differs")
            request=outstanding.pop(event["source"])
            responses+=1
            if cycle<request["cycle"]+1 or event["opcode"]!=(1 if request["opcode"]==4 else 0):
                raise AssertionError("Physical DMA source/response timing differs")
        else:raise AssertionError("Unknown trace event")
    words=struct.unpack("<"+"I"*(Path(program_path).stat().st_size//4),Path(program_path).read_bytes())
    if [(entry["csr"],entry["address"],entry["data"]) for entry in host] != [
            (0,0x20000+4*index,value) for index,value in enumerate(words)]+[(1,0x18,1)]:
        raise AssertionError("Observed host writes differ from raw image/start")
    if len(acks)!=len(host) or outstanding:
        raise AssertionError("Outstanding host/DMA transaction at completion")
    wrong_halt=case["id"]=="wrong_halt"
    expected_status=7 if wrong_halt else 5
    if (len(status_a),len(status_d))!=(1,1) or status_a[0]["address"]!=8 or status_d[0]["data"]!=expected_status or status_d[0]["cycle"]<status_a[0]["cycle"]+1:
        raise AssertionError("Physical CSR Get did not prove the required halt status")
    if not any(state["cycle"]>=host[-1]["cycle"] and not state["reset"] and not state["halted"] for state in states):
        raise AssertionError("No observed running transition after start")
    if not wrong_halt:
        if len(states)<32 or any(not state["halted"] or not state["public_idle"] or state["outstanding"] for state in states[-32:]):
            raise AssertionError("Completion quiet margin differs")
        if any(event["cycle"]>=states[-32]["cycle"] for event in requests):
            raise AssertionError("DMA activity during completion margin")
    actual=bytes(output) if wrong_halt else Path(actual_path).read_bytes()
    if actual!=output or actual!=bytes.fromhex(case["actual_oracle"]):
        raise AssertionError("Physical output differs from independently predicted bytes")
    matched=actual==bytes.fromhex(case["expected"])
    if matched != (case["id"] in ("positive","wrong_halt")):
        raise AssertionError("Positive/control outcome differs")
    counts={"load_bytes":sum(32 for event in requests if event["opcode"]==4),
            "store_bytes":sum(32 for event in requests if event["opcode"]==0)}
    if counts!=data["requested_dma"]:raise AssertionError("Physical DMA traffic differs from declared program")
    if responses!=len(requests):raise AssertionError("Physical DMA response count differs")
    if case["id"] in ("positive","wrong_halt") and sorted(writes)!=list(range(0,len(output),32)):
        raise AssertionError("Output beats were not each written exactly once")
    return {"output_bytes":len(actual),"requests":len(requests),"host_words":len(words),
            "expected_equal":matched,"independent_mutant_oracle_equal":True,"public_idle_only":True,
            "csr_status":expected_status,"output_evidence":"physical DMA writes" if wrong_halt else "physical file and DMA writes"}
