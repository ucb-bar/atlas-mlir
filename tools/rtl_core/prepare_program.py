"""Authenticate raw images and rebuild arithmetic or explicitly reuse bound evidence."""
import importlib.util
import json
from pathlib import Path
import numpy as np


RESIDENT_B_EPOCH = (
    "2a0e480bf6b419c977486ca7f10fb696dabb0b18cc1739d8d077e5a69aeac8ce",
    "fc76d7ce7e16aeea90f1c84fc0c250b0489d921fd54b3f3a04c46112961e410d",
    "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0",
)
RESIDENT_A_EPOCH = (
    "69b3acd8ed2516b8cfa7a7ab0d2973dd6b4c306fdb7bdfe0c8449507654197c2",
    "fc76d7ce7e16aeea90f1c84fc0c250b0489d921fd54b3f3a04c46112961e410d",
    "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0",
)
PRETRANSPOSED_B_EPOCH = (
    "b5f417a1e847bd1be6133ccd4b9e89b5e4b5947a7ff1b694e7ca09ae576a6792",
    "fc76d7ce7e16aeea90f1c84fc0c250b0489d921fd54b3f3a04c46112961e410d",
    "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0",
)
# The release adds source admission; executed historical receipts keep their own epochs.
RELEASE_EPOCH = (
    "3589b96dac743fd1f607af9960b606adb8c0c15462a786422ea063091f5b25dd",
    "ef6d7060e8372062eaa393501ed83d7908ede7f0bd8a8820bcd8f31fc3593498",
    "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0",
)
ARITHMETIC_EPOCHS = {
    RESIDENT_B_EPOCH, RESIDENT_A_EPOCH, PRETRANSPOSED_B_EPOCH, RELEASE_EPOCH,
    ("06c415e57e331c0260f8072056cab9ca7048510228130aec8c6cd004203d092b",
     "298930b20b85d0d9f1612969c032319396a8cce5e65ebf8ba64b9754f6dd3410",
     "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0"),
    ("6f347e50c5a8bab0ea735015376d279a391de6e59b5c0e89640c90a9da2427f0",
     "fc76d7ce7e16aeea90f1c84fc0c250b0489d921fd54b3f3a04c46112961e410d",
     "7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0"),
}
PRODUCER_NAMES = ("run_model_program_contraction.py", "run_model_tile.py", "atlas_tile_arithmetic.py")
MODEL_PIN = "9e6313bddecb8ba936206eb347f5cd4e5335282c"


def producer_epoch(original):
    before, after = original.get("code_before"), original.get("code_after")
    if not isinstance(before, dict) or before != after or set(before) != set(PRODUCER_NAMES):
        raise ValueError("Complete-model producer code bookends differ")
    epoch = tuple(before[name] for name in PRODUCER_NAMES)
    if epoch not in ARITHMETIC_EPOCHS:
        raise ValueError("Unknown or mixed complete-model producer epoch")
    return epoch


def requested_traffic(original, m, k, n):
    """Admit one fixed producer mode and derive its complete-program DMA bytes."""
    epoch = producer_epoch(original)
    if any(type(value) is not int or value <= 0 or value % 32 for value in (m, k, n)):
        raise ValueError("Only complete physical tiles are admitted")
    command = original.get("command", [])
    flags = ("--resident-a", "--resident-b", "--pretranspose-resident-b")
    if (not isinstance(command, list) or not all(isinstance(item, str) for item in command)
            or any(command.count(flag) > 1 for flag in flags)
            or any(item.startswith(flags) and item not in flags for item in command)):
        raise ValueError("Ambiguous resident operand command flags")
    a, b, pretranspose = (flag in command for flag in flags)
    if a and b or pretranspose and not b:
        raise ValueError("Conflicting resident operand flags")
    if ((epoch == RESIDENT_A_EPOCH and (a, b, pretranspose) != (True, False, False))
            or (epoch == RESIDENT_B_EPOCH and (a, b, pretranspose) != (False, True, False))
            or (epoch == PRETRANSPOSED_B_EPOCH and (a, b, pretranspose) != (False, True, True))):
        raise ValueError("Historical optimized epoch requires its qualified mode")
    marker = original.get("operand_reuse")
    claim = original.get("resident_operand_reuse_claim")
    if epoch not in (RESIDENT_A_EPOCH, RESIDENT_B_EPOCH, PRETRANSPOSED_B_EPOCH, RELEASE_EPOCH):
        if a or b or pretranspose or marker is not None or claim is not None and claim is not False:
            raise ValueError("Baseline producer cannot claim operand reuse")
        loads = 2 * m * k * (n // 32)
    elif a or b:
        operand, size = ("a", m * k) if a else ("b", k * n)
        mode = "resident_a_v1" if a else "resident_b_pretransposed_v1" if pretranspose else "resident_b_v1"
        size_key, base_key = "resident_" + operand + "_bytes", "resident_" + operand + "_vmem_byte_base"
        if (claim is not True or marker != {"mode": mode, size_key: size, base_key: 0x40000}
                or type(marker.get(size_key)) is not int or type(marker.get(base_key)) is not int
                or size > 256 * 1024):
            raise ValueError("Resident operand metadata or bank capacity differs")
        loads = m * k + k * n * (m // 32) if a else m * k * (n // 32) + k * n
    else:
        if (epoch != RELEASE_EPOCH or claim is not False
                or marker != {"mode": "none", "resident_b_bytes": 0, "resident_b_vmem_byte_base": None}
                or type(marker.get("resident_b_bytes")) is not int):
            raise ValueError("Release default operand metadata differs")
        loads = 2 * m * k * (n // 32)
    return {"load_bytes": loads, "store_bytes": 2 * m * n}


def model_payload_path(support, source, artifacts):
    names = ("actual-bf16-tiles-0.bin", "actual-bf16-halves-0.bin")
    present = [support.safe_path(source / name) for name in names if support.safe_path(source / name).exists()]
    declared = [name for name in names if name in artifacts]
    if len(present) != 1 or declared != [present[0].name]:
        raise ValueError("Require exactly one authenticated known complete-model actual payload filename")
    return present[0]


def control_payload_path(support, source):
    names = ("actual-bf16-tiles-0.bin", "actual-bf16-halves-0.bin")
    present = [support.safe_path(source / name) for name in names if support.safe_path(source / name).exists()]
    if len(present) != 1:
        raise ValueError("Require exactly one known physical control payload filename")
    return present[0]


def reused_arithmetic(support, source, original, expected, model_root):
    """Admit retained complete-model arithmetic evidence without rebuilding dots."""
    import subprocess
    if model_root is None:
        raise ValueError("Arithmetic reuse requires the explicit pinned model root")
    epoch = producer_epoch(original)
    model_root = support.safe_path(model_root)
    bookends = original.get("source_bookends", {})
    before = original.get("model_binding_before")
    inputs = original.get("source_inputs")
    if (original.get("invocations") != "one_reset_core_per_repeat_all_tiles_in_one_instruction_program"
            or bookends.get("unchanged") is not True or not isinstance(before, dict)
            or before != bookends.get("model_after") or before.get("revision") != MODEL_PIN
            or before.get("tracked_source_diff_exit") != 0 or not isinstance(inputs, dict)
            or set(inputs) != {"input", "manifest"} or inputs != bookends.get("input_before")
            or inputs != bookends.get("input_after")):
        raise ValueError("Complete-model source/input/model bookends are not unchanged")
    model = original.get("model", {})
    imported = before.get("imported_source_sha256")
    if (model.get("revision") != MODEL_PIN
            or not isinstance(imported, dict) or not imported
            or model.get("backend") != "curve" or model.get("numerics") != "rtl"
            or original.get("model_input") != "new_assembler_decoded_instruction_objects"
            or model.get("module_file") != str(model_root / "npu_model/__init__.py")
            or model.get("imported_source_sha256") != before.get("imported_source_sha256")
            or model.get("curve_sha256") != before.get("curve_sha256")):
        raise ValueError("Complete-model module/curve/source binding differs")
    if subprocess.check_output(["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True).strip() != MODEL_PIN:
        raise ValueError("Explicit model root revision differs")
    records = [support.identity(source / "producer-sources" / name, sha) for name, sha in zip(PRODUCER_NAMES, epoch)]
    for relative, sha in before["imported_source_sha256"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unadmitted relative model source path")
        records.append(support.identity(model_root / relative, sha))
    records.append(support.identity(model_root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json", before["curve_sha256"]))
    command = original.get("command", [])
    for name, sha in inputs.items():
        flag = "--" + name
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError("Ambiguous complete-model native input path")
        records.append(support.identity(command[command.index(flag) + 1], sha))
    runs = original.get("runs", [])
    if not runs or any(run.get("all_output_bytes_exact") is not True
                       or run.get("input_and_guards_preserved") is not True
                       or run.get("drained_before_halt_observation") is not True
                       or run.get("halt_reason") != "ecall" for run in runs):
        raise ValueError("Complete-model execution did not qualify its actual output")
    actual = model_payload_path(support, source, original["artifacts"])
    record = support.identity(actual, original["artifacts"][actual.name])
    records.append(record)
    if actual.read_bytes() != expected:
        raise ValueError("Authenticated complete-model actual payload differs from its expected output")
    return records, {"mode": "authenticated_complete_model_payload_reuse", "fresh_arithmetic_rebuild": False,
                     "actual": record, "producer_epoch": dict(zip(PRODUCER_NAMES, epoch)), "controls_enabled": False}


def prepare(support, source, receipt_sha, oracle_path, out, controls, *, reuse_model_arithmetic=False, model_root=None):
    if reuse_model_arithmetic and controls:
        raise ValueError("Arithmetic reuse cannot be combined with controls")
    if reuse_model_arithmetic and model_root is None:
        raise ValueError("Arithmetic reuse requires the explicit pinned model root")
    source, out = support.safe_path(source), support.safe_path(out)
    records = [support.identity(source / "receipt.json", receipt_sha)]
    original = json.loads((source / "receipt.json").read_text())
    if original["status"] != "passed_model_only":
        raise ValueError("Source model program did not pass")
    recipe = original["recipe"]
    if recipe not in ("atlas_mxu0_fp8_unit_bf16acc_v1", "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1"):
        raise ValueError("Unsupported source recipe")
    m,k,n = original["shape"]
    requested = requested_traffic(original, m, k, n)
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
    arithmetic_evidence={"mode":"fresh_fraction_rebuild","fresh_arithmetic_rebuild":True}
    if reuse_model_arithmetic:
        reused,arithmetic_evidence=reused_arithmetic(support,source,original,expected,model_root)
        records.extend(reused)
        arithmetic_evidence["source_receipt"]=records[0]
        rebuilt=expected
    else:
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
            source_actual=support.identity(control_payload_path(support,directory),control["actual_sha256"])
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
            "requested_dma":original["requested_dma"],"cases":cases,"arithmetic_evidence":arithmetic_evidence},records


def replay(data, case, trace_path, actual_path, program_path):
    """Reconstruct output bytes from physical DMA events, independent of summary."""
    import struct
    from collections import deque
    memory=data["memory"]
    output=bytearray(b"\xcd"*memory["output_bytes"])
    initial=Path(data["initial"]).read_bytes()
    outstanding={};acks=0;host=[];states=deque(maxlen=32);order=deque();writes=[];responses=0;status_a=[];status_d=[]
    request_count=0;counts={"load_bytes":0,"store_bytes":0};last_running_cycle=None;last_request_cycle=None
    with Path(trace_path).open() as trace:
        for line in trace:
            event=json.loads(line);kind=event["event"];cycle=event["cycle"]
            if kind=="host_a":host.append(event)
            elif kind=="host_d":acks+=1
            elif kind=="status_a":status_a.append(event)
            elif kind=="status_d":status_d.append(event)
            elif kind=="state":
                states.append(event)
                if not event["reset"] and not event["halted"]:
                    last_running_cycle=cycle if last_running_cycle is None else max(last_running_cycle,cycle)
            elif kind=="dma_a":
                address=event["address"];offset=address-memory["dram_base"]
                if address%32 or event["source"] in outstanding or len(event["bytes"])!=32:
                    raise AssertionError("Invalid physical DMA request")
                if event["opcode"]==0:
                    index=offset-memory["output_offset"]
                    if index<0 or index+32>len(output):raise AssertionError("Physical DMA store crossed guards")
                    output[index:index+32]=bytes(event["bytes"]);writes.append(index);counts["store_bytes"]+=32
                elif event["opcode"]==4:
                    if offset<0 or offset+32>memory["input_bytes"]:raise AssertionError("Unmapped physical DMA load")
                    if bytes(event["bytes"])!=initial[offset:offset+32]:raise AssertionError("Physical DMA load data differs")
                    counts["load_bytes"]+=32
                else:raise AssertionError("Unexpected physical DMA opcode")
                outstanding[event["source"]]=event;request_count+=1;order.append(event["source"])
                last_request_cycle=cycle if last_request_cycle is None else max(last_request_cycle,cycle)
            elif kind=="dma_d":
                if not order or order.popleft()!=event["source"]:raise AssertionError("Declared FIFO response policy differs")
                request=outstanding.pop(event["source"])
                responses+=1
                if cycle<request["cycle"]+1 or event["opcode"]!=(1 if request["opcode"]==4 else 0):
                    raise AssertionError("Physical DMA source/response timing differs")
            else:raise AssertionError("Unknown trace event")
    words=struct.unpack("<"+"I"*(Path(program_path).stat().st_size//4),Path(program_path).read_bytes())
    if [(entry["csr"],entry["address"],entry["data"]) for entry in host] != [
            (0,0x20000+4*index,value) for index,value in enumerate(words)]+[(1,0x18,1)]:
        raise AssertionError("Observed host writes differ from raw image/start")
    if acks!=len(host) or outstanding:
        raise AssertionError("Outstanding host/DMA transaction at completion")
    wrong_halt=case["id"]=="wrong_halt"
    expected_status=7 if wrong_halt else 5
    if (len(status_a),len(status_d))!=(1,1) or status_a[0]["address"]!=8 or status_d[0]["data"]!=expected_status or status_d[0]["cycle"]<status_a[0]["cycle"]+1:
        raise AssertionError("Physical CSR Get did not prove the required halt status")
    if last_running_cycle is None or last_running_cycle<host[-1]["cycle"]:
        raise AssertionError("No observed running transition after start")
    if not wrong_halt:
        if len(states)<32 or any(not state["halted"] or not state["public_idle"] or state["outstanding"] for state in states):
            raise AssertionError("Completion quiet margin differs")
        if last_request_cycle is not None and last_request_cycle>=states[0]["cycle"]:
            raise AssertionError("DMA activity during completion margin")
    actual=bytes(output) if wrong_halt else Path(actual_path).read_bytes()
    if actual!=output or actual!=bytes.fromhex(case["actual_oracle"]):
        raise AssertionError("Physical output differs from independently predicted bytes")
    matched=actual==bytes.fromhex(case["expected"])
    if matched != (case["id"] in ("positive","wrong_halt")):
        raise AssertionError("Positive/control outcome differs")
    if counts!=data["requested_dma"]:raise AssertionError("Physical DMA traffic differs from declared program")
    if responses!=request_count:raise AssertionError("Physical DMA response count differs")
    if case["id"] in ("positive","wrong_halt") and sorted(writes)!=list(range(0,len(output),32)):
        raise AssertionError("Output beats were not each written exactly once")
    return {"output_bytes":len(actual),"requests":request_count,"host_words":len(words),
            "expected_equal":matched,"independent_mutant_oracle_equal":True,"public_idle_only":True,
            "csr_status":expected_status,"output_evidence":"physical DMA writes" if wrong_halt else "physical file and DMA writes"}
