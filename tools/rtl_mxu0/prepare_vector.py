"""Bind P2 pack/rescale cases to a completed fresh MXU component execution."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def prepare(helper, mxu_root, expected_receipt_sha):
    root = helper.safe_path(mxu_root)
    records = [helper.identity(root / "receipt.json", expected_receipt_sha)]
    receipt = json.loads(helper.safe_path(root / "receipt.json").read_text())
    if receipt["status"] != "passed_component_rtl" or receipt["output_vpu_rescale_executed"]:
        raise ValueError("Expected successful PRE-VPU MXU component evidence")
    for key in ("trace", "input", "case_manifest"):
        value = receipt[key]
        records.append(helper.identity(value["path"], value["sha256"]))
    traces = json.loads(helper.safe_path(receipt["trace"]["path"]).read_text())
    mxu_inputs = json.loads(helper.safe_path(receipt["input"]["path"]).read_text())["cases"]
    tasks = []
    cases = []
    for case in receipt["cases"]:
        if case["recipe"] != "atlas_mxu0_fp8_tensor_pow2_bf16acc_v1":
            continue
        for value in case["source_records"]:
            records.append(helper.identity(value["path"], value["sha256"]))
        source = helper.safe_path(case["source_records"][0]["path"]).parent
        original = json.loads(helper.safe_path(source / "receipt.json").read_text())
        native_manifest = json.loads(helper.safe_path(source / "source-manifest.json").read_text())
        source_dtype = native_manifest["source_dtype"]
        if source_dtype not in ("float32", "bfloat16"):
            raise ValueError("Unsupported declared native source dtype")
        records.append(helper.identity(source / "source.npz", original["artifacts"]["source.npz"]))
        oracle = next(value for value in case["source_records"]
                      if value["sha256"] == original["oracle_sha256"])
        spec = importlib.util.spec_from_file_location("vector_arithmetic_" + case["id"], oracle["path"])
        arithmetic = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(arithmetic)
        encoded = next(value for value in mxu_inputs if value["id"] == case["id"])
        with np.load(helper.safe_path(source / "source.npz"), allow_pickle=False) as bundle:
            for operand in ("A", "B"):
                native = bundle[operand]
                if native.dtype != np.float32 or not np.all(np.isfinite(native)):
                    raise ValueError("Expected declared finite native float32 operand views")
                bf16 = np.array([arithmetic.f32_to_bf16_bits(float(value)) for value in native.flat],
                                dtype=np.uint16).reshape(native.shape)
                if source_dtype == "bfloat16":
                    if np.any(native.view(np.uint32) & 65535) or not np.array_equal(bf16, (native.view(np.uint32) >> 16).astype(np.uint16)):
                        raise AssertionError("Native BF16 words were not preserved exactly")
                scale = original["input_scale_codes"][operand]
                packed = np.array([arithmetic.pack_bf16(int(value), scale) for value in bf16.flat],
                                  dtype=np.uint8).reshape(native.shape)
                if not np.array_equal(packed, np.asarray(encoded[operand], dtype=np.uint8)):
                    raise AssertionError("Native BF16 pack oracle differs from actual encoded MXU input")
                # Each MREG tile is32 rows of32 columns. B is loaded transposed.
                for tile, offset in enumerate(range(0, case["shape"][1], 32)):
                    words = bf16[:, offset:offset + 32] if operand == "A" else bf16[offset:offset + 32].T
                    expected = packed[:, offset:offset + 32] if operand == "A" else packed[offset:offset + 32].T
                    if words.shape != (32, 32):
                        raise ValueError("Malformed native operand tile")
                    tasks.append({"id": f"{case['id']}-{operand.lower()}-{tile}", "case_id": case["id"],
                                  "kind": "pack", "source_bf16": words.tolist(),
                                  "expected_fp8": expected.tolist(), "scale_code": scale})
        executions = [value for value in traces if value["case_id"] == case["id"] and value["run_kind"] == "normal"]
        if len(executions) != 2:
            raise ValueError("Expected two actual raw MXU executions")
        raw = None
        for execution in executions:
            physical = np.zeros((32, 32), dtype=np.uint16)
            seen = set()
            for event in execution["events"]:
                for write in event["writes"]:
                    bank, row = write["mreg"], write["row"]
                    if (bank, row) in seen or bank not in (2, 3) or not 0 <= row < 32:
                        raise AssertionError("Invalid physical MXU output extent")
                    seen.add((bank, row))
                    data = int(write["data_hex"], 16)
                    for lane in range(16):
                        physical[row, (bank - 2) * 16 + lane] = (data >> (16 * lane)) & 65535
            if len(seen) != 64 or not np.array_equal(physical, np.asarray(encoded["expected_bf16"], dtype=np.uint16)):
                raise AssertionError("Physical raw MXU words differ from authenticated oracle")
            if raw is not None and not np.array_equal(raw, physical):
                raise AssertionError("Raw MXU repeats differ")
            raw = physical
        exponent = original["output_rescale_exponent"]
        if not -126 <= exponent <= 127:
            raise ValueError("Rescale constant must be an exactly representable finite BF16 power of two")
        constant = (exponent + 127) << 7
        if arithmetic.bf16_value(constant) != arithmetic.power2(exponent):
            raise AssertionError("Rescale constant does not represent declared power of two")
        corrected = np.array([[arithmetic.rescale_bf16(int(value), exponent) for value in row] for row in raw], dtype=np.uint16)
        with np.load(helper.safe_path(source / "result.npz"), allow_pickle=False) as bundle:
            target = bundle["target_C"]
            if not np.array_equal(corrected, bundle["expected_bits"]):
                raise AssertionError("Final rescale oracle differs from saved P2 oracle")
            if target.dtype != np.float32 or np.any(target.view(np.uint32) & 65535):
                raise ValueError("Saved P2 model outputs are not exact BF16 views")
            if not np.array_equal(corrected, (target.view(np.uint32) >> 16).astype(np.uint16)):
                raise AssertionError("Final rescale oracle differs from saved P2 model output")
        tasks.append({"id": case["id"] + "-rescale", "case_id": case["id"], "kind": "rescale",
                      "source_bf16": raw.tolist(), "expected_bf16": corrected.tolist(),
                      "constant_bf16": constant, "scale_exponent": exponent})
        cases.append({"id": case["id"], "shape": case["shape"],
                      "scale_codes": original["input_scale_codes"], "rescale_exponent": exponent,
                      "source_dtype": source_dtype,
                      "input_conversion": ("exact native BF16 words from float32 views" if source_dtype == "bfloat16"
                                           else "declared float32 to BF16 round-to-nearest-even policy"),
                      "raw_input_origin": "physical write events from two initialized fresh MXU RTL runs"})
    if not cases:
        raise ValueError("No P2 cases available")
    for value in records:
        helper.identity(value["path"], value["sha256"])
    return {"schema_version": 1, "tasks": tasks}, {"cases": cases, "source_records": records}


def initial_source_rows(task):
    """Pack sees a contiguous BF16 stream; rescale sees MXU column halves."""
    pack = task["kind"] == "pack"
    result = {}
    for row in range(32):
        for half in range(2):
            values = []
            for lane in range(16):
                linear = half * 512 + row * 16 + lane
                item = task["source_bf16"][linear // 32][linear % 32] if pack else task["source_bf16"][row][half * 16 + lane]
                values.append(int(item))
            result[(0 if pack else 2) + half, row] = sum(value << (16 * lane) for lane, value in enumerate(values))
    return result


def validate(data, traces):
    """Independently replay physical MREG events, including in-place read/write order."""
    tasks = {task["id"]: (index, task) for index, task in enumerate(data["tasks"])}
    expected_runs = {(name, repeat) for name in tasks for repeat in range(2)}
    if len(tasks) != len(data["tasks"]) or len(traces) != len(expected_runs):
        raise AssertionError("Missing or duplicate VPU execution")
    if {(trace["task_id"], trace["repeat"]) for trace in traces} != expected_runs:
        raise AssertionError("Unexpected VPU run identities")
    for trace in traces:
        index, task = tasks[trace["task_id"]]
        pack = task["kind"] == "pack"
        bits = 8 if pack else 16
        pattern = index * 2 + trace["repeat"]
        if trace["initial_pattern"] != pattern or trace["word_bits"] != bits:
            raise AssertionError("Incorrect initialized pattern or output width")
        source_bank = 0 if pack else 2
        memory = initial_source_rows(task)
        def initial(bank, row):
            return sum(((bank * 7 + row * 3 + lane + 0x51 + pattern * 29) & 255) << (8 * lane)
                       for lane in range(32))
        written = set()
        for cycle, event in enumerate(trace["events"], 1):
            if event["cycle"] != cycle:
                raise AssertionError("Missing or duplicated sample cycle")
            if cycle > trace["quiet_after_cycle"] and (event["reads"] or event["writes"]):
                raise AssertionError("Late physical VPU event")
            for read in event["reads"]:
                bank, row = read["mreg"], read["row"]
                if bank not in ({0, 1} if pack else {2, 3, 4, 5}) or not 0 <= row < 32:
                    raise AssertionError("Unexpected physical input extent")
                if int(read["data_hex"], 16) != memory.get((bank, row), initial(bank, row)):
                    raise AssertionError("Physical VPU input differs from declared native/raw operands")
            for write in event["writes"]:
                bank, row = write["mreg"], write["row"]
                if bank not in ({2} if pack else {2, 3, 4, 5}) or not 0 <= row < 32 or (bank, row) in written:
                    raise AssertionError("Unexpected or duplicated physical output extent")
                written.add((bank, row))
                value = int(write["data_hex"], 16)
                if not 0 <= value < (1 << 256):
                    raise AssertionError("Physical row is wider than MREG")
                memory[bank, row] = value
        expected_extent = {(bank, row) for bank in ({2} if pack else {2, 3, 4, 5}) for row in range(32)}
        if written != expected_extent or len(trace["events"]) != trace["sample_cycles"]:
            raise AssertionError("Physical output or sampling extent differs")
        expected_commands = ([{"cycle": 1, "op": 17, "destination": 2}] if pack else
                             [{"cycle": 1, "op": 30, "destination": 4},
                              {"cycle": 100, "op": 3, "destination": 2}])
        for command in expected_commands:
            command.update(vs1=source_bank, vs2=0 if pack else 4,
                           scale_code=task["scale_code"] if pack else 127,
                           immediate=0 if pack else task["constant_bf16"])
        if trace["commands"] != expected_commands:
            raise AssertionError("Actual commands differ from declared recipe")
        if trace["reset_cycles"] != 2:
            raise AssertionError("Incorrect reset duration")
        if not pack:
            constant = sum(int(task["constant_bf16"]) << (16 * lane) for lane in range(16))
            if any(memory[bank, row] != constant for bank in (4, 5) for row in range(32)):
                raise AssertionError("Actual VLI_ALL constant pair differs")
        mask = (1 << bits) - 1
        actual = [[(memory[(2 if pack else 2 + column // 16), row] >> (bits * (column if pack else column % 16))) & mask
                   for column in range(32)] for row in range(32)]
        expected = task["expected_fp8" if pack else "expected_bf16"]
        if actual != expected or actual != trace["actual_words"] or trace["mismatches"]:
            raise AssertionError("Physical VPU output differs from independent oracle")
        if not all(trace[key] for key in ("extent_exact", "constant_exact", "idle_at_end", "quiet_after_margin")):
            raise AssertionError("VPU completion or guard gate failed")
    return {"repeats_per_task": 2, "pack_tasks": sum(task["kind"] == "pack" for _, task in tasks.values()),
            "rescale_tasks": sum(task["kind"] == "rescale" for _, task in tasks.values()),
            "physical_input_and_output_events_replayed": True, "output_vpu_rescale_executed": True}
