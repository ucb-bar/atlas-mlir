"""Check provider-exported issue constraints using Merlin schedule annotations.

The export comes directly from the selected atlas-emit invocation. It is target
evidence, never a candidate-owned timing assertion. Resource, DMA and CFG checks
remain mandatory in that invocation; this module does not replace them.
"""

from hashlib import sha256
import json

from merlin.compile.scheduling.timed_schedule import make_timing_plan, verify_issue_timing_plan
from merlin.sched.ir import Call, Const, Kernel
from merlin.xdsl_dialects._common import text

from .shared_dense import RTL_REVISION


def verify_issue_export(record, program):
    if (record.get("schema") != "atlas.issue_timing.v1" or
            record.get("selected_rtl_revision") != RTL_REVISION or
            record.get("provider") != program.get("timing_provider") or
            record.get("clock") != "block-relative-fixed-latency"):
        raise ValueError("issue timing identity differs from selected program")
    actual = program["instructions"]
    exported = [instruction for block in record["blocks"] for instruction in block["instructions"]]
    if not actual or [(row["word_index"], row["word_u32"]) for row in exported] != [
            (row["word_index"], row["word_u32"]) for row in actual]:
        raise ValueError("timing export differs from actual encoded words")
    results = []
    for number, block in enumerate(record["blocks"]):
        instructions = block["instructions"]
        count = len(instructions)
        if not count or block["covered_pairs"] != count * (count-1) // 2:
            raise ValueError("incomplete pair timing coverage")
        edges = {}
        for earlier, later, minimum in block["distances"]:
            if (any(type(v) is not int for v in (earlier, later, minimum)) or
                    not 0 <= earlier < later < count or minimum <= 0 or (earlier, later) in edges):
                raise ValueError("invalid or repeated timing edge")
            edges[earlier, later] = minimum
        calls = tuple(Call("machine.word", (("index", Const(row["word_index"])), ("bits", Const(row["word_u32"]))))
                      for row in instructions)
        kernel = Kernel(f"block{number}", (), calls)
        ordinals = {call: index for index, call in enumerate(calls)}
        starts = tuple(row["issue"] for row in instructions)
        ages = tuple(row["complete_age"] for row in instructions)
        gaps = tuple(row["issue_gap"] for row in instructions)
        if any(type(value) is not int or value <= 0 for value in (*ages, *gaps)):
            raise ValueError("unknown completion or issue gap")
        def completion(call):
            return ages[ordinals[call]]
        def distance(first, second):
            i, j = ordinals[first], ordinals[second]
            return max(edges.get((i, j), 0), gaps[i] if j == i+1 else 0)
        overlaps = tuple((i, i+1, "require" if starts[i]+ages[i] > starts[i+1] else "forbid")
                         for i in range(count-1))
        plan = make_timing_plan(kernel, starts, model_identity=record["provider"], clock=record["clock"],
                                overlaps=overlaps)
        verify_issue_timing_plan(kernel, plan, model_identity=record["provider"], clock=record["clock"],
                                 completion=completion, distance=distance, max_calls=8192)
        results.append((kernel, plan))
    return results


def write_issue_checks(record, program, destination):
    checked = verify_issue_export(record, program)
    payload = json.dumps(record, sort_keys=True, indent=2) + "\n"
    (destination / "issue-timing.json").write_text(payload)
    for index, (kernel, plan) in enumerate(checked):
        (destination / f"timing-block{index}.mk").write_text(kernel.text())
        (destination / f"timing-block{index}.mlir").write_text(text(plan, generic=True))
    return dict(scope=record["scope"], provider=record["provider"], blocks=len(checked),
                instructions=sum(len(kernel.body) for kernel, _ in checked),
                export_sha256=sha256(payload.encode()).hexdigest())
