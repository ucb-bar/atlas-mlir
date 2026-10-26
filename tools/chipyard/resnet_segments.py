#!/usr/bin/env python3
"""Split the generated ResNet graph into per-launch segments and build one SoC ELF per segment.

Segment k runs graph statements from the pack feeding launch k up to the pack feeding launch k+1 (segment 0 also
copies the network input; segment 53 also adds the FC bias and writes the logits). Every host statement and launch is
covered exactly once. Each segment ELF links the unchanged runtime and the exact statement text of graph.c; LOADMEM
places its live-in tensors and the captured DMA arena, and the ELF checks its live-out tensors and the Atlas output.
"""

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).absolute().parent
sys.path.insert(0, str(TOOLS.parent))
from run_sim import safe_path as admitted_path
from atlas_tooling import sha, write_json
from atlas_host_build import validate_graph_program

RUNTIME = TOOLS.parents[1] / "runtime"
SOURCES = ("atlas_host.c", "atlas_resnet_graph_math.c", "atlas_resnet_graph_device.c")
FLAGS = ["--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-std=c11", "-O2", "-Wall",
         "-Wextra", "-Werror", "-ffp-contract=off", "-fno-fast-math", "-ffreestanding", "-fno-builtin",
         "-fno-stack-protector", "-fno-pic", "-fno-vectorize", "-fno-slp-vectorize", "-nostdlib"]
# Argument positions written by each graph operator; every other buffer argument is read.
WRITES = {"pack": (7,), "device": (5,), "decode": (2,), "affine": (1,), "relu": (1,), "add": (2,), "maxpool": (1,),
          "average": (1,)}
BUFFER = re.compile(r"\b(tensor_\d+|packed_a|physical_output|input|output|params_\d+)\b")
LINK = """OUTPUT_ARCH(riscv)
ENTRY(_start)
PHDRS { text PT_LOAD FLAGS(5); rodata PT_LOAD FLAGS(4); data PT_LOAD FLAGS(6); dram PT_LOAD FLAGS(6); }
SECTIONS {
  . = 0x80000000;
  .text : { KEEP(*(.text.init)) *(.text .text.*) } :text
  . = ALIGN(64);
  .rodata : { *(.rodata .rodata.*) *(.seg_expected) } :rodata
  . = ALIGN(64);
  .data : { *(.data .data.* .sdata .sdata.*) *(.seg_state) } :data
  . = ALIGN(64);
  .tohost : { KEEP(*(.tohost)) } :data
  . = ALIGN(64);
  __global_pointer$ = . + 0x800;
  .seg_scratch (NOLOAD) : { *(.bss.seg_scratch) } :data
  . = ALIGN(64);
  .bss (NOLOAD) : {
    __bss_start = .;
    *(.bss .bss.* .sbss .sbss.* COMMON)
    . = ALIGN(64);
    __bss_end = .;
  } :data
  .stack (NOLOAD) : { __stack_bottom = .; . += 65536; __stack_top = .; } :data
  ASSERT(__stack_top < 0x90000000, "host exceeds bounded low DRAM region")
  .atlas_dram 0x100000000 : { __atlas_dram_start = .; KEEP(*(.atlas_dram)) __atlas_dram_end = .; } :dram
  ASSERT(__atlas_dram_end <= 0x200000000, "DMA low32 window overflow")
}
"""


def read(path):
    return admitted_path(path).read_bytes()


def context_fields(raw):
    if len(raw) != 976:
        raise ValueError("device context extent")
    completed, failed, failing, result, runtime = struct.unpack_from("<5I", raw, 88)
    return dict(completed=completed, failed=failed, failing_index=failing, result=result, runtime_result=runtime,
                activation_scale=list(struct.unpack_from("<54i", raw, 108)),
                rescale_exponent=list(struct.unpack_from("<54i", raw, 324)),
                sequence=list(struct.unpack_from("<54I", raw, 540)), cycles=list(struct.unpack_from("<54I", raw, 756)))


def host_identity(host):
    build_raw = read(host / "build.json")
    build = json.loads(build_raw)
    if (build.get("returncode") != 0 or build.get("source_bookends_equal") is not True
        or build.get("kernel_count") != 54 or sha(read(host / "host.elf")) != build.get("elf_sha256")):
        raise ValueError("full-host build identity differs")
    digests = set(build.get("inputs_sha256", {}).values()) | set(build.get("generated_inputs_sha256", {}).values())
    return build, digests


def bound_host_file(path, digests):
    raw = read(path)
    if sha(raw) not in digests:
        raise ValueError("host input is not bound to the full-host build: " + str(path))
    return raw


def capture_manifest(host, capture):
    build, digests = host_identity(host)
    raw = read(capture / "capture.json")
    manifest = json.loads(raw)
    if (manifest.get("status") != "PASS_CAPTURE" or manifest.get("elf_sha256") != build["elf_sha256"]
        or len(manifest.get("boundaries", [])) != 55 or len(manifest.get("kernels", [])) != 54):
        raise ValueError("capture is not a complete matching full-host capture")
    for k, boundary in enumerate(manifest["boundaries"]):
        fields = context_fields(read(capture / f"boundary-{k:02d}/context.bin"))
        if (boundary.get("boundary") != k or boundary.get("context") != fields or fields["completed"] != k
            or fields["failed"] or fields["sequence"][:k] != list(range(1, k + 1))):
            raise ValueError("capture boundary context differs")
    return manifest, sha(raw), digests


def boundary_region(capture, manifest, boundary, name, size=None):
    raw = read(capture / f"boundary-{boundary:02d}/{name}.bin")
    if sha(raw) != manifest["boundaries"][boundary]["regions"].get(name) or (size is not None and len(raw) != size):
        raise ValueError(f"capture boundary {boundary} region {name} differs")
    return raw


def segment_inputs(host, capture, manifest, p, seg, digests):
    k = seg["index"]
    state = {b: boundary_region(capture, manifest, k, b, p["sizes"][b]) for b in seg["live_in"]}
    expected = {b: boundary_region(capture, manifest, k + 1, b, p["sizes"][b])
                for b in seg["live_out"] if b != "output"}
    arena = boundary_region(capture, manifest, k, "arena")
    end = boundary_region(capture, manifest, k + 1, "arena", len(arena))
    kernel = manifest["kernels"][k]
    offset, size = kernel["output_offset"], seg["out_bytes"]
    if (kernel["index"] != k or not 0 <= offset < offset + size <= kernel["dram_bytes"] <= len(arena)):
        raise ValueError("captured DMA extent differs")
    expected["atlas_output"] = end[offset:offset + size]
    if sha(expected["atlas_output"]) != kernel["expected_sha256"]:
        raise ValueError("captured Atlas output differs from model payload")
    if "output" in seg["live_out"]:
        expected["output"] = boundary_region(capture, manifest, k + 1, "output", 4000)
        expected["reference_logits"] = bound_host_file(host / "graph_expected.bin", digests)
        if expected["output"] != expected["reference_logits"]:
            raise ValueError("captured logits differ from full-host reference")
    return state, expected, arena


def verify_build(out, p, seg):
    receipt = json.loads(read(out / "build.json"))
    statements = [op["text"] for op in p["ops"][seg["first"]:seg["end"]]]
    if (receipt.get("segment") != seg["index"] or receipt.get("statements") != statements
        or receipt.get("live_in") != seg["live_in"] or receipt.get("live_out") != seg["live_out"]
        or receipt.get("producers") != seg.get("producers", {})
        or sha(read(out / "host.elf")) != receipt.get("elf_sha256")
        or read(out / "segment.c").decode() != segment_source(p, seg)):
        raise ValueError("segment build identity differs")
    expected_regions = set(seg["live_out"]) | {"atlas_output"}
    if "output" in expected_regions:
        expected_regions.add("reference_logits")
    if (set(receipt.get("start_sha256", {})) != set(seg["live_in"]) | {"arena"}
        or set(receipt.get("expected_sha256", {})) != expected_regions):
        raise ValueError("segment region inventory differs")
    for key in ("start_sha256", "expected_sha256"):
        for name, digest in receipt[key].items():
            if not re.fullmatch(r"(?:tensor_\d+|arena|atlas_output|output|reference_logits)", name):
                raise ValueError("unknown segment region")
            if sha(read(out / f"{name}.bin")) != digest:
                raise ValueError("segment region differs: " + name)
    return receipt


def statement(line):
    """Return (kind, launch, reads, writes) for one graph statement line."""
    loop = re.fullmatch(r"for \(size_t j=0;j<\d+;\+\+j\) (\w+)\[j\]=(.+);", line)
    if loop:
        return "copy", None, [b for b in BUFFER.findall(loop[2])], [loop[1]]
    call = re.fullmatch(r"if \((?:atlas_graph_)?(\w+)\((.*)\)\) return \d+;", line)
    if not call or call[1] not in WRITES:
        raise ValueError("unrecognized graph statement: " + line)
    args = [a.strip() for a in call[2].split(",")]
    writes = [args[i] for i in WRITES[call[1]]]
    reads = [b for i, a in enumerate(args) if i not in WRITES[call[1]] and not a.startswith("sizeof ")
             for b in BUFFER.findall(a)]
    return call[1], int(args[1]) if call[1] == "device" else None, reads, writes


def plan(graph_c):
    """Parse graph.c and return its declarations, statements and the 54 segments with liveness."""
    text = read(graph_c).decode()
    head, body = text.split("int atlas_resnet_graph_run(", 1)
    body = body.split("\n}\n", 1)[0].splitlines()[1:]
    sizes = {name: int(count) * (4 if kind == "float" else 1)
             for kind, name, count in re.findall(r"static (float|uint8_t) (\w+)\[(\d+)\]", head)}
    prologue = [line.strip() for line in body[:2]]
    if prologue != ["if (!input || !output || !device) return -1;", "int scale=0;"] or body[-1].strip() != "return 0;":
        raise ValueError("unexpected graph prologue or epilogue")
    lines = [line.strip() for line in body[2:-1]]
    ops = [dict(zip(("kind", "launch", "reads", "writes"), statement(line)), text=line) for line in lines]
    packs = [i for i, op in enumerate(ops) if op["kind"] == "pack"]
    if len(packs) != 54:
        raise ValueError("expected one pack per launch")
    starts = [0] + packs[1:] + [len(ops)]
    segments = []
    for k in range(54):
        seg = ops[starts[k]:starts[k + 1]]
        if [op["launch"] for op in seg if op["kind"] == "device"] != [k]:
            raise ValueError("segment does not contain exactly its own launch")
        written, live_in = [], []
        for op in seg:
            for b in op["reads"]:
                if b not in written and b not in live_in and not b.startswith("params_") and b != "input":
                    live_in.append(b)
            written += [b for b in op["writes"] if b not in written]
        params = sorted({b for op in seg for b in op["reads"] if b.startswith("params_")}, key=lambda s: int(s[7:]))
        segments.append(dict(index=k, first=starts[k], end=starts[k + 1], live_in=live_in, written=written,
                             params=params, uses_input=any("input" in op["reads"] for op in seg),
                             a_bytes=None, out_bytes=None))
        device = next(op for op in seg if op["kind"] == "device")
        args = [a.strip() for a in device["text"][len("if (device("):].split(")")[0].split(",")]
        segments[-1].update(a_bytes=int(args[3]), out_bytes=int(args[6]))
    for k, seg in enumerate(segments):
        later = {b for s in segments[k + 1:] for b in s["live_in"]}
        seg["live_out"] = [b for b in seg["written"] if b in later or b == "output"]
        for b in seg["live_in"]:
            producers = [s["index"] for s in segments[:k] if b in s["written"]]
            if not b.startswith("tensor_") or len(producers) != 1:
                raise ValueError(f"segment {k} live-in {b} lacks one earlier producer")
            seg.setdefault("producers", {})[b] = producers[0]
    for b in sizes:
        owners = [s["index"] for s in segments if b in s["written"]]
        if b.startswith("tensor_") and len(owners) != 1:
            raise ValueError(f"{b} must be produced by exactly one segment")
    return dict(head=head, ops=ops, segments=segments, sizes=sizes, prologue=prologue)


def segment_source(p, seg):
    """Exact graph.c statement text for one segment, with phase marks between statements."""
    proto = [line for line in p["head"].splitlines() if line.startswith(("typedef", "int atlas_graph_"))]
    out = ["#include <stdint.h>", "#include <stddef.h>", *proto, "void atlas_segment_mark(unsigned);"]
    for b in seg["live_in"]:
        out.append(f"extern float {b}[];")
    for b in seg["written"]:
        if b == "output":
            continue
        kind = "uint8_t" if b in ("packed_a", "physical_output") else "float"
        size = p["sizes"][b] // (1 if kind == "uint8_t" else 4)
        visibility = "" if b in seg["live_out"] else "static "
        out.append(f'{visibility}{kind} {b}[{size}] __attribute__((aligned(64), section(".bss.seg_scratch")));')
    out += [f"extern const float {b}[];" for b in seg["params"]]
    out += ["int atlas_resnet_segment_run(const float *input,float *output,void *context,atlas_graph_device device) {",
            *("  " + line for line in p["prologue"])]
    for i, op in enumerate(p["ops"][seg["first"]:seg["end"]]):
        out += ["  " + op["text"], f"  atlas_segment_mark({i});"]
    out += ["  return 0;", "}", ""]
    return "\n".join(out)


def incbin(section, label, path, flags="a"):
    return [f'.section {section},"{flags}",@progbits', ".balign 64", f".global {label}", f"{label}:",
            '.incbin ' + json.dumps(str(path)), f".size {label}, .-{label}"]


def build(args):
    p = plan(args.host / "graph/graph.c")
    capture = args.capture
    manifest, capture_sha, digests = capture_manifest(args.host, capture)
    bound_host_file(args.host / "graph/graph.c", digests)
    assets = bound_host_file(args.host / "assets.c", digests).decode()
    kernels = assets.split("graph_kernels[54]={\n", 1)[1].split("\n};")[0].splitlines()
    header = bound_host_file(args.host / "assets.h", digests).decode()
    defines = ["-D" + "=".join(d) for d in re.findall(r"#define (ATLAS_GRAPH_\w+) (\d+)", header)]
    receipts = []
    for k in args.segments:
        seg = p["segments"][k]
        declared = re.findall(r"\.program_words=(\d+)(?=[,}])", kernels[k])
        if len(declared) != 1:
            raise ValueError("segment kernel must declare exactly one program word count")
        validate_graph_program(bound_host_file(args.host / f"graph_program_{k}.bin", digests), int(declared[0]))
        out = args.out / f"seg-{k:02d}"
        out.mkdir(parents=True)
        state, expected, arena_start = segment_inputs(args.host, capture, manifest, p, seg, digests)
        offset, out_bytes = manifest["kernels"][k]["output_offset"], seg["out_bytes"]
        runtime = manifest["boundaries"][k + 1]["context"]
        for name, raw in [*state.items(), *expected.items(), ("arena", arena_start)]:
            if len(raw) % 8:
                raise ValueError("region not a multiple of 64-bit words: " + name)
            (out / f"{name}.bin").write_bytes(raw)
        (out / "segment.c").write_text(segment_source(p, seg))
        asm = []
        for b in seg["live_in"]:
            asm += incbin(".seg_state", b, out / f"{b}.bin", "aw")
        for name in expected:
            asm += incbin(".seg_expected", "expected_" + name, out / f"{name}.bin")
        asm += incbin(".rodata", "graph_rescale", args.host / "graph_rescale.bin")
        asm += incbin(".rodata", f"graph_program_{k}", args.host / f"graph_program_{k}.bin")
        asm += incbin(".rodata", f"graph_weight_{k}", args.host / f"graph_weight_{k}.bin")
        if seg["uses_input"]:
            asm += incbin(".rodata", "graph_entry", args.host / "graph_entry.bin")
        for b in seg["params"]:
            asm += incbin(".rodata", b, args.host / f"graph/params-{b[7:]}.bin")
        asm += ['.section .atlas_dram,"aw",@progbits', '.incbin ' + json.dumps(str(out / 'arena.bin'))]
        host_inputs = [args.host / "graph_rescale.bin", args.host / f"graph_program_{k}.bin",
                       args.host / f"graph_weight_{k}.bin",
                       *(args.host / f"graph/params-{b[7:]}.bin" for b in seg["params"])]
        if seg["uses_input"]:
            host_inputs.append(args.host / "graph_entry.bin")
        for path in host_inputs:
            bound_host_file(path, digests)
        (out / "images.S").write_text("\n".join(asm) + "\n")
        table = [re.sub(r"\.program=graph_program_\d+,\.weights=graph_weight_\d+", ".program=0,.weights=0", row)
                 if i != k else row for i, row in enumerate(kernels)]
        regions = [(b, b, len(raw)) for b, raw in expected.items() if b not in ("atlas_output", "output",
                                                                               "reference_logits")]
        regions.append(("atlas_output", f"(const void *)(uintptr_t)(ATLAS_DRAM_BASE+{offset})", out_bytes))
        if "output" in expected:
            regions += [("output", "logits", 4000), ("reference_logits", "logits", 4000)]
        if len(regions) > 8:
            raise ValueError("too many segment end-state regions")
        lines = ['#include "atlas_resnet_graph_device.h"', '#include "atlas_host_layout.h"', "#include \"atlas_segment.h\"",
                 *(f"extern const uint32_t graph_program_{k}[];", f"extern const uint8_t graph_weight_{k}[];",
                   "extern const uint32_t graph_rescale[];"),
                 *(f"extern const uint8_t expected_{name}[];" for name in expected),
                 *(f"extern float {b}[];" for b, *_ in regions if b.startswith("tensor_")),
                 "extern float logits[1000];" if "output" in expected else "",
                 ("static const float graph_entry_unused[16] = {0};" if not seg["uses_input"]
                  else "extern const float graph_entry[];"),
                 "const atlas_graph_kernel graph_kernels[54]={", *table, "};",
                 f"const atlas_segment_info atlas_segment={{{k},{seg['end'] - seg['first']},{len(regions)},"
                 f"{runtime['activation_scale'][k]},{runtime['rescale_exponent'][k]},"
                 f"{'graph_entry' if seg['uses_input'] else 'graph_entry_unused'},{{"]
        lines += [f'{{"{name}",(const void *){address},expected_{name},{size}u}},' for name, address, size in regions]
        lines += ["}};", ""]
        (out / "segment_data.c").write_text("\n".join(lines))
        (out / "link.ld").write_text(LINK)
        sources = [args.start, TOOLS / "resnet_segment.c", out / "segment.c", out / "segment_data.c", out / "images.S",
                   *(args.runtime / n for n in SOURCES)]
        command = [str(args.clang), *FLAGS, "--ld-path=" + str(args.lld), "-I" + str(args.runtime),
                   "-I" + str(RUNTIME), "-I" + str(TOOLS), *defines,
                   *map(str, sources), "-Wl,-T," + str(out / "link.ld"), "-Wl,--no-relax",
                   "-Wl,-Map," + str(out / "host.map"), "-o", str(out / "host.elf")]
        bound_host_file(args.start, digests)
        for name in (*SOURCES, "atlas_host.h", "atlas_resnet_graph_device.h"):
            bound_host_file(args.runtime / name, digests)
        for name in ("atlas_host_layout.h", "atlas_numeric.h"):
            path = admitted_path(args.runtime / name)
            if path.exists():
                bound_host_file(path, digests)
        result = subprocess.run(command, timeout=args.wall_seconds, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        (out / "build.log").write_text(result.stdout)
        if result.returncode:
            raise RuntimeError(result.stdout)
        receipt = dict(capture_sha256=capture_sha, segment=k,
                       statements=[op["text"] for op in p["ops"][seg["first"]:seg["end"]]],
                       live_in=seg["live_in"], live_out=seg["live_out"], written=seg["written"],
                       producers=seg.get("producers", {}), params=seg["params"],
                       start_sha256={n: sha(r) for n, r in [*state.items(), ("arena", arena_start)]},
                       expected_sha256={n: sha(r) for n, r in expected.items()}, command=command,
                       elf_sha256=sha((out / "host.elf").read_bytes()))
        write_json(out / "build.json", receipt)
        receipts.append(dict(segment=k, elf=str(out / "host.elf"), elf_sha256=receipt["elf_sha256"]))
        print(json.dumps(receipts[-1]), flush=True)
    return receipts


def stitch(args):
    """Bind build bytes to captured boundaries and compare producer-to-consumer tensor regions."""
    p = plan(args.host / "graph/graph.c")
    manifest, capture_sha, digests = capture_manifest(args.host, args.capture)
    bound_host_file(args.host / "graph/graph.c", digests)
    rows = []
    for seg in p["segments"]:
        out = args.out / f"seg-{seg['index']:02d}"
        receipt = verify_build(out, p, seg)
        state, expected, arena = segment_inputs(args.host, args.capture, manifest, p, seg, digests)
        if (receipt.get("capture_sha256") != capture_sha
            or receipt["start_sha256"] != {n: sha(r) for n, r in [*state.items(), ("arena", arena)]}
            or receipt["expected_sha256"] != {n: sha(r) for n, r in expected.items()}):
            raise ValueError("build regions are not bound to the supplied capture")
        for name, producer in seg.get("producers", {}).items():
            ok = read(out / f"{name}.bin") == read(args.out / f"seg-{producer:02d}/{name}.bin")
            rows.append(dict(segment=seg["index"], region=name, producer=producer, equal=ok))
    covered = [i for seg in p["segments"] for i in range(seg["first"], seg["end"])]
    once = sorted(covered) == list(range(len(p["ops"])))
    launches = [op["launch"] for op in p["ops"] if op["kind"] == "device"] == list(range(54))
    bad = sum(not row["equal"] for row in rows)
    report = dict(status="PASS" if not bad and once and launches else "FAIL", mismatches=bad,
                  statements=len(p["ops"]), statements_covered_once=once, launches=launches,
                  capture_sha256=capture_sha, forward_regions=len(rows), rows=rows)
    write_json(args.out / "stitch.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))
    return 0 if report["status"] == "PASS" else 1


def fields(line):
    return {key: int(value) if re.fullmatch(r"-?\d+", value) else value for key, value in re.findall(r"(\w+)=(\S+)", line)}


def results(args):
    """Require a successful process exit and complete, consistent segment checks."""
    p = plan(args.host / "graph/graph.c")
    rows = []
    for k in args.segments:
        receipt = verify_build(args.out / f"seg-{k:02d}", p, p["segments"][k])
        run = admitted_path(args.runs / args.run_name.format(k=k))
        log = read(run / "sim.log").decode(errors="replace") if (run / "sim.log").exists() else ""
        ops = [fields(line) for line in log.splitlines() if line.startswith("SEG_OP ")]
        checks = [fields(line) for line in log.splitlines() if line.startswith("SEG_CHECK ")]
        finals = [line for line in log.splitlines() if line.startswith(("SEG_PASS ", "SEG_FAIL ", "SEG_TRAP "))]
        status = read(run / "exit-status.txt").decode().strip() if (run / "exit-status.txt").exists() else None
        run_receipt = json.loads(read(run / "run.json")) if (run / "run.json").exists() else None
        run_ok = (run_receipt is not None and run_receipt.get("elf_sha256") == receipt["elf_sha256"]
                                         and run_receipt.get("returncode") == 0 and run_receipt.get("timed_out") is False)
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
        kinds = [re.match(r"(?:if \((?:atlas_graph_)?(\w+)|for)", t)[1] or "copy" for t in receipt["statements"]]
        final = fields(finals[-1]) if finals else {}
        expected = receipt["expected_sha256"]
        checks_ok = (len(checks) == len(expected) and {c.get("region") for c in checks} == set(expected)
                     and all(c.get("mismatches") == 0 and c.get("first") == -1
                             and c.get("words") == len(read(args.out / f"seg-{k:02d}/{c['region']}.bin")) // 8
                             for c in checks))
        ops_ok = (len(ops) == len(kinds) and all(op.get("segment") == k and op.get("statement") == i
                                               for i, op in enumerate(ops)))
        passed = (status == "0" and run_ok and not (run / "STOPPED.txt").exists() and len(finals) == 1
                  and finals[0].startswith("SEG_PASS ") and final.get("segment") == k
                  and final.get("state_ok") == 1 and final.get("mismatches") == 0 and checks_ok and ops_ok)
        result = "PASS" if passed else ("RUNNING" if status is None and log and not finals else "FAIL")
        row = dict(segment=k, statements=kinds, run=str(run), done=len(ops), total=len(kinds), exit=status,
                   result=result, wall=wall[1] if wall else None,
                   op_cpu_cycles=[o.get("cpu_cycles") for o in ops], op_instret=[o.get("instret") for o in ops],
                   checks={c.get("region"): c.get("mismatches") for c in checks}, final=final)
        row.update({key: final[key] for key in ("cpu_cycles", "atlas_cycles", "mismatches") if key in final})
        rows.append(row)
    write_json(args.out / "results.json", rows)
    for row in rows:
        print(json.dumps({key: row.get(key) for key in ("segment", "result", "done", "total", "cpu_cycles",
                                                         "atlas_cycles", "mismatches", "wall", "exit")}))
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "build", "stitch", "results"))
    parser.add_argument("--host", type=admitted_path, required=True,
                        help="full-host build directory (graph/, assets.c, *.bin)")
    parser.add_argument("--capture", type=admitted_path)
    parser.add_argument("--out", type=admitted_path)
    parser.add_argument("--clang", type=admitted_path)
    parser.add_argument("--lld", type=admitted_path)
    parser.add_argument("--start", type=admitted_path, help="startup assembly (the full host's start.S)")
    parser.add_argument("--runtime", type=admitted_path, default=RUNTIME, help="runtime source directory (build)")
    parser.add_argument("--segments", type=int, nargs="+", default=list(range(54)))
    parser.add_argument("--runs", type=admitted_path, help="simulator run directory root (results)")
    parser.add_argument("--run-name", default="seg-{k:02d}", help="run directory name per segment (results)")
    parser.add_argument("--wall-seconds", type=int, default=1500, help="compiler subprocess deadline (build)")
    args = parser.parse_args()
    required = {"plan": (), "build": ("capture", "out", "clang", "lld", "start"),
                "stitch": ("capture", "out"), "results": ("out", "runs")}[args.command]
    for name in required:
        if getattr(args, name) is None:
            parser.error("--" + name + " is required for " + args.command)
    if not args.segments or len(set(args.segments)) != len(args.segments) or any(k < 0 or k >= 54 for k in args.segments):
        parser.error("--segments requires unique indices in 0..53")
    if args.wall_seconds <= 0:
        parser.error("--wall-seconds must be positive")
    if args.command in ("plan", "stitch") and args.segments != list(range(54)):
        parser.error("--segments applies only to build and results")
    try:
        for k in args.segments:
            name = args.run_name.format(k=k)
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError("run name must be one directory component")
            admitted_path(name)
    except (KeyError, ValueError, IndexError) as error:
        parser.error("invalid --run-name: " + str(error))
    if args.command == "plan":
        p = plan(args.host / "graph/graph.c")
        for seg in p["segments"]:
            print(json.dumps({k: seg[k] for k in ("index", "first", "end", "live_in", "live_out", "params")}))
    elif args.command == "build":
        build(args)
    elif args.command == "stitch":
        return stitch(args)
    else:
        return results(args)


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Python -O is unsupported")
    raise SystemExit(main())
