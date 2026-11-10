"""Initial Atlas consumer of Merlin's checked ordered dense tile schedule.

This is a delegated diagnostic compiler, not a generated backend. Merlin owns
source recognition, blocking, loops and their coverage/residency proof. Atlas
owns the tile ABI, resource mapping and instruction expansion. Only full tiles
on MXU0 are admitted; no implicit precision conversion or padding is performed.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import struct
import subprocess

from merlin.compile.scheduling.compact_dense import (
    DenseLoopAdapter, DenseLoopStorage, DenseMemoryView, schedule_compact_dense,
)
from merlin.compile.scheduling.dense_tile_ir import encode_dense_tile, project_dense_tile
from merlin.sched.ir import instances
from merlin.sched.ir.kernel import concretize
from merlin.transforms.dense_planning import DenseProblem, DenseResources
from merlin.transforms.ordered_float_dense import OrderedFloatPolicy, read_ordered_float_dense

RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
MODEL_REVISION = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
POLICY = OrderedFloatPolicy("f8E4M3FN", "bf16", "mxu0-e4m3-ordered-bf16-finite-v1",
                            "finite-normal-e4m3-or-positive-zero")
SLOT_BYTES = 2048
INPUT_BASE = 0x90000000
OUTPUT_BASE = 0x90080000
MAX_TILE_PRODUCTS = 64


def plan(source, *, entry):
    selected = read_ordered_float_dense(source, entry=entry, policy=POLICY)
    if any(d % 32 for d in (selected.m, selected.k, selected.n)):
        raise ValueError("initial Atlas dense path requires full 32x32 tiles; padding is not admitted")
    mt, kt, nt = selected.m // 32, selected.k // 32, selected.n // 32
    if mt * kt * nt > MAX_TILE_PRODUCTS:
        raise ValueError("initial Atlas dense path exceeds its finite instruction expansion budget")
    problem = DenseProblem(selected.m, selected.k, selected.n, 32, 32, 32, 1, 1, 2)
    # A row view of one assigned tile per role. The physical accumulator is a
    # slot, never individually row-addressed; the lowerer admits its base only.
    storage = DenseLoopStorage(
        DenseMemoryView("lhs", "mreg", 32, 0, 32),
        DenseMemoryView("rhs", "mreg", 32, 32, 32),
        DenseMemoryView("acc", "mxu0.acc", 64, 0, 32),
    )
    adapter = DenseLoopAdapter(encode_dense_tile, project_dense_tile, RTL_REVISION,
                               POLICY.identity, True, True, 64)
    scheduled = schedule_compact_dense(
        problem, DenseResources(2048, 2048, 1, 1, 1), storage, adapter,
        name="ordered_dense", dtypes=("fp8_e4m3", "fp8_e4m3", "bf16"),
        static_budget=32, external_buffers_disjoint=True,
    )
    if scheduled is None:
        raise ValueError("no legal shared dense schedule")
    return selected, scheduled


def virtual_program(selected, scheduled):
    """Expand the actual checked calls into existing Atlas virtual SSA."""
    scheduled.revalidate()
    if (selected.policy != POLICY or tuple((a.name, a.shape, a.dtype) for a in scheduled.kernel.args) != (
            ("lhs", (selected.m, selected.k), "fp8_e4m3"),
            ("rhs", (selected.k, selected.n), "fp8_e4m3"),
            ("output", (selected.m, selected.n), "bf16"))):
        raise ValueError("Atlas source selection differs from the checked logical schedule")
    kt, nt = selected.k // 32, selected.n // 32
    rhs_base = selected.m // 32 * kt
    lines = ["module {", "  func.func @ordered_dense() -> !atlas.virtual_state attributes {"
             f"atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {OUTPUT_BASE} : i64" + "} {"]
    state, serial = "%s0", 0
    lines.append(f'    {state} = "atlas.virtual_start"() : () -> !atlas.virtual_state')
    lhs = rhs = weight = accumulator = pending = readout_value = None
    def emit(op, operands, types, result_type=None, attrs=""):
        nonlocal serial, state
        serial += 1
        new_state, value = f"%s{serial}", f"%v{serial}"
        names = new_state if result_type is None else f"{new_state}, {value}"
        result = "!atlas.virtual_state" if result_type is None else f"(!atlas.virtual_state, {result_type})"
        lines.append(f'    {names} = "atlas.{op}"({", ".join([state, *operands])}){attrs} : '
                     f'({", ".join(["!atlas.virtual_state", *types])}) -> {result}')
        state = new_state
        return value
    fp8, bf16 = "!atlas.virtual_fp8", "!atlas.virtual_bf16"
    wt, acc = "!atlas.virtual_mxu_weight<0>", "!atlas.virtual_mxu_acc<0>"
    for call, env in instances(scheduled.kernel):
        project_dense_tile(call)
        a = concretize(call, env)
        if call.instr.startswith("dense.load_"):
            if (a["rows"], a["cols"], a["step"]) != (32, 32, 1):
                raise ValueError("Atlas tile expansion received a partial or strided load")
            if call.instr == "dense.load_lhs":
                if a["dst"] != 0:
                    raise ValueError("Atlas lhs assignment differs")
                index = (a["src_row"] // 32) * kt + a["src_col"] // 32
                lhs = emit("virtual_input_fp8", [], [], fp8, f" {{index = {index} : i32}}")
            else:
                if a["dst"] != 32:
                    raise ValueError("Atlas rhs assignment differs")
                index = rhs_base + (a["src_row"] // 32) * nt + a["src_col"] // 32
                rhs = emit("virtual_input_fp8", [], [], fp8, f" {{index = {index} : i32}}")
        elif call.instr == "dense.preload":
            if rhs is None or (a["weight"], a["acc"], a["rows"], a["cols"], a["depth"]) != (32, 0, 32, 32, 32):
                raise ValueError("Atlas preload assignment differs")
            pending = a["continue"]
            # A complete reset/readout can use the existing combined operation.
            # Continued reductions retain explicit resource-version operations.
            weight = rhs if kt == 1 else emit("virtual_mxu_load_weight", [rhs], [fp8], wt, " {unit = 0 : i32}")
        elif call.instr == "dense.compute":
            if lhs is None or weight is None or pending not in (0, 1) or (
                    a["lhs"], a["rows"], a["depth"], a["preloaded"]) != (0, 32, 32, 1):
                raise ValueError("Atlas compute assignment differs")
            if kt == 1:
                if pending or accumulator is not None:
                    raise ValueError("combined Atlas matmul requires one complete reset")
                serial += 1
                readout_value = f"%v{serial}"
                lines.append(f'    {readout_value} = "atlas.virtual_mxu_matmul"({lhs}, {weight}) '
                             f'{{unit = 0 : i32}} : ({fp8}, {fp8}) -> {bf16}')
                accumulator = readout_value
            elif pending:
                if accumulator is None:
                    raise ValueError("Atlas continuation has no live accumulator")
                accumulator = emit("virtual_mxu_accumulate", [lhs, weight, accumulator], [fp8, wt, acc], acc)
            else:
                if accumulator is not None:
                    raise ValueError("Atlas reset would discard a live accumulator")
                accumulator = emit("virtual_mxu_reset", [lhs, weight], [fp8, wt], acc)
            pending = None
        elif call.instr == "dense.store":
            if accumulator is None or (a["acc"], a["rows"], a["cols"]) != (0, 32, 32):
                raise ValueError("Atlas readout assignment differs")
            value = readout_value if kt == 1 else emit("virtual_mxu_readout_bf16", [accumulator], [acc], bf16)
            index = (a["row"] // 32) * nt + a["col"] // 32
            emit("virtual_output_bf16", [value], [bf16], attrs=f" {{index = {index} : i32}}")
            accumulator = None
    if accumulator is not None or pending is not None:
        raise ValueError("Atlas dense expansion left live state")
    lines.extend((f"    return {state} : !atlas.virtual_state", "  }", "}"))
    return "\n".join(lines) + "\n"


def pack_inputs(selected, lhs: bytes, rhs: bytes):
    """Enforce the source domain and map row-major inputs to Atlas tile slots."""
    if len(lhs) != selected.m * selected.k or len(rhs) != selected.k * selected.n:
        raise ValueError("input byte extent differs from source")
    if any(v and (v & 0x78 == 0 or v & 0x7f == 0x7f) for v in lhs + rhs):
        raise ValueError("input outside finite-normal E4M3 or positive-zero domain")
    payloads = []
    for m in range(0, selected.m, 32):
        for k in range(0, selected.k, 32):
            payloads.append(b"".join(lhs[(m+r)*selected.k+k:(m+r)*selected.k+k+32] for r in range(32)))
    for k in range(0, selected.k, 32):
        for n in range(0, selected.n, 32):
            payloads.append(bytes(rhs[(n+c)*selected.k+k+r] if selected.rhs_transposed else
                                  rhs[(k+r)*selected.n+n+c] for c in range(32) for r in range(32)))
    return tuple((i*SLOT_BYTES, payload) for i, payload in enumerate(payloads))


def compile_dense(source, *, entry, destination, atlas_bin_dir, pipeline="source-order"):
    if pipeline not in ("source-order", "optimized"):
        raise ValueError("choose source-order or optimized pipeline")
    selected, scheduled = plan(source, entry=entry)
    virtual = virtual_program(selected, scheduled)
    destination, bins = Path(destination), Path(atlas_bin_dir)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source.mlir").write_text(source)
    (destination / "schedule.mk").write_text(scheduled.kernel.text())
    (destination / "virtual.mlir").write_text(virtual)
    command = [str(bins / "atlas-opt"), str(destination / "virtual.mlir"),
               "--atlas-compile-unscheduled" if pipeline == "source-order" else "--atlas-compile",
               "--verify-atlas-generated-schedule", "--verify-atlas-machine-stream",
               "-o", str(destination / "machine.mlir")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    words = subprocess.run([str(bins / "atlas-emit"), str(destination / "machine.mlir")],
                           check=True, capture_output=True, text=True).stdout
    program = subprocess.run([str(bins / "atlas-emit"), "--program-json", str(destination / "machine.mlir")],
                             check=True, capture_output=True, text=True).stdout
    if json.loads(program).get("selected_rtl_revision") != RTL_REVISION:
        raise ValueError("Atlas compiler selected a different RTL revision")
    raw = b"".join(struct.pack("<I", int(word, 16)) for word in words.split())
    (destination / "program.bin").write_bytes(raw)
    (destination / "program.json").write_text(program)
    manifest = {"schema": "atlas.shared-dense-diagnostic.v1", "source": asdict(selected),
                "pipeline": pipeline, "words": len(raw)//4, "binary_sha256": sha256(raw).hexdigest(),
                "kernel_sha256": scheduled.kernel.digest(), "shared_realization": scheduled.realization,
                "reference_model_revision": MODEL_REVISION,
                "scope": "shared logical schedule; Atlas delegated physical lowering; execution required",
                "input_guard": "pack_inputs requires finite-normal E4M3 or positive zero",
                "tools": {name: sha256((bins/name).read_bytes()).hexdigest() for name in ("atlas-opt", "atlas-emit")}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--atlas-bin-dir", type=Path, required=True)
    parser.add_argument("--pipeline", choices=("source-order", "optimized"), default="source-order")
    args = parser.parse_args()
    print(json.dumps(compile_dense(args.source.read_text(), entry=args.entry, destination=args.destination,
                                   atlas_bin_dir=args.atlas_bin_dir, pipeline=args.pipeline), indent=2))
