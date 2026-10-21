"""Repeat a selected softmax row body with private writable row frames."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.compile.counted_loop import repeat_state_cfg, simplify_state_cfg
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.semantic_compiler.target_binding import NativeCompilationError, verify_native_publication
from merlin.semantic_compiler.tiled_softmax import RowSoftmaxTiling

from .binding import (
    DRAM_BASE,
    NATIVE_CHECKS,
    NATIVE_PIPELINES,
    RTL_REVISION,
    TILE_BYTES,
    AtlasNativeBinding,
    _native_pipeline,
    _run,
    dram_slots,
)
from .resident_graph import is_staged_bf16, staged_memory_plan
from .row_softmax import prepare_row_softmax


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RowLoopPlan:
    tiling: RowSoftmaxTiling
    readonly_bytes: int
    output_bytes: int
    scratch_bytes: int

    @property
    def groups(self):
        return len(self.tiling.regions)

    @property
    def writable_base(self):
        return DRAM_BASE + self.groups * self.readonly_bytes

    @property
    def scratch_base(self):
        return self.writable_base + self.output_bytes

    @property
    def writable_bytes(self):
        return self.output_bytes + self.scratch_bytes

    @property
    def end(self):
        return self.writable_base + self.groups * self.writable_bytes

    def address(self, original, group):
        if type(group) is not int or not 0 <= group < self.groups:
            raise ValueError("row group is outside the selected loop")
        offset = original - DRAM_BASE
        if not 0 <= offset < self.readonly_bytes + self.output_bytes + self.scratch_bytes:
            raise ValueError("row-body address is outside its selected frame")
        if offset < self.readonly_bytes:
            return original + group * self.readonly_bytes
        return self.writable_base + group * self.writable_bytes + offset - self.readonly_bytes

    def record(self):
        return {
            "schema": "atlas.row_loop_plan.v1",
            "source_request_digest": self.tiling.source.digest(),
            "body_request_digest": self.tiling.regions[0].request.digest(),
            "groups": self.groups,
            "readonly_base": DRAM_BASE,
            "readonly_stride_bytes": self.readonly_bytes,
            "output_base": self.writable_base,
            "output_stride_bytes": self.writable_bytes,
            "scratch_address": self.scratch_base,
            "scratch_bytes": self.scratch_bytes,
            "scratch_stride_bytes": self.writable_bytes,
            "scratch_lifetime": "private row frame retained until program completion",
            "shape": list(self.tiling.shape),
        }


def prepare_plan(source):
    tiling = prepare_row_softmax(source)
    body = tiling.regions[0].request
    if any(replace(row.request, source_identity=body.source_identity) != body for row in tiling.regions):
        raise NativeCompilationError("unsupported_semantics", "row bodies differ beyond source provenance")
    abi = AtlasNativeBinding.plan_boundary_abi(body)
    readonly = len(abi["fixed_inputs"])
    outputs = len(abi["fixed_outputs"])
    if list(abi["fixed_inputs"].values()) != list(range(readonly)) or list(abi["fixed_outputs"]) != list(
        range(readonly, readonly + outputs)
    ):
        raise NativeCompilationError("unsupported_semantics", "row loop needs a dense selected boundary frame")
    total = readonly + outputs
    if is_staged_bf16(body):
        _, slots, _, _ = staged_memory_plan(body)
        total = max(slots.values()) + 1
    plan = RowLoopPlan(tiling, readonly * TILE_BYTES, outputs * TILE_BYTES, (total - readonly - outputs) * TILE_BYTES)
    if plan.end > DRAM_BASE + dram_slots() * TILE_BYTES:
        raise NativeCompilationError("resource_limit", "complete row loop exceeds the selected I/O window")
    return plan


def virtual_program(plan, body_text):
    """Relocate only explicit DMA addresses; retain selected computation and CFG."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i32
    from xdsl.dialects.func import FuncOp
    from xdsl.printer import Printer

    module = parse_mlir_text(body_text)
    operations = tuple(module.body.block.ops)
    if len(operations) != 1 or not isinstance(operations[0], FuncOp) or module.attributes:
        raise ValueError("row body must contain one function")
    function = operations[0]
    initial_state = None
    if len(function.body.blocks) == 1:
        first = function.body.block.first_op
        if first is None or first.name != "builtin.unregistered" or first.op_name.data != "atlas.virtual_start":
            raise ValueError("straight-line row body must initialize its virtual state")
        if len(first.results) != 1:
            raise ValueError("row-body initialization must produce one state")
        initial_state = first.results[0]
    loop = repeat_state_cfg(
        function,
        iterations=plan.groups,
        steps=(plan.readonly_bytes, plan.writable_bytes),
        index_type=i32,
        initial_state=initial_state,
    )
    accesses = 0
    for op in tuple(function.walk()):
        name = op.op_name.data if op.name == "builtin.unregistered" else op.name
        if name not in {"atlas.virtual_dma_load_bf16", "atlas.virtual_dma_store_bf16"}:
            continue
        index = 1 if name.endswith("load_bf16") else 2
        pointer = op.operands[index]
        if not isinstance(pointer.owner, arith.ConstantOp):
            raise ValueError("row-body DMA address must be a selected constant")
        original = int(pointer.owner.value.value.data) & 0xFFFFFFFF
        offset = original - DRAM_BASE
        base = plan.address(original, 0)
        if name.endswith("store_bf16") and offset < plan.readonly_bytes:
            raise ValueError("row body writes its immutable frame")
        constant = arith.ConstantOp.from_int_and_width(base, i32, truncate_bits=True)
        inserted = [constant]
        position = 0 if offset < plan.readonly_bytes else 1
        address = arith.AddiOp(constant, loop.offsets[position])
        inserted.append(address)
        value = address.result
        op.parent_block().insert_ops_before(inserted, op)
        operands = list(op.operands)
        operands[index] = value
        op.operands = operands
        accesses += 1
    if not accesses:
        raise ValueError("row body contains no explicit DMA accesses")
    function.attributes["atlas.output_dram_base"] = IntegerAttr(plan.writable_base, 64)
    function.attributes["atlas.output_dram_bytes"] = IntegerAttr(plan.end - plan.writable_base, 64)
    simplify_state_cfg(function)
    module.verify()
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def compile_row_loop(snapshot, source, *, target_source, destination, limits):
    plan = prepare_plan(source)
    body = plan.tiling.regions[0].request
    abi = AtlasNativeBinding.plan_boundary_abi(body)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "source-request.json").write_bytes(_json(source.record()))
    (destination / "row-softmax-plan.json").write_bytes(_json(plan.tiling.record()))
    manifest = AtlasNativeBinding.compile(
        snapshot,
        body,
        fixed_inputs=abi["fixed_inputs"],
        fixed_outputs=tuple(abi["fixed_outputs"]),
        target_source=target_source,
        destination=destination / "body",
        limits=limits,
    )
    verify_native_publication(
        destination / "body",
        manifest,
        engine="merlin_native",
        request_digest=body.digest(),
        target_identity=source.target_identity,
    )
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    explicit = _run(
        [str(tools / "atlas-opt"), "--lower-atlas-virtual-boundary-to-dma", "-"],
        input_text=(destination / "body/atlas-virtual.mlir").read_text(),
    )
    virtual = virtual_program(plan, explicit)
    (destination / "atlas-virtual.mlir").write_text(virtual)
    try:
        machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[_native_pipeline()], "-"], input_text=virtual)
    except NativeCompilationError as error:
        diagnostic = destination / "lowering-error.txt"
        diagnostic.write_text(error.reason + "\n")
        raise NativeCompilationError(error.status, error.reason[:4096] + f"\nFull diagnostic: {diagnostic}") from error
    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise NativeCompilationError("unqualified_target", "row loop emitter uses a different RTL revision")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    execution = {**plan.record(), "schema": "atlas.row_loop_execution_plan.v1", "launches_per_invocation": 1}
    files = {
        "source-request.json": _json(source.record()),
        "row-softmax-plan.json": _json(plan.tiling.record()),
        "execution_plan.json": _json(execution),
        "atlas-virtual.mlir": virtual.encode(),
        "atlas-machine.mlir": machine.encode(),
        "physical-program.json": _json(program),
        "program.bin": binary,
    }
    for name, data in files.items():
        (destination / name).write_bytes(data)
    result = {
        "schema": "atlas.row_loop_compilation.v1",
        "engine": "merlin_native",
        "artifact_kind": "program",
        "compile_mode": source.lowering_policy,
        "qualification_scope": "diagnostic_only",
        "request_digest": source.digest(),
        "target_identity": source.target_identity,
        "binary_sha256": _sha(binary),
        "execution_plan_sha256": _sha(files["execution_plan.json"]),
        "native_snapshot_sha256": _sha((snapshot.root / "manifest.json").read_bytes()),
        "io_memory_bytes": dram_slots() * TILE_BYTES,
        "lowering_pipeline": _native_pipeline(),
        "body_manifest_sha256": _sha((destination / "body/manifest.json").read_bytes()),
        "target_inventory_sha256": _sha((target_source / "docs/selected-variant-inventory.json").read_bytes()),
        "atlas_opt_sha256": _sha((tools / "atlas-opt").read_bytes()),
        "atlas_emit_sha256": _sha((tools / "atlas-emit").read_bytes()),
        "provider_sources_sha256": {p.name: _sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "stage_artifacts": {
            p.relative_to(destination).as_posix(): _sha(p.read_bytes())
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        },
    }
    (destination / "manifest.json").write_bytes(_json(result))
    verify_row_loop(source, destination, target_source=target_source)
    return result


def verify_row_loop(source, compiled, *, target_source):
    """Replay row/frame formation and physical lowering against the native body."""
    from .binding import check_io_memory_manifest

    plan = prepare_plan(source)
    manifest = json.loads((compiled / "manifest.json").read_text())
    check_io_memory_manifest(manifest)
    if manifest.get("schema") != "atlas.row_loop_compilation.v1":
        raise ValueError("unsupported row-loop compilation")
    verify_native_publication(
        compiled,
        manifest,
        engine="merlin_native",
        request_digest=source.digest(),
        target_identity=source.target_identity,
    )
    body = plan.tiling.regions[0].request
    body_manifest = json.loads((compiled / "body/manifest.json").read_text())
    verify_native_publication(
        compiled / "body",
        body_manifest,
        engine="merlin_native",
        request_digest=body.digest(),
        target_identity=source.target_identity,
    )
    if (
        json.loads((compiled / "source-request.json").read_text()) != source.record()
        or json.loads((compiled / "row-softmax-plan.json").read_text()) != plan.tiling.record()
        or json.loads((compiled / "execution_plan.json").read_text())
        != {**plan.record(), "schema": "atlas.row_loop_execution_plan.v1", "launches_per_invocation": 1}
        or manifest.get("body_manifest_sha256") != _sha((compiled / "body/manifest.json").read_bytes())
        or manifest.get("target_inventory_sha256")
        != _sha((target_source / "docs/selected-variant-inventory.json").read_bytes())
    ):
        raise ValueError("row loop differs from its source, selected body or physical frame")
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", target_source / "build/bin"))
    for key, filename in (("atlas_opt_sha256", "atlas-opt"), ("atlas_emit_sha256", "atlas-emit")):
        if manifest.get(key) != _sha((tools / filename).read_bytes()):
            raise ValueError("row-loop lowering tool changed")
    pipeline = manifest.get("lowering_pipeline")
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError("unknown row-loop lowering pipeline")
    explicit = _run(
        [str(tools / "atlas-opt"), "--lower-atlas-virtual-boundary-to-dma", "-"],
        input_text=(compiled / "body/atlas-virtual.mlir").read_text(),
    )
    virtual = virtual_program(plan, explicit)
    if virtual != (compiled / "atlas-virtual.mlir").read_text():
        raise ValueError("row-loop CFG or address transformation changed")
    machine = _run([str(tools / "atlas-opt"), *NATIVE_PIPELINES[pipeline], "-"], input_text=virtual)
    if machine != (compiled / "atlas-machine.mlir").read_text():
        raise ValueError("row-loop physical lowering changed")
    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    if (
        program != json.loads((compiled / "physical-program.json").read_text())
        or binary != (compiled / "program.bin").read_bytes()
    ):
        raise ValueError("row-loop program differs from its checked physical lowering")
    return plan, manifest
