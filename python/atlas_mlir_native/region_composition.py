"""Encode already connected, globally allocated virtual stages as one image.

The caller supplies the physical buffer contract. This module does not insert
host conversions, rewrite addresses, change arithmetic, or join program sets.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeviceBuffer:
    name: str
    address: int
    bytes: int
    layout: str


@dataclass(frozen=True)
class VirtualStage:
    name: str
    source: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]


def _contract(stages, buffers, initialized, outputs, io_memory_bytes):
    from .binding import DRAM_BASE, TILE_BYTES

    if not stages or len({stage.name for stage in stages}) != len(stages):
        raise ValueError("composition requires nonempty stages with unique names")
    if type(io_memory_bytes) is not int or io_memory_bytes <= 0 or io_memory_bytes % TILE_BYTES:
        raise ValueError("composition I/O window must contain whole positive slots")
    if DRAM_BASE + io_memory_bytes > 1 << 32:
        raise ValueError("composition I/O window exceeds the 32-bit buffer ABI")
    by_name = {buffer.name: buffer for buffer in buffers}
    if len(by_name) != len(buffers) or any(not key for key in by_name):
        raise ValueError("composition buffers require unique nonempty names")
    ordered = sorted(buffers, key=lambda buffer: buffer.address)
    for index, buffer in enumerate(ordered):
        if (type(buffer.address) is not int or type(buffer.bytes) is not int
                or buffer.bytes <= 0 or buffer.address % TILE_BYTES or buffer.bytes % TILE_BYTES
                or not buffer.layout or buffer.address < DRAM_BASE
                or buffer.address + buffer.bytes > DRAM_BASE + io_memory_bytes):
            raise ValueError("composition buffer must fit whole slots in the selected I/O window")
        if index and ordered[index - 1].address + ordered[index - 1].bytes > buffer.address:
            raise ValueError("distinct composition buffers must not overlap")
    immutable = set(initialized)
    if not immutable <= by_name.keys():
        raise ValueError("initialized buffer is undeclared")
    ready, written = set(immutable), set()
    for stage in stages:
        if not stage.name or not stage.writes:
            raise ValueError("each composition stage needs a name and output storage")
        if not set(stage.reads + stage.writes) <= by_name.keys():
            raise ValueError("stage references an undeclared buffer")
        if not set(stage.reads) <= ready:
            raise ValueError("stage reads a buffer before its producer")
        if immutable.intersection(stage.writes):
            raise ValueError("stage writes an immutable input buffer")
        ready.update(stage.writes)
        written.update(stage.writes)
    if not outputs or not set(outputs) <= written:
        raise ValueError("composition outputs require completed stage producers")
    mutable = sorted((by_name[name] for name in written), key=lambda buffer: buffer.address)
    for previous, current in zip(mutable, mutable[1:]):
        if previous.address + previous.bytes != current.address:
            raise ValueError("composition requires contiguous output permissions")
    return by_name


def _prepare_source(stage, buffers, program_size_limit_bytes):
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.func import FuncOp
    from xdsl.printer import Printer
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(stage.source)
    functions = list(module.body.block.ops)
    if module.attributes or len(functions) != 1 or not isinstance(functions[0], FuncOp):
        raise ValueError("stage source must contain one function and no module attributes")
    function = functions[0]
    writes = sorted((buffers[name] for name in set(stage.writes)), key=lambda buffer: buffer.address)
    for previous, current in zip(writes, writes[1:]):
        if previous.address + previous.bytes != current.address:
            raise ValueError("each stage requires contiguous output permissions")
    expected = {
        "atlas.output_dram_base": writes[0].address,
        "atlas.output_dram_bytes": sum(buffer.bytes for buffer in writes),
    }
    for name, value in expected.items():
        attr = function.attributes.get(name)
        if not isinstance(attr, IntegerAttr) or attr.value.data != value:
            raise ValueError("stage output bounds differ from its physical buffer contract")
    if program_size_limit_bytes is not None:
        previous = function.attributes.get("atlas.program_size_limit_bytes")
        limit = min(program_size_limit_bytes, previous.value.data) if isinstance(previous, IntegerAttr) else program_size_limit_bytes
        function.attributes["atlas.program_size_limit_bytes"] = IntegerAttr(limit, 64)
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def rematerialize_scalar_constants(source):
    """Clone integer constants at each consumer to shorten scalar lifetimes.

    The rewrite retains the exact typed value and all SSA uses. It can avoid
    scalar induction spills when a composed PACK stage reserves extra scalar
    registers. Normal allocation, address, initialization, and timing checks
    still run on the resulting program.
    """
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.printer import Printer
    from merlin.frontends.linalg_mlir import parse_mlir_text

    module = parse_mlir_text(source)
    for operation in list(module.walk()):
        if operation.name != "arith.constant" or not isinstance(operation.properties.get("value"), IntegerAttr):
            continue
        value = operation.results[0]
        consumers = list(dict.fromkeys(use.operation for use in value.uses))
        for consumer in consumers:
            clone = operation.clone()
            consumer.parent_block().insert_op_before(clone, consumer)
            consumer.operands = tuple(clone.results[0] if operand is value else operand for operand in consumer.operands)
        operation.parent_block().erase_op(operation)
    stream = io.StringIO()
    Printer(stream=stream).print_op(module)
    return stream.getvalue()


def compose_regions(
    stages, *, buffers, initialized, outputs, destination, tools=None,
    pipeline="mandatory", io_memory_bytes=262144, program_size_limit_bytes=None,
    rematerialize_constants=False,
):
    """Compile ordered explicit-DMA stages, with no host work between stages.

    Reads/writes describe a caller-supplied physical contract, not an inferred
    source equivalence proof. Producers and consumers must already agree on
    bytes, layouts, scales, and DMA addresses. Streaming and host conversions
    must have been eliminated before calling this function.
    """
    from .binding import NATIVE_CHECKS, NATIVE_PIPELINES, RTL_REVISION, _run
    from .resident_graph import compose_fragments

    stages, buffers = tuple(stages), tuple(buffers)
    initialized, outputs = tuple(initialized), tuple(outputs)
    by_name = _contract(stages, buffers, initialized, outputs, io_memory_bytes)
    if pipeline not in NATIVE_PIPELINES:
        raise ValueError("composition pipeline must be mandatory or optimized")
    if program_size_limit_bytes is not None and (type(program_size_limit_bytes) is not int or program_size_limit_bytes <= 0):
        raise ValueError("composition code budget must be positive")
    tools = Path(tools or os.environ.get("ATLAS_OOT_BIN_DIR", "build/bin"))
    fragments = [_prepare_source(stage, by_name, program_size_limit_bytes) for stage in stages]
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    virtual = compose_fragments(fragments, tools=tools, boundaries_lowered=True)
    if rematerialize_constants:
        virtual = rematerialize_scalar_constants(virtual)
    (destination / "atlas-virtual.mlir").write_text(virtual)
    machine = compile_virtual_with_receipt(virtual, tools=tools, pipeline=pipeline, destination=destination)
    _run([str(tools / "atlas-opt"), *NATIVE_CHECKS, "--verify-atlas-timing", "-"], input_text=machine)
    program = json.loads(_run([str(tools / "atlas-emit"), "--program-json", "-"], input_text=machine))
    if program.get("selected_rtl_revision") != RTL_REVISION:
        raise ValueError("composition emitter uses a different selected RTL")
    traps = [row for row in program["instructions"] if row["operation"] == "atlas.trap"]
    if len(traps) != 1 or traps[0]["fields"]["kind"] != "ecall":
        raise ValueError("composition must encode exactly one completion ECALL")
    binary = b"".join(row["word_u32"].to_bytes(4, "little") for row in program["instructions"])
    # Capacity is checked by the backend after scheduling and encoding. Also
    # enforce the explicitly requested ceiling on the final published bytes.
    if program_size_limit_bytes is not None and len(binary) > program_size_limit_bytes:
        raise ValueError("final composition exceeds the selected code budget")
    execution = {
        "schema": "atlas.region_composition_execution.v1",
        "launches_per_invocation": 1,
        "lowering_pipeline": pipeline,
        "rematerialize_constants": rematerialize_constants,
        "io_memory_bytes": io_memory_bytes,
        "buffers": [asdict(buffer) for buffer in buffers],
        "initialized": list(initialized),
        "outputs": list(outputs),
        "stages": [{"name": stage.name, "reads": list(stage.reads), "writes": list(stage.writes),
                    "source_sha256": hashlib.sha256(stage.source.encode()).hexdigest()} for stage in stages],
        "physical_contract": "caller_declared; no inferred boundary conversions or source equivalence",
    }
    (destination / "atlas-machine.mlir").write_text(machine)
    (destination / "program.bin").write_bytes(binary)
    for name, record in (("execution_plan.json", execution), ("physical-program.json", program)):
        (destination / name).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    artifact_names = ("atlas-virtual.mlir", "atlas-machine.mlir", "program.bin", "physical-program.json",
                      "execution_plan.json", "compilation-pipeline.json")
    if (destination / "mandatory-spilled-virtual.mlir").is_file():
        artifact_names += ("mandatory-spilled-virtual.mlir",)
    manifest = {
        "schema": "atlas.region_composition.v1", "artifact_kind": "program",
        "selected_rtl_revision": RTL_REVISION, "program_bytes": len(binary),
        "launches_per_invocation": 1, "stage_count": len(stages),
        "spill_stores": program.get("spill_stores", 0), "spill_reloads": program.get("spill_reloads", 0),
        "files": {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in artifact_names},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def allocate_adjacent_arena(sizes, names):
    """Place whole tile buffers in one checked off-chip application arena."""
    from .binding import DRAM_BASE, TILE_BYTES, dram_slots

    addresses, cursor = {}, DRAM_BASE
    for name in names:
        if name in addresses or sizes[name] <= 0 or sizes[name] % TILE_BYTES:
            raise ValueError("adjacent arena needs distinct complete tile buffers")
        addresses[name] = cursor
        cursor += sizes[name]
    if cursor - DRAM_BASE > dram_slots() * TILE_BYTES:
        raise ValueError("adjacent arena exceeds the selected off-chip window")
    return addresses, cursor - DRAM_BASE


def compile_pointwise_member(precision, plan, snapshot, *, addresses, output,
                             target_source, destination, limits, ordinary_abi=False):
    """Qualify an unchanged child and return its selected body plus receipt."""
    from .binding import DRAM_BASE, TILE_BYTES
    from .pointwise import compile_pointwise, plan_abi, selected_body_descriptor

    abi = plan_abi(precision.request) if ordinary_abi else dict(
        fixed_inputs={node.id: (address - DRAM_BASE) // TILE_BYTES
                      for node, address in zip(precision.request.nodes[:-1], addresses, strict=True)},
        fixed_outputs=((output - DRAM_BASE) // TILE_BYTES,))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    compile_pointwise(snapshot, precision.request, fixed_inputs=abi['fixed_inputs'],
        fixed_outputs=tuple(abi['fixed_outputs']), target_source=target_source,
        destination=destination, limits=limits)
    return selected_body_descriptor(plan, destination / 'body'), dict(
        request_digest=precision.request.digest(), body_digest=plan.body_request().digest(),
        manifest_sha256=hashlib.sha256((destination / 'manifest.json').read_bytes()).hexdigest())


def compile_adjacent_virtual(virtual, *, destination, target_source):
    """Lower one linked body through the selected registered compiler pipeline."""
    from .binding import NATIVE_CHECKS, NATIVE_PIPELINES, RTL_REVISION, _native_pipeline, _run

    tools = Path(os.environ.get('ATLAS_OOT_BIN_DIR', Path(target_source) / 'build/bin'))
    machine = compile_virtual_with_receipt(virtual, tools=tools, pipeline=_native_pipeline(), destination=destination)
    _run([str(tools / 'atlas-opt'), *NATIVE_CHECKS, '--verify-atlas-timing', '-'], input_text=machine)
    program = json.loads(_run([str(tools / 'atlas-emit'), '--program-json', '-'], input_text=machine))
    traps = [row for row in program['instructions'] if row['operation'] == 'atlas.trap']
    if (program.get('selected_rtl_revision') != RTL_REVISION or len(traps) != 1
            or traps[0]['fields']['kind'] != 'ecall'):
        raise ValueError('adjacent composition must retain selected RTL and one completion')
    destination = Path(destination)
    (destination / 'atlas-virtual.mlir').write_text(virtual)
    (destination / 'atlas-machine.mlir').write_text(machine)
    (destination / 'physical-program.json').write_text(json.dumps(program, sort_keys=True, indent=2) + '\n')
    (destination / 'program.bin').write_bytes(b''.join(
        row['word_u32'].to_bytes(4, 'little') for row in program['instructions']))


def link_adjacent_stages(stages, *, buffers, initialized, outputs, destination, target_source):
    """Compose checked explicit-buffer members and retain their final image."""
    import shutil
    from .binding import TILE_BYTES, _native_pipeline, dram_slots

    destination = Path(destination)
    linked = destination / 'linked'
    compose_regions(stages, buffers=buffers, initialized=initialized, outputs=outputs,
        destination=linked, tools=os.environ.get('ATLAS_OOT_BIN_DIR', str(Path(target_source) / 'build/bin')),
        pipeline=_native_pipeline(), io_memory_bytes=dram_slots() * TILE_BYTES, rematerialize_constants=True)
    for path in linked.iterdir():
        shutil.copyfile(path, destination / path.name)
    shutil.rmtree(linked)


def publish_adjacent_group(group, destination, *, kind, execution, enlargement):
    """Bind the exact source, proof and every artifact to one native image."""
    from .binding import TILE_BYTES, dram_slots
    from .resident_graph import _artifact_files

    destination = Path(destination)
    region = group['region']
    for name, record in (('source-request.json', region.record()), ('execution_plan.json', execution),
                         ('enlargement.json', enlargement)):
        (destination / name).write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    artifacts = {path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in _artifact_files(destination) if path != destination / 'manifest.json'}
    manifest = dict(schema='atlas.' + kind + '.v1', engine='merlin_native', artifact_kind='program',
        request_digest=region.digest(), target_identity=region.target_identity,
        binary_sha256=artifacts['program.bin'], execution_plan_sha256=artifacts['execution_plan.json'],
        stage_artifacts=artifacts, io_memory_bytes=dram_slots() * TILE_BYTES)
    (destination / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return manifest


def verify_adjacent_group(group, compiled, *, precision_plan=None, plans=None):
    from merlin.semantic_compiler.target_binding import verify_native_publication

    compiled, region = Path(compiled), group['region']
    manifest = json.loads((compiled / 'manifest.json').read_text())
    verify_native_publication(compiled, manifest, engine='merlin_native',
                              request_digest=region.digest(), target_identity=region.target_identity)
    record = json.loads((compiled / 'enlargement.json').read_text())
    expected = dict(proof=group['proof'], source_request_digest=region.digest())
    if precision_plan is not None:
        expected['precision_plan'] = precision_plan
    if plans is not None:
        expected['plans'] = plans
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError('adjacent source or intermediate conversion differs from publication')
    return json.loads((compiled / 'execution_plan.json').read_text())


def export_adjacent_package(group, compiled, name, *, kind, target_source, execution,
                            input_sources, packed_sizes, helpers, metadata=None,
                            completion_check='', source_input_indices=None):
    """Export shared checked f32 boundary dispatch around a single device image."""
    from merlin.compile.semantic_application import CompiledCRegion
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType
    from .c_runtime import RUNTIME_SHARED_FILES
    from .tile_io_c import render_tile_io_package

    region, compiled = group['region'], Path(compiled)
    nodes = tuple(SemanticNode('input' + str(index), 'input', (),
        TensorType(region.node(source).type.shape, 'bf16', 'raw-bit-pattern'), effect='input')
        for index, source in enumerate(input_sources))
    result = SemanticNode('result', 'identity', tuple(node.id for node in nodes),
        TensorType(region.node(region.outputs[0]).type.shape, 'bf16', 'raw-bit-pattern'))
    boundary = KernelRequest((*nodes, result), ('result',), ('external',),
        tuple((node.id, 'external') for node in nodes), region.target_identity, 'diagnostic', region.digest())
    options = {} if source_input_indices is None else dict(source_input_indices=source_input_indices)
    record, files = render_tile_io_package(boundary, (compiled / 'program.bin').read_bytes(), execution, name,
        target_source=Path(target_source), runtime_linkage='separate', packed_sizes=packed_sizes,
        tile_bytes=2048, helpers='\n'.join(helpers), source_dtype='f32', completion_check=completion_check, **options)
    record.update(kind=kind, context_type=name + '_context', source_context_type=name + '_source_context',
        binding_function=name + '_bind', source_binding_function=name + '_source_bind',
        input_node_ids=list(group['inputs']), output_node_ids=list(region.outputs),
        intermediate_domain_proof=group['proof'], source_request_digest=region.digest(), launches_per_invocation=1,
        **(metadata or {}))
    return CompiledCRegion(name + '_source_dispatch', name + '.h', group['inputs'], region.outputs,
        files, RUNTIME_SHARED_FILES, ('segments/' + name,), record)


def compile_virtual_with_receipt(virtual, *, tools, pipeline, destination):
    """Record one bounded compile; opt in to post-spill audit capture explicitly."""
    from time import perf_counter
    from .binding import NATIVE_PIPELINES, _run

    destination, tools = Path(destination), Path(tools)
    capture = os.environ.get('ATLAS_CAPTURE_POSTSPILL') == '1'
    capture_options = ['--mlir-disable-threading',
        '--mlir-print-ir-before=lower-atlas-virtual-to-machine',
        '--mlir-print-ir-module-scope'] if capture else []
    command = [str(tools / 'atlas-opt'), *capture_options, *NATIVE_PIPELINES[pipeline], '-']
    started, diagnostics = perf_counter(), []
    machine = _run(command, input_text=virtual, diagnostics=diagnostics)
    receipt = dict(argv=command, elapsed_s=perf_counter() - started,
        input_sha256=hashlib.sha256(virtual.encode()).hexdigest(),
        output_sha256=hashlib.sha256(machine.encode()).hexdigest(),
        verifier_limits='compiler defaults', wall_limit_s=30,
        tool_sha256={name: hashlib.sha256((tools / name).read_bytes()).hexdigest()
                     for name in ('atlas-opt', 'atlas-emit')})
    if capture:
        diagnostic_text = ''.join(diagnostics)
        marker = '// -----// IR Dump Before'
        if diagnostic_text.count(marker) != 1:
            raise ValueError('single composition compile did not capture one post-spill module')
        spilled = diagnostic_text[diagnostic_text.index('\n', diagnostic_text.index(marker)) + 1:].strip() + '\n'
        (destination / 'mandatory-spilled-virtual.mlir').write_text(spilled)
        receipt['spilled_virtual_sha256'] = hashlib.sha256(spilled.encode()).hexdigest()
    (destination / 'compilation-pipeline.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return machine
