"""Complete contiguous matrix-prefix slices with atomic byte publication.

The original dense prefix is traversed in row-major order. Each child retains
whole last-two-axis matrices and the original transpose permutation or concat
axis. Child inputs and outputs are contiguous byte intervals; no element bits
are interpreted. Every distinct child still passes the complete native pipeline
and the independent default instruction bound once before it can be reused.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.target_binding import verify_native_publication
from . import layout_device as device

MAX_XLU_TILES = 256
MAX_CONCAT_BYTES = 524288


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


@dataclass(frozen=True)
class PrefixSchedule:
    plan: object
    bodies: tuple
    launches: tuple
    matrix_shapes: tuple
    prefix_shape: tuple

    def record(self):
        return dict(schema="atlas.layout_prefix_schedule.v1", source_request_digest=self.plan.source.digest(),
            operation=self.plan.output.op, original_attributes=dict(self.plan.output.attrs),
            original_index_maps=[mapping.record() for mapping in self.plan.output.index_maps],
            prefix_shape=list(self.prefix_shape), prefix_order="dense row-major complete matrices",
            input_matrix_shapes=[list(shape) for shape in self.matrix_shapes],
            output_matrix_shape=list(self.plan.output.type.shape[-2:]), element_bytes=self.plan.width,
            max_xlu_tiles=MAX_XLU_TILES, max_concat_bytes=MAX_CONCAT_BYTES,
            input_bytes=list(self.plan.input_bytes), output_bytes=self.plan.output_bytes,
            input_node_ids=[key for key, _ in self.plan.source.input_storages],
            output_node_ids=list(self.plan.source.outputs),
            bodies=[dict(index=index, path=f"segments/body_{index:03d}", request_digest=body.digest())
                    for index, body in enumerate(self.bodies)],
            launches=list(self.launches), launches_per_invocation=len(self.launches),
            runtime_arithmetic="none; exact byte intervals only", weight_conversion="none",
            publication="copy complete private output only after every child succeeds")


def prepare_plan(plan):
    """Return a bounded complete-prefix construction, or the ordinary leaf path."""
    shape = plan.output.type.shape
    if len(shape) < 3:
        return None
    prefix = tuple(shape[:-2])
    count = prod(prefix)
    if count <= 1 or any(tuple(node.type.shape[:-2]) != prefix for node in plan.operands):
        return None
    if (plan.output.op == "transpose" and plan.width == 1
            and len(plan.segments) == 1 and plan.segments[0]["mode"] == "xlu"):
        matrix_tiles = prod(shape[-2:]) // 1024
        capacity = MAX_XLU_TILES // matrix_tiles
    elif plan.output.op == "concat" and dict(plan.output.attrs)["axis"] >= len(shape) - 2:
        capacity = MAX_CONCAT_BYTES // (prod(shape[-2:]) * plan.width)
    else:
        return None
    # An indivisible matrix follows the normal leaf path, which independently
    # refuses an excessive instruction bound. It is never split to reset proofs.
    if capacity < 1 or count <= capacity:
        return None
    matrix_shapes = tuple(tuple(node.type.shape[-2:]) for node in plan.operands)
    constants = {binding.node_id: binding for binding in plan.source.constants}
    bodies, seen, launches = [], {}, []
    output_matrix_bytes = prod(shape[-2:]) * plan.width
    for first in range(0, count, capacity):
        length = min(capacity, count - first)
        child_prefix = (*((1,) * (len(prefix) - 1)), length)
        nodes = [replace(node, type=replace(node.type, shape=(*child_prefix, *matrix_shape)))
                 for node, matrix_shape in zip(plan.operands, matrix_shapes, strict=True)]
        output = replace(plan.output, type=replace(plan.output.type, shape=(*child_prefix, *shape[-2:])))
        child_constants = []
        spans, constant_spans = [], []
        for original, child, matrix_shape in zip(plan.operands, nodes, matrix_shapes, strict=True):
            matrix_bytes = prod(matrix_shape) * plan.width
            offset, size = first * matrix_bytes, length * matrix_bytes
            if child.effect == "constant":
                binding = constants[child.id]
                child_constants.append(replace(binding, data_hex=binding.data_hex[offset * 2:(offset + size) * 2]))
                constant_spans.append(dict(node_id=child.id, offset_bytes=offset, bytes=size,
                    source_sha256=sha256(bytes.fromhex(binding.data_hex)).hexdigest(),
                    child_sha256=sha256(bytes.fromhex(child_constants[-1].data_hex)).hexdigest()))
            else:
                spans.append(dict(node_id=child.id, offset_bytes=offset, bytes=size))
        body = replace(plan.source, nodes=(*nodes, output), constants=tuple(child_constants),
            source_identity="atlas.layout_prefix.v1:" + plan.source.digest())
        device.prepare_source(body)
        digest = body.digest()
        if digest not in seen:
            seen[digest] = len(bodies)
            bodies.append(body)
        launches.append(dict(body_index=seen[digest], first_matrix=first, matrix_count=length,
            inputs=spans, constants=constant_spans, output_offset_bytes=first * output_matrix_bytes,
            output_bytes=length * output_matrix_bytes))
    # Contiguity proves exhaustive coverage and disjointness without visiting
    # any data byte or weakening a compiler proof budget.
    position = 0
    for launch in launches:
        if launch["output_offset_bytes"] != position:
            raise ValueError("layout prefix intervals overlap or leave a gap")
        position += launch["output_bytes"]
    if position != plan.output_bytes:
        raise ValueError("layout prefix intervals do not cover the source output")
    return PrefixSchedule(plan, tuple(bodies), tuple(launches), matrix_shapes, prefix)


def compile_source(snapshot, source, schedule, *, target_source, destination, limits):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "segments").mkdir()
    records = {"source-request.json": source.record(), "layout-prefix-schedule.json": schedule.record()}
    for name, record in records.items():
        (destination / name).write_bytes(_json(record))
    segments = []
    for index, body in enumerate(schedule.bodies):
        relative = f"segments/body_{index:03d}"
        manifest = device.compile_source(snapshot, body, target_source=target_source,
            destination=destination / relative, limits=limits)
        segments.append(dict(index=index, path=relative, request_digest=body.digest(),
            binary_sha256=manifest["binary_sha256"],
            manifest_sha256=sha256((destination / relative / "manifest.json").read_bytes()).hexdigest()))
    manifest = dict(schema="atlas.layout_prefix_application.v1", artifact_kind="program_set",
        engine="merlin_native", compile_mode="diagnostic", qualification_scope="diagnostic_only",
        request_digest=source.digest(), target_identity=source.target_identity,
        segments=segments, launches_per_invocation=len(schedule.launches),
        execution_requires="complete_matrix_prefix_slices_then_atomic_publication",
        stage_artifacts={name: sha256((destination / name).read_bytes()).hexdigest() for name in records})
    (destination / "layout-prefix-manifest.json").write_bytes(_json(manifest))
    _verify(destination, source, schedule)
    return manifest


def _verify(root, source, schedule):
    root = Path(root)
    expected = {"source-request.json", "layout-prefix-schedule.json", "layout-prefix-manifest.json", "segments"}
    if root.is_symlink() or not root.is_dir() or {path.name for path in root.iterdir()} != expected:
        raise ValueError("layout prefix publication inventory differs")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected - {"segments"}):
        raise ValueError("layout prefix metadata is absent or linked")
    manifest = json.loads((root / "layout-prefix-manifest.json").read_text())
    if (manifest.get("schema") != "atlas.layout_prefix_application.v1"
            or manifest.get("artifact_kind") != "program_set" or manifest.get("engine") != "merlin_native"
            or manifest.get("compile_mode") != "diagnostic" or manifest.get("qualification_scope") != "diagnostic_only"
            or manifest.get("execution_requires") != "complete_matrix_prefix_slices_then_atomic_publication"
            or manifest.get("request_digest") != source.digest() or manifest.get("target_identity") != source.target_identity
            or manifest.get("launches_per_invocation") != len(schedule.launches)
            or json.loads((root / "source-request.json").read_text()) != source.record()
            or json.loads((root / "layout-prefix-schedule.json").read_text()) != schedule.record()):
        raise ValueError("layout prefix source or slice schedule differs")
    artifacts = manifest.get("stage_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"source-request.json", "layout-prefix-schedule.json"}:
        raise ValueError("layout prefix metadata map differs")
    for name, digest in artifacts.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("layout prefix metadata hash differs")
    segments = root / "segments"
    names = [f"body_{index:03d}" for index in range(len(schedule.bodies))]
    if segments.is_symlink() or not segments.is_dir() or {path.name for path in segments.iterdir()} != set(names):
        raise ValueError("layout prefix native body inventory differs")
    rows = manifest.get("segments")
    if not isinstance(rows, list) or len(rows) != len(names):
        raise ValueError("layout prefix native body count differs")
    for index, (name, body, row) in enumerate(zip(names, schedule.bodies, rows, strict=True)):
        leaf = segments / name
        if leaf.is_symlink() or not leaf.is_dir() or (leaf / "manifest.json").is_symlink():
            raise ValueError("layout prefix body is absent or linked")
        detail = json.loads((leaf / "manifest.json").read_text())
        artifacts = detail.get("stage_artifacts")
        if (not isinstance(artifacts, dict) or {path.name for path in leaf.iterdir()} != {*artifacts, "manifest.json"}
                or any((leaf / member).is_symlink() or not (leaf / member).is_file() for member in artifacts)):
            raise ValueError("layout prefix native body has unaccounted or linked members")
        if (row.get("index") != index or row.get("path") != "segments/" + name
                or row.get("request_digest") != body.digest()
                or row.get("manifest_sha256") != sha256((leaf / "manifest.json").read_bytes()).hexdigest()
                or row.get("binary_sha256") != detail.get("binary_sha256")):
            raise ValueError("layout prefix native body identity differs")
        verify_native_publication(leaf, detail, engine="merlin_native", request_digest=body.digest(),
            target_identity=source.target_identity)
    return manifest


def prepare_c_package(source, compiled, name, *, target_source, runtime_linkage="separate"):
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.launch_schedule import make_launch_schedule
    from .c_runtime import RUNTIME_SHARED_FILES
    from .slice_dispatch import render_slice_dispatch
    schedule = prepare_plan(device.prepare_source(source))
    if schedule is None or runtime_linkage != "separate":
        raise ValueError("layout prefix export requires the checked complete-prefix schedule")
    compiled = Path(compiled)
    manifest = _verify(compiled, source, schedule)
    leaves = [device.prepare_c_package(body, compiled / row["path"], name + f"_body_{index:03d}",
        target_source=target_source, runtime_linkage=runtime_linkage)
        for index, (body, row) in enumerate(zip(schedule.bodies, manifest["segments"], strict=True))]
    maximum = max(record["workspace_bytes"] for record, _ in leaves)
    workspace_bytes = maximum + schedule.plan.output_bytes
    upper = name.upper()
    fields = "\n".join(f"  {name}_body_{index:03d}_context body_{index}; {name}_body_{index:03d}_source_context source_{index};"
                       for index in range(len(leaves)))
    bind_calls = []
    for index in range(len(leaves)):
        inner = name + f"_body_{index:03d}"
        bind_calls.append(f"  if ({inner}_bind(&ctx->body_{index},host,io,workspace,{maximum}u,polls) || {inner}_source_bind(&ctx->source_{index},&ctx->body_{index},0,0)) return {upper}_INVALID;")
    inputs = [key for key, _ in source.input_storages]
    calls = []
    for launch in schedule.launches:
        index = launch["body_index"]
        inner = name + f"_body_{index:03d}"
        spans = {span["node_id"]: span for span in launch["inputs"]}
        child_inputs = [key for key, _ in schedule.bodies[index].input_storages]
        operands = ",".join(f"(const uint8_t *)inputs[{inputs.index(key)}]+{spans[key]['offset_bytes']}u" for key in child_inputs)
        calls.append(f"  {{ const void *body_inputs[{max(1,len(launch['inputs']))}]={{{operands or '0'}}}; void *body_outputs[1]={{region->workspace+{maximum+launch['output_offset_bytes']}u}}; int status={inner}_source_dispatch(&region->source_{index},body_inputs,body_outputs); if(status) return status; }}")
    declarations, implementation = render_slice_dispatch(name, workspace_bytes=workspace_bytes, fields=fields,
        bind_calls=bind_calls, input_bytes=[prod(source.node(key).type.shape) * schedule.plan.width for key in inputs],
        output_slices=[(maximum, schedule.plan.output_bytes)], launch_calls=calls)
    public, sources, files = {}, {}, {}
    for index, (_, members) in enumerate(leaves):
        for filename, data in members.items():
            if filename in RUNTIME_SHARED_FILES:
                if filename in files and files[filename] != data:
                    raise ValueError("layout prefix bodies disagree on shared runtime")
                files[filename] = data
            elif filename.endswith(".h"):
                public[filename] = data.decode()
            elif filename.endswith(".c"):
                sources[filename] = data.decode()
            elif filename == "program.bin":
                files[f"body_{index:03d}.bin"] = data
            else:
                raise ValueError("unexpected layout prefix C export member")
    sources["layout_prefix.c"] = implementation
    files.update({filename: data.encode() for filename, data in c_source_pair(name, public_headers=public,
        private_headers={}, sources=sources, declarations=declarations).items()})
    runs = []
    for launch in schedule.launches:
        image = launch["body_index"]
        if runs and runs[-1]["image"] == image:
            runs[-1]["count"] += 1
        else:
            runs.append(dict(image=image, count=1))
    launch_schedule = make_launch_schedule([row["binary_sha256"] for row in manifest["segments"]], [dict(repeat=1, runs=runs)])
    record = dict(kind="layout_device", schema="atlas.layout_prefix_c_package.v1", workspace_bytes=workspace_bytes,
        source_workspace_bytes=1, context_type=name + "_context", source_context_type=name + "_source_context",
        binding_function=name + "_bind", source_binding_function=name + "_source_bind",
        dispatch_function=name + "_source_dispatch", input_node_ids=inputs, output_node_ids=list(source.outputs),
        launches_per_invocation=len(schedule.launches), device_launch_schedule=launch_schedule,
        native_program_paths=[row["path"] for row in manifest["segments"]], layout_prefix_schedule=schedule.record(),
        weight_conversion="none", runtime_arithmetic="none", runtime_linkage="separate",
        runtime_source="runtime/atlas_host.c", native_manifest_sha256=sha256((compiled / "layout-prefix-manifest.json").read_bytes()).hexdigest())
    return record, files
