"""Target-owned exact quantizer scheduling with raw-byte application dispatch.

Static packing is independent per element. Dynamic packing is split only at
complete authored reduction-group boundaries. The original semantic request
and every source/body byte slice are authenticated by the program-set manifest.
Invocation C performs pointer checks, copies, padding, and device dispatch only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import prod
from pathlib import Path

from merlin.semantic_compiler.model import KernelRequest
from merlin.semantic_compiler.target_binding import verify_native_publication

from . import vector_quantize as device
from .binding import TARGET_IDENTITY, TILE_BYTES, dram_slots
from .limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT

STATIC_CHUNK_ELEMENTS = 2048
# A static body has no amax or per-row exponent packing. Even the worst
# 32-lane prefix path costs at most 50 instructions/tile plus 128 fixed:
# 50 * (2**19 / 32) + 128 = 819,328, below the default 1M bound. Unchanged
# npu_model measured 819,249 at count 524,256 (the largest gcd32 extent).
# Larger inputs or smaller I/O windows retain bounded chunk dispatch.
STATIC_WHOLE_BODY_MAX_ELEMENTS = 1 << 19
# Legacy measured scheduling estimate retained to preserve chunking and images.
# The current exceptional-lane loop has 25 words, so the older 119,525/group
# measurement is not a worst-case proof. Admission uses the replayed machine
# instruction bound; do not use this scheduling constant as that proof.
DYNAMIC_GROUP_WORST_INSTRUCTIONS = 119_525
DYNAMIC_GROUPS_PER_LAUNCH = DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT // DYNAMIC_GROUP_WORST_INSTRUCTIONS
# Counted full-tile groups scale with their number of 4,096-element blocks.
# Charge each block the complete measured single-group cost, including fixed
# exponent and launch work. This conservative envelope admits a single
# 32,768-element window512 head and preserves eight 4,096-element heads.
MAX_DYNAMIC_GROUP_ELEMENTS = 4096 * DYNAMIC_GROUPS_PER_LAUNCH


def _dynamic_group_instruction_bound(length):
    return DYNAMIC_GROUP_WORST_INSTRUCTIONS * ((length + 4095) // 4096)


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


@dataclass(frozen=True)
class QuantizerSchedule:
    source: KernelRequest
    x: object
    codes: object
    exponents: object
    attrs: dict
    bodies: tuple[KernelRequest, ...]
    launches: tuple[dict, ...]

    @property
    def count(self):
        return prod(self.x.type.shape)

    @property
    def width(self):
        return 2 if self.x.type.dtype == "bf16" else 4

    def record(self):
        return dict(schema="atlas.activation_quantize_schedule.v1", source_request_digest=self.source.digest(),
                    input_node_id=self.x.id, output_node_ids=list(self.source.outputs),
                    mode=self.attrs["mode"], reduce=self.attrs["reduce"],
                    input_bytes=self.count * self.width, code_bytes=self.count,
                    exponent_bytes=prod(self.exponents.type.shape) * 4,
                    static_chunk_elements=STATIC_CHUNK_ELEMENTS,
                    static_whole_body_max_elements=STATIC_WHOLE_BODY_MAX_ELEMENTS,
                    max_dynamic_group_elements=MAX_DYNAMIC_GROUP_ELEMENTS,
                    dynamic_groups_per_launch=DYNAMIC_GROUPS_PER_LAUNCH,
                    dynamic_group_worst_instructions=DYNAMIC_GROUP_WORST_INSTRUCTIONS,
                    dynamic_group_bound_block_elements=4096,
                    dynamic_group_bound_rule="119525 * ceil(group_elements / 4096)",
                    functional_instruction_limit=DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT,
                    bodies=[dict(index=index, path=f"segments/body_{index:03d}", request_digest=body.digest(),
                                 shape=list(body.node(self.x.id).type.shape)) for index, body in enumerate(self.bodies)],
                    launches=list(self.launches), launches_per_invocation=len(self.launches),
                    runtime_arithmetic="none; raw copies, padding, and device dispatch only",
                    dynamic_reduction="counted complete source groups per launch; no split amax",
                    weight_conversion="none")


def _body(source, x, codes, exponents, attrs, shape):
    exp_shape = (() if attrs["mode"] == "static" or attrs["reduce"] == "tensor" else
                 (*shape[:-2], 1, 1) if attrs["reduce"] == "head" else (*shape[:-1], 1))
    types = {x.id: replace(x.type, shape=shape), codes.id: replace(codes.type, shape=shape),
             exponents.id: replace(exponents.type, shape=exp_shape)}
    identity = sha256(_json(dict(source_request_digest=source.digest(), body_shape=shape))).hexdigest()
    body = replace(source, nodes=tuple(replace(node, type=types[node.id]) for node in source.nodes),
                   source_identity=identity)
    device.prepare_plan(body)
    return body


def prepare_plan(source):
    from . import dynamic_quantize, static_quantize

    code = next((node for node in source.nodes if node.op == "atlas.quantize_act.codes"), None)
    if code is None:
        raise ValueError("activation quantizer needs its code projection")
    mode = dict(code.attrs).get("mode")
    if mode == "static":
        x, codes, exponents, _ = static_quantize.prepare_source(source)
        attrs = dict(codes.attrs)
    elif mode == "dynamic":
        x, codes, exponents, attrs = dynamic_quantize.prepare_source(source)
    else:
        raise ValueError("unknown activation quantizer mode")
    if (source.target_identity != TARGET_IDENTITY or source.lowering_policy != "diagnostic"
            or source.constants or x.effect != "input" or x.op != "input"
            or source.input_storages != ((x.id, "external"),)
            or source.output_storages != ("external", "external")):
        raise ValueError("scheduled quantizer needs the selected target and dense external boundaries")
    count = prod(x.type.shape)
    if not 0 < count < 1 << 28:
        raise ValueError("scheduled quantizer extent exceeds bounded addressing")
    spans = []
    if mode == "static":
        chunk = (count if x.type.dtype == 'bf16' and count % 32 == 0
                 and count <= STATIC_WHOLE_BODY_MAX_ELEMENTS
                 and count * 3 + 32 <= dram_slots() * TILE_BYTES else STATIC_CHUNK_ELEMENTS)
        for first in range(0, count, chunk):
            length = min(chunk, count - first)
            spans.append((first, length, 0, 1, (length,)))
    else:
        k = x.type.shape[-1]
        total_rows = count // k
        m = x.type.shape[-2] if len(x.type.shape) > 1 else 1
        reduction, rows = attrs["reduce"], attrs["rows"]
        if reduction == "tensor":
            groups = [(0, total_rows)]
        elif reduction == "head":
            groups = [(first, first + m) for first in range(0, total_rows, m)]
        elif reduction == "token":
            groups = [(first, first + 1) for first in range(total_rows)]
        elif reduction == "block":
            groups = [(first, min(first + rows, total_rows)) for first in range(0, total_rows, rows)]
        else:
            groups = [(head + first, head + min(first + rows, m))
                      for head in range(0, total_rows, m) for first in range(0, m, rows)]
        group = 0
        while group < len(groups):
            first, end = groups[group]
            length = (end - first) * k
            if length > MAX_DYNAMIC_GROUP_ELEMENTS:
                raise ValueError(f"indivisible dynamic reduction group has {length} elements; selected device body supports at most {MAX_DYNAMIC_GROUP_ELEMENTS}")
            if length > 4096 and length % 1024:
                raise ValueError("indivisible dynamic reduction group above 4096 elements requires complete 1024-element tiles")
            packed = 1
            bounded_tiles = length % 1024 == 0 or length <= 1024
            if reduction == "head" and x.type.dtype == "bf16" and length % 32 == 0 and bounded_tiles:
                capacity = min(DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT // _dynamic_group_instruction_bound(length), 256,
                               max(1, (dram_slots() * TILE_BYTES - 1024) // (length * 3)))
                while (packed < capacity and group + packed < len(groups)
                       and groups[group + packed][1] - groups[group + packed][0] == end - first):
                    packed += 1
            exp_first, exp_count = ((group, packed) if reduction in ("tensor", "head") else (first, end - first))
            shape = ((length,) if reduction == "tensor" else
                     (packed, end - first, k) if reduction == "head" else (end - first, k))
            spans.append((first * k, length * packed, exp_first, exp_count, shape))
            group += packed
    bodies, shapes, launches = [], {}, []
    for first, length, exp_first, exp_count, shape in spans:
        if shape not in shapes:
            shapes[shape] = len(bodies)
            bodies.append(_body(source, x, codes, exponents, attrs, shape))
        launches.append(dict(body_index=shapes[shape], input_offset_bytes=first * (2 if x.type.dtype == "bf16" else 4),
                             input_bytes=length * (2 if x.type.dtype == "bf16" else 4),
                             code_offset_bytes=first, code_bytes=length,
                             exponent_offset_bytes=exp_first * 4, exponent_bytes=exp_count * 4))
    return QuantizerSchedule(source, x, codes, exponents, attrs, tuple(bodies), tuple(launches))


def compile_source(snapshot, region, *, target_source, destination, limits):
    plan = prepare_plan(region)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "segments").mkdir()
    records = {"source-request.json": region.record(), "quantizer-schedule.json": plan.record()}
    for filename, record in records.items():
        (destination / filename).write_bytes(_json(record))
    segments = []
    for index, body in enumerate(plan.bodies):
        relative = f"segments/body_{index:03d}"
        detail = device.compile_source(snapshot, body, target_source=target_source,
                                       destination=destination / relative, limits=limits)
        segments.append(dict(index=index, path=relative, request_digest=body.digest(),
                             binary_sha256=detail["binary_sha256"],
                             manifest_sha256=sha256((destination / relative / "manifest.json").read_bytes()).hexdigest()))
    manifest = dict(schema="atlas.activation_quantize_application.v1", artifact_kind="program_set",
                    engine="merlin_native", compile_mode="diagnostic", qualification_scope="diagnostic_only",
                    request_digest=region.digest(), target_identity=region.target_identity, segments=segments,
                    launches_per_invocation=len(plan.launches), execution_requires="ordered_exact_quantizer_slices",
                    stage_artifacts={filename: sha256((destination / filename).read_bytes()).hexdigest() for filename in records})
    (destination / "quantizer-manifest.json").write_bytes(_json(manifest))
    _verify_publication(destination, manifest, plan)
    return manifest


def _verify_publication(root, manifest, plan):
    """Authenticate the scheduling envelope and unchanged native leaf format.

    The application publication flattens the image leaves. Its branch inventory
    deliberately reserves manifest.json for actual image leaves, so this target
    envelope uses quantizer-manifest.json and performs the same immutable checks
    locally, including an exact flat inventory at every level we create.
    """
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("scheduled quantizer root is absent or linked")
    expected = {"source-request.json", "quantizer-schedule.json", "quantizer-manifest.json", "segments"}
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("scheduled quantizer root has an unaccounted member")
    if any((root / name).is_symlink() or not (root / name).is_file() for name in expected - {"segments"}):
        raise ValueError("scheduled quantizer metadata is absent or linked")
    if (json.loads((root / "quantizer-manifest.json").read_text()) != manifest
            or manifest.get("schema") != "atlas.activation_quantize_application.v1"
            or manifest.get("engine") != "merlin_native" or manifest.get("artifact_kind") != "program_set"
            or manifest.get("request_digest") != plan.source.digest()
            or manifest.get("target_identity") != plan.source.target_identity
            or manifest.get("launches_per_invocation") != len(plan.launches)):
        raise ValueError("scheduled quantizer manifest differs from checked source")
    artifacts = manifest.get("stage_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"source-request.json", "quantizer-schedule.json"}:
        raise ValueError("scheduled quantizer metadata inventory differs")
    for name, digest in artifacts.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("scheduled quantizer metadata digest differs")
    segments = root / "segments"
    names = [f"body_{index:03d}" for index in range(len(plan.bodies))]
    if segments.is_symlink() or not segments.is_dir() or {path.name for path in segments.iterdir()} != set(names):
        raise ValueError("scheduled quantizer body directory set differs")
    rows = manifest.get("segments")
    if not isinstance(rows, list) or len(rows) != len(names):
        raise ValueError("scheduled quantizer body manifest count differs")
    for index, (name, body, row) in enumerate(zip(names, plan.bodies, rows, strict=True)):
        leaf = segments / name
        if leaf.is_symlink() or not leaf.is_dir() or not isinstance(row, dict):
            raise ValueError("scheduled quantizer body is absent or linked")
        leaf_manifest = leaf / "manifest.json"
        if leaf_manifest.is_symlink() or not leaf_manifest.is_file():
            raise ValueError("scheduled quantizer leaf manifest is absent or linked")
        detail = json.loads(leaf_manifest.read_text())
        if (row.get("index") != index or row.get("path") != "segments/" + name
                or row.get("request_digest") != body.digest()
                or row.get("manifest_sha256") != sha256(leaf_manifest.read_bytes()).hexdigest()
                or row.get("binary_sha256") != detail.get("binary_sha256")):
            raise ValueError("scheduled quantizer leaf identity differs")
        members = detail.get("stage_artifacts")
        if not isinstance(members, dict) or {path.name for path in leaf.iterdir()} != {*members, "manifest.json"}:
            raise ValueError("scheduled quantizer leaf has unaccounted members")
        if any((leaf / member).is_symlink() or not (leaf / member).is_file() for member in members):
            raise ValueError("scheduled quantizer leaf contains a linked or absent file")
        verify_native_publication(leaf, detail, engine="merlin_native", request_digest=body.digest(), target_identity=plan.source.target_identity)


def prepare_c_package(region, compiled, name, *, target_source, runtime_linkage="separate"):
    from merlin.compile.c_sources import c_source_pair
    from merlin.compile.semantic_application import CompiledCRegion
    from .c_runtime import RUNTIME_SHARED_FILES

    if runtime_linkage != "separate":
        raise ValueError("scheduled quantizer requires shared runtime linkage")
    plan, compiled = prepare_plan(region), Path(compiled)
    manifest = json.loads((compiled / "quantizer-manifest.json").read_text())
    _verify_publication(compiled, manifest, plan)
    if (json.loads((compiled / "source-request.json").read_text()) != region.record()
            or json.loads((compiled / "quantizer-schedule.json").read_text()) != plan.record()):
        raise ValueError("scheduled quantizer source or slice mapping differs from publication")
    expected_paths = [f"segments/body_{index:03d}" for index in range(len(plan.bodies))]
    if [row["path"] for row in manifest["segments"]] != expected_paths:
        raise ValueError("scheduled quantizer body set differs from source schedule")
    lowered = [device.prepare_c_package(body, compiled / path, name + f"_body_{index:03d}",
                                       target_source=target_source, runtime_linkage=runtime_linkage)
               for index, (body, path) in enumerate(zip(plan.bodies, expected_paths, strict=True))]
    maximum = max(row.description["workspace_bytes"] for row in lowered)
    code_at, exp_at = maximum, maximum + plan.count
    exp_bytes = prod(plan.exponents.type.shape) * 4
    workspace_bytes = maximum + plan.count + exp_bytes
    upper = name.upper()
    fields = "\n".join(f"  {name}_body_{index:03d}_context body_{index}; {name}_body_{index:03d}_source_context source_{index};" for index in range(len(lowered)))
    bind_calls = []
    for index in range(len(lowered)):
        inner = name + f"_body_{index:03d}"
        bind_calls.append(f"  if ({inner}_bind(&ctx->body_{index},host,io,workspace,{maximum}u,polls) || {inner}_source_bind(&ctx->source_{index},&ctx->body_{index},0,0)) return {upper}_INVALID;")
    sizes = {plan.codes.id: plan.count, plan.exponents.id: exp_bytes}
    launch_code = []
    for call in plan.launches:
        index = call["body_index"]
        inner = name + f"_body_{index:03d}"
        output_values = {plan.codes.id: f"region->workspace+{code_at + call['code_offset_bytes']}u",
                         plan.exponents.id: f"region->workspace+{exp_at + call['exponent_offset_bytes']}u"}
        launch_code.append("  { const void *body_inputs[1]={(const uint8_t *)inputs[0]+" + str(call["input_offset_bytes"]) + "u}; void *body_outputs[2]={" + ",".join(output_values[key] for key in region.outputs) + f"}}; int result={inner}_source_dispatch(&region->source_{index},body_inputs,body_outputs); if (result) return result; }}")
    offsets = {plan.codes.id: code_at, plan.exponents.id: exp_at}
    from .slice_dispatch import render_slice_dispatch
    declarations, implementation = render_slice_dispatch(
        name, workspace_bytes=workspace_bytes, fields=fields, bind_calls=bind_calls,
        input_bytes=[plan.count * plan.width],
        output_slices=[(offsets[key], sizes[key]) for key in region.outputs], launch_calls=launch_code)
    public, sources, files = {}, {}, {}
    for index, item in enumerate(lowered):
        for filename, data in item.files.items():
            if filename in RUNTIME_SHARED_FILES:
                if filename in files and files[filename] != data:
                    raise ValueError("quantizer bodies disagree on shared runtime")
                files[filename] = data
            elif filename.endswith(".h"):
                public[filename] = data.decode()
            elif filename.endswith(".c"):
                sources[filename] = data.decode()
            elif filename == "program.bin":
                files[f"body_{index:03d}.bin"] = data
            else:
                raise ValueError("unexpected device quantizer C export member")
    sources["quantizer_schedule.c"] = implementation
    files.update({filename: data.encode() for filename, data in c_source_pair(name, public_headers=public,
                 private_headers={}, sources=sources, declarations=declarations).items()})
    from merlin.compile.launch_schedule import make_launch_schedule
    runs = []
    for call in plan.launches:
        index = call["body_index"]
        if runs and runs[-1]["image"] == index:
            runs[-1]["count"] += 1
        else:
            runs.append(dict(image=index, count=1))
    launch_schedule = make_launch_schedule([row["binary_sha256"] for row in manifest["segments"]],
                                          [dict(repeat=1, runs=runs)])
    description = dict(kind="activation_quantize_device", quantizer_kind=plan.attrs["mode"] + "_quantize",
                       context_type=name + "_context", source_context_type=name + "_source_context",
                       binding_function=name + "_bind", source_binding_function=name + "_source_bind",
                       dispatch_function=name + "_source_dispatch", workspace_bytes=workspace_bytes,
                       source_workspace_bytes=1, input_node_ids=[plan.x.id], output_node_ids=list(region.outputs),
                       device_launch_schedule=launch_schedule, launches_per_invocation=len(plan.launches),
                       quantizer_schedule=plan.record(), weight_conversion="none", runtime_arithmetic="none",
                       runtime_linkage="separate", runtime_source="runtime/atlas_host.c",
                       native_manifest_sha256=sha256((compiled / "quantizer-manifest.json").read_bytes()).hexdigest())
    return CompiledCRegion(name + "_source_dispatch", name + ".h", (plan.x.id,), tuple(region.outputs), files,
                           RUNTIME_SHARED_FILES, tuple("segments/" + name + "/" + path for path in expected_paths), description)
