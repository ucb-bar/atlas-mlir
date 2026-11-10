"""Atlas-owned exact-float rebinding of checked canonical materializations.

Only machine artifacts with byte-identical source-generated virtual programs
are reused. Original requests, schedules, plans, manifests and C exports are
regenerated structurally, then authenticated by the ordinary source replay.
"""

from hashlib import sha256
from functools import lru_cache
import json
import os
from pathlib import Path

from merlin.semantic_compiler.target_binding import verify_native_publication

from . import exact_float_application as application
from . import exact_float_device as device


@lru_cache(maxsize=512)
def _source_digest(path, device_id, inode, size, modified, changed):
    return sha256(Path(path).read_bytes()).hexdigest()


def provider_sources():
    """Content identity for every Python provider component, before reuse."""
    root = Path(__file__).parent
    result = {}
    for directory, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if name != "__pycache__")
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("provider source identity requires regular source files")
            stat = path.stat()
            result[path.relative_to(root).as_posix()] = _source_digest(
                str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return result


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _publish_leaf(original, canonical, materialized, destination):
    plan, checked = device.prepare_plan(original), device.prepare_plan(canonical)
    manifest = json.loads((materialized / "manifest.json").read_bytes())
    verify_native_publication(materialized, manifest, engine="merlin_native",
                              request_digest=canonical.digest(), target_identity=canonical.target_identity)
    virtual = device.virtual_program(plan).encode()
    if (virtual != device.virtual_program(checked).encode()
            or virtual != (materialized / "atlas-virtual.mlir").read_bytes()
            or json.loads((materialized / "source-request.json").read_bytes()) != canonical.record()
            or json.loads((materialized / "float-plan.json").read_bytes()) != checked.record()
            or json.loads((materialized / "execution_plan.json").read_bytes()) != checked.record()):
        raise ValueError("canonical exact float materialization differs from source")
    # Reuse these checked target outputs; never rewrite opaque strings or copy
    # request-bound plans/manifests from another source boundary.
    files = {name: (materialized / name).read_bytes() for name in (
        "atlas-machine.mlir", "physical-program.json", "program.bin", "float-instruction-bound.json")}
    files.update({"source-request.json": _json(original.record()), "float-plan.json": _json(plan.record()),
                  "execution_plan.json": _json(plan.record()), "atlas-virtual.mlir": virtual})
    artifacts = {name: sha256(data).hexdigest() for name, data in files.items()}
    publication = dict(schema="atlas.exact_float_compilation.v1", engine="merlin_native", artifact_kind="program",
                       request_digest=original.digest(), target_identity=original.target_identity, compile_mode="diagnostic",
                       qualification_scope="diagnostic_only", binary_sha256=artifacts["program.bin"],
                       execution_plan_sha256=artifacts["execution_plan.json"], stage_artifacts=artifacts,
                       program_bytes=len(files["program.bin"]), launches_per_invocation=1,
                       io_memory_bytes=plan.descriptor()["io_memory_bytes"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_bytes(_json(publication))
    verify_native_publication(destination, publication, engine="merlin_native",
                              request_digest=original.digest(), target_identity=original.target_identity)
    return publication


def bind_exact_float(original, canonical, materialized, destination):
    """Publish a fresh original boundary against an authenticated checked body."""
    materialized, destination = Path(materialized), Path(destination)
    plan, checked = application.prepare_plan(original), application.prepare_plan(canonical)
    manifest = json.loads((materialized / "float-manifest.json").read_bytes())
    application._verify_publication(materialized, manifest, checked)
    if plan.launches != checked.launches or len(plan.bodies) != len(checked.bodies):
        raise ValueError("canonical exact float schedule differs from original source")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "segments").mkdir()
    records = {"source-request.json": original.record(), "float-schedule.json": plan.record()}
    for filename, record in records.items():
        (destination / filename).write_bytes(_json(record))
    segments = []
    for index, (body, canonical_body) in enumerate(zip(plan.bodies, checked.bodies, strict=True)):
        relative = f"segments/body_{index:03d}"
        detail = _publish_leaf(body, canonical_body, materialized / relative, destination / relative)
        segments.append(dict(index=index, path=relative, request_digest=body.digest(),
                             binary_sha256=detail["binary_sha256"],
                             manifest_sha256=sha256((destination / relative / "manifest.json").read_bytes()).hexdigest()))
    publication = dict(schema="atlas.exact_float_application.v1", artifact_kind="program_set", engine="merlin_native",
                       compile_mode="diagnostic", qualification_scope="diagnostic_only", request_digest=original.digest(),
                       target_identity=original.target_identity, segments=segments, launches_per_invocation=len(plan.launches),
                       execution_requires="ordered_exact_float_slices_then_atomic_publication",
                       stage_artifacts={name: sha256((destination / name).read_bytes()).hexdigest() for name in records})
    (destination / "float-manifest.json").write_bytes(_json(publication))
    application._verify_publication(destination, publication, plan)


def _checked_machine_files(original, canonical, materialized, *, original_plan, canonical_plan,
                           virtual, names, plan_files):
    """Reuse only immutable machine files after complete source correspondence."""
    manifest = json.loads((materialized / "manifest.json").read_bytes())
    verify_native_publication(materialized, manifest, engine="merlin_native",
                              request_digest=canonical.digest(), target_identity=canonical.target_identity)
    if (json.loads((materialized / "source-request.json").read_bytes()) != canonical.record()
            or virtual(original_plan) != virtual(canonical_plan)
            or virtual(original_plan).encode() != (materialized / "atlas-virtual.mlir").read_bytes()
            or any(json.loads((materialized / name).read_bytes()) != canonical_plan.record() for name in plan_files)):
        raise ValueError("canonical target materialization differs from complete source")
    files = {name: (materialized / name).read_bytes() for name in names}
    files.update({name: _json(original_plan.record()) for name in plan_files})
    files.update({"source-request.json": _json(original.record()), "atlas-virtual.mlir": virtual(original_plan).encode()})
    return manifest, files


def _write_files(destination, files, manifest):
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_bytes(_json(manifest))
    verify_native_publication(destination, manifest, engine="merlin_native",
                              request_digest=manifest["request_digest"], target_identity=manifest["target_identity"])
    return manifest


def _publish_layout_leaf(original, canonical, materialized, destination):
    from merlin.semantic_compiler.codegen import current_codegen_options
    from . import layout_device as layout
    from .binding import RTL_REVISION, _native_pipeline

    plan, checked = layout.prepare_source(original), layout.prepare_source(canonical)
    virtual = lambda p: layout.virtual_program(p, size_limit=current_codegen_options().program_size_limit_bytes)
    previous, files = _checked_machine_files(
        original, canonical, materialized, original_plan=plan, canonical_plan=checked, virtual=virtual,
        names=("atlas-machine.mlir", "physical-program.json", "program.bin", "layout-instruction-bound.json"),
        plan_files=("execution_plan.json",))
    if previous["lowering_pipeline"] != _native_pipeline() or previous["selected_rtl_revision"] != RTL_REVISION:
        raise ValueError("canonical layout lowering choices differ")
    hashes = {name: sha256(data).hexdigest() for name, data in files.items()}
    manifest = dict(schema="atlas.native_layout_device.v1", artifact_kind="program", engine="merlin_native",
                    request_digest=original.digest(), target_identity=original.target_identity,
                    binary_sha256=hashes["program.bin"], selected_rtl_revision=RTL_REVISION,
                    lowering_pipeline=_native_pipeline(), execution_plan_sha256=hashes["execution_plan.json"],
                    stage_artifacts=hashes)
    return _write_files(destination, files, manifest)


def bind_layout(original, canonical, materialized, destination):
    from . import layout_device as layout
    from . import layout_prefix_application as prefix

    materialized, destination = Path(materialized), Path(destination)
    plan = prefix.prepare_plan(layout.prepare_source(original))
    checked = prefix.prepare_plan(layout.prepare_source(canonical))
    if (plan is None) != (checked is None):
        raise ValueError("canonical layout envelope differs")
    if plan is None:
        return _publish_layout_leaf(original, canonical, materialized, destination)
    prefix._verify(materialized, canonical, checked)
    if len(plan.bodies) != len(checked.bodies):
        raise ValueError("canonical layout body inventory differs")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "segments").mkdir()
    records = {"source-request.json": original.record(), "layout-prefix-schedule.json": plan.record()}
    for filename, record in records.items():
        (destination / filename).write_bytes(_json(record))
    segments = []
    for index, (body, canonical_body) in enumerate(zip(plan.bodies, checked.bodies, strict=True)):
        relative = f"segments/body_{index:03d}"
        detail = _publish_layout_leaf(body, canonical_body, materialized / relative, destination / relative)
        segments.append(dict(index=index, path=relative, request_digest=body.digest(),
                             binary_sha256=detail["binary_sha256"],
                             manifest_sha256=sha256((destination / relative / "manifest.json").read_bytes()).hexdigest()))
    manifest = dict(schema="atlas.layout_prefix_application.v1", artifact_kind="program_set", engine="merlin_native",
                    compile_mode="diagnostic", qualification_scope="diagnostic_only", request_digest=original.digest(),
                    target_identity=original.target_identity, segments=segments, launches_per_invocation=len(plan.launches),
                    execution_requires="complete_matrix_prefix_slices_then_atomic_publication",
                    stage_artifacts={name: sha256((destination / name).read_bytes()).hexdigest() for name in records})
    (destination / "layout-prefix-manifest.json").write_bytes(_json(manifest))
    prefix._verify(destination, original, plan)
    return manifest


def _publish_quantizer_leaf(original, canonical, materialized, destination):
    from . import activation_quantize_device as quantizer
    from .binding import _native_pipeline

    plan, checked = quantizer.prepare_plan(original), quantizer.prepare_plan(canonical)
    previous, files = _checked_machine_files(
        original, canonical, materialized, original_plan=plan, canonical_plan=checked, virtual=quantizer.virtual_program,
        names=("atlas-machine.mlir", "physical-program.json", "program.bin",
               "quantizer-instruction-bound.json"),
        plan_files=("execution_plan.json", "quantizer-plan.json"))
    if previous["lowering_pipeline"] != _native_pipeline():
        raise ValueError("canonical quantizer lowering choices differ")
    hashes = {name: sha256(data).hexdigest() for name, data in files.items()}
    manifest = dict(schema="atlas.activation_quantize_compilation.v1", engine="merlin_native", artifact_kind="program",
                    request_digest=original.digest(), target_identity=original.target_identity, compile_mode="diagnostic",
                    qualification_scope="diagnostic_only", binary_sha256=hashes["program.bin"],
                    execution_plan_sha256=hashes["execution_plan.json"], stage_artifacts=hashes,
                    program_bytes=len(files["program.bin"]), lowering_pipeline=_native_pipeline(),
                    launches_per_invocation=1, io_memory_bytes=plan.descriptor()["io_memory_bytes"])
    return _write_files(destination, files, manifest)


def bind_quantizer(original, canonical, materialized, destination):
    from . import activation_quantize_application as quantizer

    materialized, destination = Path(materialized), Path(destination)
    plan, checked = quantizer.prepare_plan(original), quantizer.prepare_plan(canonical)
    previous = json.loads((materialized / "quantizer-manifest.json").read_bytes())
    quantizer._verify_publication(materialized, previous, checked)
    if len(plan.bodies) != len(checked.bodies) or plan.launches != checked.launches:
        raise ValueError("canonical quantizer schedule differs")
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "segments").mkdir()
    records = {"source-request.json": original.record(), "quantizer-schedule.json": plan.record()}
    for filename, record in records.items():
        (destination / filename).write_bytes(_json(record))
    segments = []
    for index, (body, canonical_body) in enumerate(zip(plan.bodies, checked.bodies, strict=True)):
        relative = f"segments/body_{index:03d}"
        detail = _publish_quantizer_leaf(body, canonical_body, materialized / relative, destination / relative)
        segments.append(dict(index=index, path=relative, request_digest=body.digest(), binary_sha256=detail["binary_sha256"],
                             manifest_sha256=sha256((destination / relative / "manifest.json").read_bytes()).hexdigest()))
    manifest = dict(schema="atlas.activation_quantize_application.v1", artifact_kind="program_set", engine="merlin_native",
                    compile_mode="diagnostic", qualification_scope="diagnostic_only", request_digest=original.digest(),
                    target_identity=original.target_identity, segments=segments, launches_per_invocation=len(plan.launches),
                    execution_requires="ordered_exact_quantizer_slices",
                    stage_artifacts={name: sha256((destination / name).read_bytes()).hexdigest() for name in records})
    (destination / "quantizer-manifest.json").write_bytes(_json(manifest))
    quantizer._verify_publication(destination, manifest, plan)
    return manifest
