"""Independent source rebinding of retained native tile selection evidence."""

from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
from dataclasses import asdict

from merlin.semantic_compiler.target_binding import verify_native_publication
from merlin.semantic_compiler.verify import check_selection


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


class SelectionCapture:
    """Retain actual checked search results without changing selection behavior."""

    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.records = []

    def __getattr__(self, name):
        return getattr(self.snapshot, name)

    def select(self, request, **kwargs):
        result = self.snapshot.select(request, **kwargs)
        self.records.append((request, kwargs, result))
        return result


def supports_captured_tile(compiled, capture):
    """Decline staged/unknown native envelopes before rebinding any request."""
    compiled = Path(compiled)
    if not (compiled / "manifest.json").is_file() or not (compiled / "source-request.json").is_file():
        return False
    manifest = json.loads((compiled / "manifest.json").read_bytes())
    if manifest.get("schema") != "atlas.native_tile_compilation.v6":
        return False
    source = json.loads((compiled / "source-request.json").read_bytes())
    matching = [selected for request, _, selected in capture.records if request.record() == source]
    return (len(matching) == 1 and matching[0].status == "selected"
            and matching[0].check_fingerprint == manifest.get("native_check_fingerprint"))


def replay_selection(original, canonical, capture):
    """Recheck a retained candidate and allocation, without search or emission."""
    if replace(original, source_identity=canonical.source_identity).record() != canonical.record():
        raise ValueError("native selection rebinding requires identical named source semantics")
    records = [(kwargs, selected) for request, kwargs, selected in capture.records if request == canonical]
    if len(records) != 1:
        raise ValueError("canonical body needs exactly one retained selection")
    kwargs, selected = records[0]
    profile = capture.profile
    if selected.status != "selected":
        raise ValueError("canonical body has no retained checked selection")
    def check(request):
        return check_selection(
            request, profile.descriptors, selected.rules, selected.exploration, selected.candidate,
            selected.graph, selected.allocation, profile.banks, views=profile.views,
            fixed_inputs=kwargs.get("fixed_inputs"), fixed_outputs=kwargs.get("fixed_outputs"),
            reservations=kwargs.get("reservations", ()))
    before, rebound = check(canonical), check(original)
    if not before.valid or before.fingerprint != selected.check_fingerprint or not rebound.valid:
        raise ValueError("native candidate source correspondence replay failed")
    return selected, rebound.fingerprint


def bind_tile(original, canonical, materialized, destination, capture):
    """Replay an existing candidate against the original source; never search."""
    if replace(original, source_identity=canonical.source_identity).record() != canonical.record():
        raise ValueError("native body rebinding requires identical named source semantics")
    manifest = json.loads((materialized / "manifest.json").read_bytes())
    verify_native_publication(materialized, manifest, engine="merlin_native",
                              request_digest=canonical.digest(), target_identity=canonical.target_identity)
    if json.loads((materialized / "source-request.json").read_bytes()) != canonical.record():
        raise ValueError("canonical native body differs from its source")
    records = [(kwargs, selected) for request, kwargs, selected in capture.records if request == canonical]
    if len(records) != 1:
        raise ValueError("canonical native body needs exactly one retained selection")
    kwargs, selected = records[0]
    profile = capture.profile
    if selected.status != "selected" or selected.check_fingerprint != manifest["native_check_fingerprint"]:
        raise ValueError("canonical native body has no retained checked selection")
    def check(request):
        return check_selection(
            request, profile.descriptors, selected.rules, selected.exploration, selected.candidate,
            selected.graph, selected.allocation, profile.banks, views=profile.views,
            fixed_inputs=kwargs.get("fixed_inputs"), fixed_outputs=kwargs.get("fixed_outputs"),
            reservations=kwargs.get("reservations", ()))
    before, rebound = check(canonical), check(original)
    if (not before.valid or before.fingerprint != selected.check_fingerprint or not rebound.valid):
        raise ValueError("native candidate source correspondence replay failed")
    selection = json.loads((materialized / "native-selection.json").read_bytes())
    if (selection["request_digest"] != canonical.digest()
            or selection["check_fingerprint"] != before.fingerprint
            or selection["candidate_digest"] != selected.candidate.digest()):
        raise ValueError("canonical native selection publication differs from retained evidence")
    selection.update(request_digest=original.digest(), check_fingerprint=rebound.fingerprint)
    names = {*manifest["stage_artifacts"], "program.bin", "execution_plan.json"}
    files = {name: (materialized / name).read_bytes() for name in names}
    files.update({"source-request.json": _json(original.record()), "native-selection.json": _json(selection)})
    # Every remaining field describes the independently replayed, unchanged
    # selected candidate or immutable tool output, not the source identity.
    manifest = dict(manifest, request_digest=original.digest(), native_check_fingerprint=rebound.fingerprint,
                    stage_artifacts={name: sha256(files[name]).hexdigest() for name in manifest["stage_artifacts"]})
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_bytes((json.dumps(manifest, sort_keys=True) + "\n").encode())
    verify_native_publication(destination, manifest, engine="merlin_native",
                              request_digest=original.digest(), target_identity=original.target_identity)
    return manifest


def _inventory(root):
    files = {}
    for directory, directories, names in os.walk(root, followlinks=False):
        directories.sort()
        for name in names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("native materialization contains linked files")
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def bind_pointwise(original, canonical, materialized, destination, capture, *, reduction=False):
    from . import pointwise
    from . import reduction as reduction_module
    from .binding import AtlasNativeBinding, DRAM_BASE, TILE_BYTES, dram_slots

    module = reduction_module if reduction else pointwise
    if replace(original, source_identity=canonical.source_identity).record() != canonical.record():
        raise ValueError("loop rebinding requires identical named lowered semantics")
    previous = json.loads((materialized / "manifest.json").read_bytes())
    verify_native_publication(materialized, previous, engine="merlin_native",
                              request_digest=canonical.digest(), target_identity=canonical.target_identity)
    if json.loads((materialized / "source-request.json").read_bytes()) != canonical.record():
        raise ValueError("canonical loop source differs")
    abi = module.plan_abi(original)
    if reduction:
        plan, canonical_plan = module.prepare_plan(original), module.prepare_plan(canonical)
        body, canonical_body = module.body_request(plan), module.body_request(canonical_plan)
    else:
        full_plan, plan, streaming = module.prepare_execution_plan(original)
        _, canonical_plan, _ = module.prepare_execution_plan(canonical)
        body, canonical_body = plan.body_request(), canonical_plan.body_request()
    destination.mkdir(parents=True, exist_ok=False)
    checked_body = bind_tile(body, canonical_body, materialized / "body", destination / "body", capture)
    if reduction:
        slots = tuple(abi["fixed_inputs"][node.id] for node in original.nodes[:2]) + tuple(abi["fixed_outputs"])
        addresses = tuple(DRAM_BASE + slot * TILE_BYTES for slot in slots)
        descriptor = module.selected_body_descriptors(plan, destination / "body")
        virtual = module.virtual_program(plan, addresses[:2], addresses[2])
        execution = dict(schema="atlas.reduction_execution_plan.v1", input_addresses=list(addresses[:2]),
                         input_buffer_bytes=list(plan.packed_sizes[:2]), output_addresses=[addresses[2]],
                         output_bytes=plan.packed_sizes[2], output_physical_layouts=["row_column_zero_bf16_pair_tiles"],
                         tiling=plan.record())
        changed = {"source-request.json": original.record(), "execution_plan.json": execution,
                   "reduction-tiling.json": plan.record()}
        manifest = dict(schema="atlas.native_reduction_compilation.v1", artifact_kind="program", engine="merlin_native",
                        compile_mode="diagnostic", qualification_scope="diagnostic_only", request_digest=original.digest(),
                        target_identity=original.target_identity, binary_sha256=previous["binary_sha256"],
                        body_check_fingerprint=checked_body["native_check_fingerprint"],
                        selection_scope="representative_row_tree_and_seeded_combination", selected_body_descriptors=descriptor,
                        lowering_pipeline=previous["lowering_pipeline"], semantic_qualification_pipeline="mandatory",
                        native_snapshot_sha256=previous["native_snapshot_sha256"], io_memory_bytes=dram_slots() * TILE_BYTES,
                        reduction_lowering_source_sha256=sha256(Path(module.__file__).read_bytes()).hexdigest(),
                        target_inventory_sha256=previous["target_inventory_sha256"])
    else:
        addresses = [DRAM_BASE + abi["fixed_inputs"][node.id] * TILE_BYTES for node in original.nodes[:-1]]
        output = DRAM_BASE + abi["fixed_outputs"][0] * TILE_BYTES
        descriptor = module.selected_body_descriptor(plan, destination / "body")
        virtual = module.virtual_program(plan, addresses, output, descriptor)
        execution = dict(schema="atlas.pointwise_execution_plan.v1", shape=list(plan.shape), input_addresses=addresses,
                         input_buffer_bytes=list(plan.input_packed_bytes), output_addresses=[output], output_bytes=plan.packed_bytes,
                         output_physical_layouts=["mn_tiles_bf16_pair_halves"], tiling=plan.record(),
                         **({"streaming": streaming, "source_tiling": full_plan.record()} if streaming else {}))
        changed = {"source-request.json": original.record(), "execution_plan.json": execution,
                   "pointwise-tiling.json": full_plan.record()}
        manifest = dict(schema="atlas.native_pointwise_compilation.v1", artifact_kind="program", engine="merlin_native",
                        compile_mode="diagnostic", qualification_scope="diagnostic_only",
                        **({"execution_requires": "streamed_host_dispatch"} if streaming else {}),
                        request_digest=original.digest(), target_identity=original.target_identity,
                        binary_sha256=previous["binary_sha256"], body_check_fingerprint=checked_body["native_check_fingerprint"],
                        selected_body_descriptor=descriptor, lowering_pipeline=previous["lowering_pipeline"],
                        semantic_qualification_pipeline="mandatory", target_inventory_sha256=previous["target_inventory_sha256"],
                        pointwise_lowering_source_sha256=sha256(Path(module.__file__).read_bytes()).hexdigest(),
                        native_snapshot_sha256=previous["native_snapshot_sha256"], io_memory_bytes=dram_slots() * TILE_BYTES)
    if virtual.encode() != (materialized / "atlas-virtual.mlir").read_bytes():
        raise ValueError("rebound loop virtual program differs from checked canonical source")
    from .restoration_budget import instruction_bound
    from .limits import functional_instruction_limit
    from types import SimpleNamespace
    bound_name = "reduction-instruction-bound.json" if reduction else "pointwise-instruction-bound.json"
    saved_bound = json.loads((materialized / bound_name).read_text())
    cap = functional_instruction_limit(SimpleNamespace(functional_instruction_limit=saved_bound["max_instructions"]))
    bound = instruction_bound(virtual, (materialized / "atlas-machine.mlir").read_text(),
                              json.loads((materialized / "physical-program.json").read_text()), max_instructions=cap)
    if not bound["admitted"] or saved_bound != bound:
        raise ValueError("rebound loop instruction bound differs or exceeds its saved limit")
    for name, data in _inventory(materialized).items():
        if name.startswith("body/") or name == "manifest.json":
            continue
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(data)
    for name, record in changed.items():
        (destination / name).write_bytes(_json(record))
    files = _inventory(destination)
    manifest.update(execution_plan_sha256=sha256(files["execution_plan.json"]).hexdigest(),
                    stage_artifacts={name: sha256(data).hexdigest() for name, data in files.items() if name != "program.bin"})
    (destination / "manifest.json").write_bytes(_json(manifest))
    verify_native_publication(destination, manifest, engine="merlin_native",
                              request_digest=original.digest(), target_identity=original.target_identity)
    return manifest


def bind_row_loop(original, canonical, materialized, destination, capture, *, target_source):
    from . import row_loop
    from .binding import _run
    from merlin.semantic_compiler.materialization import canonical_request_record

    if (canonical_request_record(original, source_identity_policy="provenance")
            != canonical_request_record(canonical, source_identity_policy="provenance")):
        raise ValueError("row loop rebinding requires identical lowered semantics")
    previous = json.loads((materialized / "manifest.json").read_bytes())
    row_loop.verify_row_loop(canonical, materialized, target_source=target_source)
    plan, checked = row_loop.prepare_plan(original), row_loop.prepare_plan(canonical)
    body, canonical_body = plan.tiling.regions[0].request, checked.tiling.regions[0].request
    destination.mkdir(parents=True, exist_ok=False)
    bind_tile(body, canonical_body, materialized / "body", destination / "body", capture)
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", Path(target_source) / "build/bin"))
    explicit = _run([str(tools / "atlas-opt"), "--lower-atlas-virtual-boundary-to-dma", "-"],
                    input_text=(destination / "body/atlas-virtual.mlir").read_text())
    virtual = row_loop.virtual_program(plan, explicit)
    if virtual.encode() != (materialized / "atlas-virtual.mlir").read_bytes():
        raise ValueError("rebound row virtual program differs from checked canonical source")
    for name, data in _inventory(materialized).items():
        if name.startswith("body/") or name == "manifest.json":
            continue
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        (destination / name).write_bytes(data)
    for name, value in {"source-request.json": original.record(), "row-softmax-plan.json": plan.tiling.record(),
                        "execution_plan.json": plan.execution_record()}.items():
        (destination / name).write_bytes(_json(value))
    files = _inventory(destination)
    manifest = dict(schema="atlas.row_loop_compilation.v1", engine="merlin_native", artifact_kind="program",
                    compile_mode=original.lowering_policy, qualification_scope="diagnostic_only",
                    request_digest=original.digest(), target_identity=original.target_identity,
                    binary_sha256=previous["binary_sha256"], execution_plan_sha256=sha256(files["execution_plan.json"]).hexdigest(),
                    native_snapshot_sha256=previous["native_snapshot_sha256"], io_memory_bytes=previous["io_memory_bytes"],
                    lowering_pipeline=previous["lowering_pipeline"], body_manifest_sha256=sha256(files["body/manifest.json"]).hexdigest(),
                    target_inventory_sha256=previous["target_inventory_sha256"], atlas_opt_sha256=previous["atlas_opt_sha256"],
                    atlas_emit_sha256=previous["atlas_emit_sha256"], provider_sources_sha256=previous["provider_sources_sha256"],
                    stage_artifacts={name: sha256(data).hexdigest() for name, data in files.items()})
    (destination / "manifest.json").write_bytes(_json(manifest))
    row_loop.verify_row_loop(original, destination, target_source=target_source)
    return manifest


def bind_counted(original, canonical, materialized, destination, capture, *, target_source):
    from . import quantized_loop as loop
    from .quantized_loop_sizing import Sizing, publish_route, selected_route
    from .restoration_budget import instruction_bound, fallback_tiles
    from .exponent_restoration import exponent_geometry, fast_factor_range

    plan, checked = loop.prepare_source(original), loop.prepare_source(canonical)
    if selected_route(checked, materialized) != "counted":
        raise ValueError("only counted contraction materializations support binding")
    previous = json.loads((materialized / "manifest.json").read_bytes())
    layout = loop.execution_layout(plan)
    virtual = loop.virtual_source(plan, layout)
    if virtual.encode() != (materialized / "atlas-virtual.mlir").read_bytes():
        raise ValueError("rebound counted virtual program differs from checked canonical source")
    if json.loads((materialized / "source-request.json").read_bytes()) != canonical.record():
        raise ValueError("canonical counted source differs")
    selections = {}
    previous_selections = json.loads((materialized / "tile-body-selection.json").read_bytes())
    roles = [("reset", plan.reset, checked.reset), ("continue", plan.continuation, checked.continuation)]
    epilogue, canonical_epilogue = getattr(plan, "epilogue_precision", None), getattr(checked, "epilogue_precision", None)
    if (epilogue is None) != (canonical_epilogue is None):
        raise ValueError("canonical counted epilogue differs")
    if epilogue is not None:
        from .pointwise import prepare_plan
        roles.append(("epilogue", prepare_plan(epilogue.request).body_request(),
                      prepare_plan(canonical_epilogue.request).body_request()))
    for role, body, canonical_body in roles:
        selected, fingerprint = replay_selection(body, canonical_body, capture)
        instructions = [selected.rules.symbols[value.symbol].get("descriptor", {}).get("name")
                        for value in selected.graph.values if value.kind == "instruction"]
        expected = dict(request_digest=canonical_body.digest(), check_fingerprint=selected.check_fingerprint,
                        instructions=instructions, allocation=asdict(selected.allocation))
        if role == "epilogue":
            from .binding import AtlasNativeBinding, BF16_BINARY_MODES
            descriptor = next(item.name for item in AtlasNativeBinding.profile().descriptors
                              if item.computation == body.nodes[-1].op and item.name in BF16_BINARY_MODES)
            expected["descriptor"] = descriptor
        if json.loads(json.dumps(expected)) != previous_selections[role]:
            raise ValueError("canonical counted selection differs from retained evidence")
        selections[role] = dict(expected, request_digest=body.digest(), check_fingerprint=fingerprint)
    files = {name: data for name, data in _inventory(materialized).items()
             if name not in {"manifest.json", "quantized-sizing.json", "quantized-route.json"}}
    program = json.loads(files["physical-program.json"])
    bound = instruction_bound(virtual, files["atlas-machine.mlir"].decode(), program)
    if not bound["admitted"]:
        raise ValueError("rebound counted instruction bound exceeds unchanged policy")
    bound["scalar_fallback_tile_budget"] = fallback_tiles(plan)
    bound["restoration_admission"] = layout["restoration_admission"]
    guard = dict(schema="atlas.exponent_restoration_guard.v1", guard_block="restore_factor_done",
                 vector_block="restore_vector", scalar_block="restore_scalar", factor_body="restore_factor_body",
                 factor_done="restore_factor_done", exponent_modes=[geometry["mode"] for geometry in exponent_geometry(plan)],
                 exponent_roles=["lhs", "rhs"], scalar_factor_broadcast="sparse_pow2_s_minus_1_col_sum_then_row_sum",
                 join_block="restore_join", factor_header="restore_factor_header", factor_bound=8,
                 lhs_index_coefficient=4, lhs_index_offsets=[0, 1, 2, 3], rhs_index_coefficient=2,
                 rhs_index_offsets=[0, 1, 16, 17], per_operand_range=list(fast_factor_range(plan)),
                 raw_nonzero_abs_exp2_bounds=[-26, 32], exponent_pages=layout["exponent_addresses"])
    files.update({name: _json(record) for name, record in {
        "source-request.json": original.record(), "already-quantized-plan.json": plan.record(), "execution_plan.json": layout,
        "tile-body-selection.json": selections, "restoration-instruction-bound.json": bound, "restoration-guard.json": guard}.items()})
    manifest = dict(schema="atlas.quantized_loop_compilation.v1", artifact_kind="program", engine="merlin_native",
                    request_digest=original.digest(), target_identity=original.target_identity,
                    binary_sha256=sha256(files["program.bin"]).hexdigest(),
                    execution_plan_sha256=sha256(files["execution_plan.json"]).hexdigest(),
                    stage_artifacts={name: sha256(data).hexdigest() for name, data in files.items() if name != "program.bin"},
                    target_inventory_sha256=previous["target_inventory_sha256"], io_memory_bytes=previous["io_memory_bytes"],
                    atlas_opt_sha256=previous["atlas_opt_sha256"], atlas_emit_sha256=previous["atlas_emit_sha256"])
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (destination / name).write_bytes(data)
    (destination / "manifest.json").write_bytes(_json(manifest))
    sizing = json.loads((materialized / "quantized-sizing.json").read_bytes())
    if (sizing.get("source_request_digest") != canonical.digest()
            or sizing["identity"]["source_request_digest"] != canonical.digest()
            or sizing["identity"]["virtual_sha256"] != sha256(virtual.encode()).hexdigest()):
        raise ValueError("canonical counted sizing differs from checked source")
    sizing = dict(sizing, source_request_digest=original.digest(),
                  identity=dict(sizing["identity"], source_request_digest=original.digest()))
    publish_route(plan, destination, Sizing(None, _json(sizing)), target_source=target_source)
    selected_route(plan, destination)
    return manifest
