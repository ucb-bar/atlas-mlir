"""Real exact-float native template, checked source rebinding and refusals."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native import exact_float_application, exact_float_device
from atlas_mlir_native.options import CodegenOptions, codegen_options
from test_exact_float_device import request

ROOT = Path(__file__).resolve().parents[1]
CHOICE = SimpleNamespace(metadata=(("kind", "exact_float_device"),), placement="accelerator")


def rename(source):
    names = {node.id: "current_" + node.id for node in source.nodes}
    nodes = tuple(replace(node, id=names[node.id], inputs=tuple(names[x] for x in node.inputs)) for node in source.nodes)
    return replace(source, nodes=nodes, outputs=tuple(names[x] for x in source.outputs),
                   input_storages=tuple((names[key], storage) for key, storage in source.input_storages),
                   source_identity="different capture provenance"), names


@pytest.fixture(scope="module")
def native_template(tmp_path_factory):
    tools = Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
    assert (tools / "atlas-opt").is_file() and (tools / "atlas-emit").is_file(), "native tools required; do not skip"
    segments = tmp_path_factory.mktemp("exact-loop-template") / "segments"
    source, options = request("bf16_neg", 3), CodegenOptions()
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    with codegen_options(options):
        package = adapter.compile_region(None, source, CHOICE, "region_000", segments, ROOT, None)
    return source, options, package, segments


def captured(native_template):
    source, options, _, segments = native_template
    adapter = AtlasApplicationAdapter.__new__(AtlasApplicationAdapter)
    with codegen_options(options):
        package = adapter.export_region(source, CHOICE, "region_000", segments, ROOT)
    return source, options, package, segments, adapter


def test_real_template_binds_current_source_without_selection_or_compilation(native_template, monkeypatch):
    source, options, package, segments, adapter = captured(native_template)
    assert package == native_template[2]  # Private capture changes no ordinary publication bytes.
    current, names = rename(source)
    leaf = segments / "region_000/segments/body_000/program.bin"
    binary = leaf.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("shared body binding must not select or compile a source")

    monkeypatch.setattr(adapter, "plan", forbidden)
    monkeypatch.setattr(adapter, "compile_region", forbidden)
    monkeypatch.setattr(exact_float_application, "compile_source", forbidden)
    monkeypatch.setattr(exact_float_device, "compile_source", forbidden)
    with codegen_options(options):
        bound = adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT)
        replayed = adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT)
    assert bound is not None and replayed == bound
    assert bound.files == package.files and bound.programs == package.programs
    assert bound.symbol == package.symbol and bound.header == package.header
    assert bound.inputs == tuple(names[key] for key in package.inputs)
    assert bound.outputs == tuple(names[key] for key in package.outputs)
    assert bound.description["float_schedule"]["source_request_digest"] == current.digest()
    assert bound.description["input_node_ids"] == list(bound.inputs)
    assert bound.description["native_manifest_sha256"] == package.description["native_manifest_sha256"]
    proof = bound.description["source_binding_proof"]
    assert proof["source_request_digest"] == current.digest()
    assert proof["template_source_request_digest"] == source.digest()
    assert proof["template_native_manifest_sha256"] == package.description["native_manifest_sha256"]
    assert len(proof["regenerated_metadata_sha256"]) == 64
    assert leaf.read_bytes() == binary
    assert bound.description["device_launch_schedule"]["images"] == [sha256(binary).hexdigest()]
    assert adapter._loop_body_packages[package.symbol]["root"] == (segments / "region_000").resolve()
    # Finalization ends the invocation-private proof lifetime.
    assert adapter.finalize_loop_regions((package, bound), target_source=ROOT) == (package, bound)
    assert adapter._loop_body_packages == {}
    with codegen_options(options):
        assert adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT) is None


def test_changed_source_options_correspondence_and_unissued_package_decline(native_template):
    source, options, package, segments, adapter = captured(native_template)
    current, names = rename(source)
    changed = replace(current, nodes=(*current.nodes[:-1],
                      replace(current.nodes[-1], attrs=(("numeric_semantics", "different policy"),))))
    with codegen_options(options):
        assert adapter.bind_loop_region(package, changed, CHOICE, names, segments, ROOT) is None
        assert adapter.bind_loop_region(package, current, CHOICE, {}, segments, ROOT) is None
        assert adapter.bind_loop_region(replace(package, files=dict(package.files)), current,
                                        CHOICE, names, segments, ROOT) is None
    # This option need not alter a three-element body: captured options still
    # prevent a cross-policy binding even if generated bytes happen to match.
    with codegen_options(replace(options, execution_instruction_limit=1_000_001)):
        assert adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT) is None
    with codegen_options(replace(options, resident_bindings={current.digest(): {"unproved": {}}})):
        assert adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT) is None


def test_nonidentical_regenerated_c_declines(native_template, monkeypatch):
    source, options, package, segments, adapter = captured(native_template)
    current, names = rename(source)
    exporter = adapter.export_region

    def different(*args, **kwargs):
        exported = exporter(*args, **kwargs)
        return replace(exported, files={**exported.files, exported.header: exported.files[exported.header] + b"\n"})

    monkeypatch.setattr(adapter, "export_region", different)
    with codegen_options(options):
        assert adapter.bind_loop_region(package, current, CHOICE, names, segments, ROOT) is None
    assert adapter._loop_body_packages[package.symbol]["files"] is package.files


def test_changed_template_identity_is_not_a_share_candidate(native_template):
    source, options, package, segments, adapter = captured(native_template)
    current, names = rename(source)
    forged = replace(package, description={**package.description, "native_manifest_sha256": "0" * 64})
    with codegen_options(options), pytest.raises(ValueError, match="publication identity"):
        adapter.bind_loop_region(forged, current, CHOICE, names, segments, ROOT)


def test_private_loop_receipts_do_not_change_materialization_identity(native_template):
    source, options, package, segments, adapter = captured(native_template)

    class Choice:
        def to_dict(self):
            return dict(id="choice", nodes=[], inputs=[], outputs=[], resident_inputs=[], resident_outputs=[])

    with codegen_options(options):
        original = adapter.materialization_identity(source, Choice())
        adapter._loop_body_packages = {"pending": object()}
        assert adapter.materialization_identity(source, Choice()) == original
        json.dumps(original, allow_nan=False)


def test_onchip_multileaf_and_resident_binders_decline_before_body_hook(native_template, monkeypatch):
    from atlas_mlir_native import loop_body_binding

    source, options, package, segments, adapter = captured(native_template)
    current, names = rename(source)

    def forbidden(*args, **kwargs):
        pytest.fail("onchip multileaf or resident body reached sharing hook")

    monkeypatch.setattr(loop_body_binding, "bind_replayed_loop_region", forbidden)
    with codegen_options(replace(options, application_deployment="onchip")):
        assert adapter.bind_loop_region(replace(package, programs=("one", "two")), current,
                                        CHOICE, names, segments, ROOT) is None
        resident = SimpleNamespace(placement="accelerator", metadata=(("kind", "already_quantized"),))
        assert adapter.bind_loop_region(package, current, resident, names, segments, ROOT) is None


def test_normal_deployment_shares_one_verified_native_image_with_distinct_contexts(tmp_path):
    from merlin.compiler.build.regions import compile_semantic_application, verify_semantic_application
    from atlas_test_fixtures.semantic_regions import plan_for
    from atlas_mlir_native.provider_deployment import WINDOW_ADDRESS, WINDOW_BYTES

    source = request("bf16_neg", 3)
    second = replace(source.nodes[-1], id="second", inputs=(source.outputs[0],))
    source = replace(source, nodes=(*source.nodes, second), outputs=(second.id,))
    plan = plan_for(source, ((0,), (1,)))
    plan = replace(plan, selected=tuple(replace(row, placement="accelerator", implementation="exact_float_device",
                                               metadata=(("kind", "exact_float_device"),)) for row in plan.selected))

    class Adapter(AtlasApplicationAdapter):
        compiles = 0

        def plan(self, request):
            return plan

        def compile_region(self, *args, **kwargs):
            self.compiles += 1
            return super().compile_region(*args, **kwargs)

    adapter = Adapter.__new__(Adapter)
    snapshot_root = tmp_path / "snapshot"
    snapshot_root.mkdir()
    (snapshot_root / "manifest.json").write_text("{}")
    snapshot = SimpleNamespace(root=snapshot_root)
    artifact = tmp_path / "application"
    options = CodegenOptions(application_deployment="onchip", io_memory_base=WINDOW_ADDRESS,
                             io_memory_bytes=WINDOW_BYTES, io_memory_capacity_base=WINDOW_ADDRESS,
                             io_memory_capacity_bytes=WINDOW_BYTES)
    with codegen_options(options):
        manifest = compile_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=ROOT, destination=artifact, limits=None, name="model", capacity=WINDOW_BYTES)
        checked = verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=ROOT, artifact=artifact)
    assert adapter.compiles == 1 and checked["status"] == "checked"
    assert manifest["dispatch_loop"] is True
    regions = json.loads((artifact / "compiled-regions.json").read_bytes())
    assert [row.get("compiled_body_index") for row in regions] == [None, 0]
    assert regions[0]["context_index"] == 0 and regions[1]["context_index"] == 1
    assert regions[0]["native_programs"] == regions[1]["native_programs"]
    assert len(manifest["segments"]) == 1
    deployment = json.loads((artifact / "deployment.json").read_bytes())
    assert [row["launches_per_invocation"] for row in deployment["calls"]] == [1, 1]
    assert len(deployment["provider"]["images"]) == 1
    reservations = deployment["provider"]["reservations"]
    assert reservations and all(WINDOW_ADDRESS <= row["address"] < row["address"] + row["bytes"]
                                <= WINDOW_ADDRESS + WINDOW_BYTES for row in reservations)
    header = (artifact / "handoff/model_combined.h").read_text()
    assert " region_0;" in header and " region_1;" in header
    assert adapter._loop_body_packages == {}
    # A claimed current binding proof cannot replace source replay: changing it
    # in persisted summaries must fail the application publication verifier.
    regions[1]["description"]["source_binding_proof"]["source_request_digest"] = "0" * 64
    summary_path = artifact / "compiled-regions.json"
    summary_path.write_text(json.dumps(regions))
    # Rehash both public inventories so refusal must come from fresh source
    # replay rather than merely the ordinary artifact checksum guard.
    manifest["regions"] = regions
    manifest["stage_artifacts"]["compiled-regions.json"] = sha256(summary_path.read_bytes()).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    with codegen_options(options), pytest.raises(ValueError, match="manifest differs from replayed region compilation"):
        verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
            target_source=ROOT, artifact=artifact)
