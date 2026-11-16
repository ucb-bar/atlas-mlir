"""Deployment serialization and opt-in fixed-address application contexts."""
import hashlib
import json
import subprocess

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_c_context import application_binding
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError


BASE = 0x200C0000


def regions():
    return [dict(placement="accelerator", context_index=0, native_programs=["segments/r"], description=dict(
        context_type="r_context", source_context_type="r_source_context", binding_function="r_bind",
        source_binding_function="r_source_bind", workspace_bytes=4096, source_workspace_bytes=1024,
        execution_route="checked_counted_image")), dict(placement="host", description={}, native_programs=[])]


def interface():
    return dict(inputs=["a"], outputs=["y"], workspace_bytes=512)


def publish(root, plan):
    folder = root / "segments/r"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "execution_plan.json").write_text(json.dumps(plan))
    (folder / "physical-program.json").write_text(json.dumps(dict(
        schema="atlas.physical_program.v1", instructions=[dict(word_u32=0x73)])))
    (folder / "program.bin").write_bytes(b"\x73\0\0\0")


def counted():
    return dict(schema="atlas.quantized_loop_execution.v1", input_addresses=[BASE, BASE + 2048],
                input_buffer_bytes=[1024, 1024], exponent_addresses=[BASE + 4096, BASE + 6144],
                exponent_buffer_bytes=[2048, 2048], output_addresses=[BASE + 8192], output_bytes=2048,
                status_address=BASE + 10240, status_bytes=2048)


def requirements(root, rows=None):
    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432,
            io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)):
        return AtlasApplicationAdapter.deployment_requirements(rows or regions(), root)


def test_counted_reservations_and_relative_exact_image(tmp_path):
    publish(tmp_path, counted())
    result = requirements(tmp_path)
    assert result["schema"] == "merlin.provider_deployment.v1"
    assert result["window"] == dict(address=BASE, bytes=786432)
    assert [row["bytes"] for row in result["reservations"]] == [1024, 1024, 2048, 2048, 2048, 2048]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=0)
               and row["promotion_receipts"] == [] for row in result["reservations"])
    assert result["images"] == [dict(path="segments/r/program.bin", bytes=4,
        sha256=hashlib.sha256(b"\x73\0\0\0").hexdigest(), imem_offset_bytes=0,
        capacity_bytes=131072, upload_convention="counted_ensure_loaded")]
    assert (result["raw_scratch_bytes"], result["conversion_scratch_bytes"]) == (4096, 1024)
    assert result["regions"] == [dict(context_index=0, placement="accelerator", launches_per_invocation=1),
                                  dict(context_index=1, placement="host", launches_per_invocation=0)]
    assert result["host_map"] == dict(imem_address=0x20000, imem_bytes=131072, csr_address=0x40000)
    assert result["host_allocation"]["measurement"] == "declared_reservations"
    assert sum(row["bytes"] for row in result["host_allocation"]["declared_reservations"]) == 65536
    assert result["host_context_reservation_bytes"] == 4096
    assert result["allocation_alignment"] == 64
    assert result["imem_address_convention"] == "host_map.imem_address + image.imem_offset_bytes"
    assert result["runtime"]["image_completion"] == "ensure_loaded_before_start"
    assert result["runtime"]["reload_policy"] == "generated counted ensure_loaded"


def test_deployment_selection_is_explicit_and_default_identity_is_unchanged():
    from argparse import ArgumentParser
    from atlas_mlir_native.options import configure_codegen_parser, expand_codegen_arguments, options_from_args

    parser = ArgumentParser()
    configure_codegen_parser(parser)
    ordinary = parser.parse_args([])
    expand_codegen_arguments(ordinary)
    assert options_from_args(ordinary).application_deployment is None
    assert "application_deployment" not in options_from_args(ordinary).identity_record()
    chosen = parser.parse_args(["--application-deployment", "onchip"])
    expand_codegen_arguments(chosen)
    assert options_from_args(chosen).identity_record()["application_deployment"] == "onchip"
    with pytest.raises(ValueError, match="application deployment"):
        CodegenOptions(application_deployment="other")


def test_provider_declines_unsupported_geometry(tmp_path):
    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432)):
        assert AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path) is None


@pytest.mark.parametrize("description,expected", [
    ({"launches_per_invocation": 3}, 3),
    ({"streaming": {"launch_count": 4}}, 4),
    ({"native_export": {"streaming": {"launch_count": 5}}}, 5),
])
def test_provider_publishes_explicit_region_launch_counts(tmp_path, description, expected):
    publish(tmp_path, counted())
    rows = regions()
    rows[0]["description"].update(description)
    assert requirements(tmp_path, rows)["regions"][0]["launches_per_invocation"] == expected


def test_provider_refuses_invalid_region_launch_count(tmp_path):
    publish(tmp_path, counted())
    rows = regions()
    rows[0]["description"]["launches_per_invocation"] = True
    with pytest.raises(NativeCompilationError, match="explicit positive launch count"):
        requirements(tmp_path, rows)


def test_lifetime_and_promotion_join_preserves_shared_reservations(tmp_path):
    lifetime = dict(name="private", first_event=1, last_event=2, persistent=False)
    promotion = dict(schema="atlas.private_staging.v1", slot_addresses=[BASE], selected="vmem")
    plan = dict(schema="atlas.native_tile_execution_plan.v8", inputs=[], outputs=[dict(node="out")],
                buffers=[dict(node=node, address=BASE, bytes=2048) for node in ("private", "out")],
                source_liveness_buffers=[lifetime], private_staging=promotion)
    publish(tmp_path, plan)
    result = requirements(tmp_path)["reservations"]
    assert len(result) == 2
    assert result[0]["lifetime"] == lifetime and result[1]["lifetime"] is None
    assert all(row["promotion_receipts"] == [promotion] for row in result)
    assert result[0]["address"] == result[1]["address"] == BASE


def test_readonly_promotion_join_and_host_only_regions(tmp_path):
    promotion = dict(schema="atlas.private_staging.v2", slot_addresses=[], input_cache=dict(
        schema="atlas.readonly_staging.v1", cached_addresses=[BASE]))
    publish(tmp_path, dict(schema="atlas.native_tile_execution_plan.v8", inputs=[dict(node="a")],
        outputs=[], buffers=[dict(node="a", address=BASE, bytes=2048)], private_staging=promotion))
    assert requirements(tmp_path)["reservations"][0]["promotion_receipts"] == [promotion]
    host = requirements(tmp_path, [regions()[1]])
    assert host["images"] == host["reservations"] == []
    assert host["raw_scratch_bytes"] == host["conversion_scratch_bytes"] == 1


def test_pointwise_bias_frame_records_its_sequential_call(tmp_path):
    publish(tmp_path, counted())
    folder = tmp_path / "segments/bias"
    folder.mkdir()
    for member in ("program.bin", "physical-program.json"):
        (folder / member).write_bytes((tmp_path / "segments/r" / member).read_bytes())
    (folder / "execution_plan.json").write_text(json.dumps(dict(
        schema="atlas.pointwise_execution_plan.v1", input_addresses=[BASE, BASE + 2048],
        input_buffer_bytes=[2048, 2048], output_addresses=[BASE + 4096], output_bytes=2048)))
    rows = regions()
    rows.append(dict(placement="accelerator", context_index=2, native_programs=["segments/bias"], description={}))
    result = requirements(tmp_path, rows)
    assert len(result["images"]) == 2
    first, second = result["reservations"][:6], result["reservations"][6:]
    assert [row["address"] for row in second] == [BASE, BASE + 2048, BASE + 4096]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=0) for row in first)
    assert all(row["lifetime"] == dict(scope="region_call", context_index=2) for row in second)


@pytest.mark.parametrize("schema,output_bytes", [
    ("atlas.layout_device.v1", 1920), ("atlas.exact_float.v1", 1952),
])
def test_layout_and_exact_float_preserve_padded_published_payloads(tmp_path, schema, output_bytes):
    # Exact-float's combined output already contains its 32-byte status tail.
    # The provider must preserve that publication without counting it twice.
    plan = dict(schema=schema, input_addresses=[BASE], input_payload_bytes=[1920],
                input_logical_bytes=[1918], output_addresses=[BASE + 1920],
                output_bytes=output_bytes, logical_output_bytes=1918,
                status_address=BASE + 3840)
    publish(tmp_path, plan)
    result = requirements(tmp_path)
    assert [(row["address"], row["bytes"], row["role"]) for row in result["reservations"]] == [
        (BASE, 1920, "input"), (BASE + 1920, output_bytes, "output")]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=0)
               for row in result["reservations"])
    assert result["images"][0]["bytes"] == 4


@pytest.mark.parametrize("change,reason", [
    ({"schema": "atlas.future_execution.v2"}, "unsupported provider execution schema"),
    ({"status_address": BASE + 786432}, "outside selected window"),
    ({"resident_bindings": [dict(node_id="w")]}, "does not support resident bindings"),
])
def test_refuses_unsupported_or_invalid_execution(tmp_path, change, reason):
    publish(tmp_path, {**counted(), **change})
    with pytest.raises(NativeCompilationError, match=reason):
        requirements(tmp_path)


def test_refuses_changed_binary_and_noncounted_route(tmp_path):
    publish(tmp_path, counted())
    (tmp_path / "segments/r/program.bin").write_bytes(b"\0\0\0\0")
    with pytest.raises(NativeCompilationError, match="differs from physical program"):
        requirements(tmp_path)
    publish(tmp_path, counted())
    rows = regions()
    rows[0]["description"]["execution_route"] = "bounded_k32_tile_dispatch"
    with pytest.raises(NativeCompilationError, match="checked_counted_image"):
        requirements(tmp_path, rows)


@pytest.mark.parametrize("external,expected", [
    (False, ("4d985b2b2ace56d6439224b751d8493277187a135cf722380170c1ec52b8fcff",
             "80e0c2651cc53ad494023ff0cdd03bbbcec3f939eb0c703d728f197c39b6d33a")),
    (True, ("01ddf9f5054c3483a863beb0042f87277ef33e897c7b371cde84790dbe8b3dc7",
            "7191ff23b5caddde9b700aba9c8e60131258834b5ef72f71988cbbf41ef72e85")),
])
def test_default_binding_byte_identical(external, expected):
    # Frozen hashes of declarations and implementation at 256cb53.
    contract = interface()
    if external:
        contract.update(model_parameters=dict(payload_bytes=128), pointer_abi="opaque_native_dense_buffers")
    binding = application_binding("sample", regions(), contract)
    assert binding.interface["private_region_scratch_included"] is True
    assert "private_region_scratch_storage" not in binding.interface
    assert tuple(hashlib.sha256(text.encode()).hexdigest() for text in
                 (binding.declarations, binding.implementation)) == expected


def fixed_binding():
    contract = interface()
    contract["deployment"] = dict(provider=dict(allocation_alignment=64, host_context_reservation_bytes=4096),
        allocations=[dict(name=key, address=BASE + offset, bytes=size)
        for key, offset, size in (("device_scratch", 0, 4096), ("conversion_scratch", 4096, 1024),
                                 ("workspace", 8192, 512))])
    return application_binding("sample", regions(), contract)


def test_fixed_context_compiles_and_passes_buffer_sizes(tmp_path):
    binding = fixed_binding()
    assert binding.interface["host_context_reservation_bytes"] == 4096
    assert binding.interface["private_region_scratch_included"] is False
    assert binding.interface["private_region_scratch_storage"] == "published_deployment_window"
    assert "It points to private region scratch in the published deployment window." in binding.declarations
    assert "sizeof(ctx->device_scratch)" not in binding.implementation
    assert "sizeof(ctx->conversion_scratch)" not in binding.implementation
    assert "sizeof(ctx->workspace)" not in binding.implementation
    source = """
#include <stdint.h>
#include <stddef.h>
typedef struct { int active; } atlas_host;
typedef struct { int unused; } atlas_tile_memory_io;
typedef struct { int active; } sample_driver_state;
typedef struct { void **items; size_t count; } sample_driver_contexts;
typedef struct { int unused; } r_context;
typedef struct { int unused; } r_source_context;
static int r_bind(r_context *r,atlas_host *h,atlas_tile_memory_io io,void *p,size_t n,size_t polls) {
  (void)r;(void)h;(void)io;
  return !polls || (uintptr_t)p!=0x200c0000u || n!=4096u;
}
static int r_source_bind(r_source_context *s,r_context *r,void *p,size_t n) {
  (void)s;(void)r;return (uintptr_t)p!=0x200c1000u || n!=1024u;
}
static int sample_invoke(sample_driver_state *s,void *p,size_t n,const float *const *in,
                         float *const *out,sample_driver_contexts *dispatch) {
  (void)s;(void)in;(void)out;return (uintptr_t)p!=0x200c2000u || n!=512u || dispatch->count!=2;
}
""" + binding.declarations + binding.implementation + """
int main(void) {
  sample_context ctx={0};atlas_host host={0};atlas_tile_memory_io io={0};
  const void *inputs[]={0};void *outputs[]={0};
  return sample_bind(&ctx,&host,io,1) || sample_run(&ctx,inputs,outputs);
}
"""
    path = tmp_path / "check.c"
    path.write_text(source)
    command = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(tmp_path / "check")]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([str(tmp_path / "check")], check=True)
    path.write_text(source.replace("typedef struct { int unused; } r_context;",
                                   "typedef struct { char excess[4096]; } r_context;"))
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "deployment host context exceeds reservation" in failed.stderr


def test_fixed_binding_refuses_undersized_allocation():
    contract = interface()
    contract["deployment"] = dict(provider=dict(allocation_alignment=64, host_context_reservation_bytes=4096),
                                 allocations=[dict(name="device_scratch", address=BASE, bytes=8)])
    with pytest.raises(ValueError, match="deployment allocation differs for device_scratch"):
        application_binding("sample", regions(), contract)

def test_real_exact_float_full_and_tail_placement(tmp_path):
    from copy import deepcopy
    from test_exact_float_device import request
    from atlas_mlir_native import exact_float_application as exact
    from merlin.compile.application_deployment import place
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    from types import SimpleNamespace
    from pathlib import Path

    source = request('bf16_add', 1025)
    options = CodegenOptions(application_deployment="onchip", io_memory_base=BASE, io_memory_bytes=786432,
        io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)
    with codegen_options(options):
        compiled = tmp_path / 'segments' / 'float'
        exact.compile_source(None, source, target_source=Path(__file__).resolve().parents[1],
                             destination=compiled, limits=None)
        package = exact.prepare_c_package(source, compiled, 'float',
                                          target_source=Path(__file__).resolve().parents[1])
        rows = [dict(id='float', placement='accelerator', context_index=0,
            native_programs=['segments/float/' + body['path']
                             for body in package.description['float_schedule']['bodies']],
            description=package.description)]
        assert len(rows[0]['native_programs']) == 2
        assert [body.nodes[-1].type.shape for body in exact.prepare_plan(source).bodies] == [(1024,), (1,)]
        provider = AtlasApplicationAdapter.deployment_requirements(rows, tmp_path)
        (frame,) = provider['reservations']
        assert frame['sequential_images']['total_launches'] == 2
        assert len(frame['source_reservations']) == 6
        assert frame['address'] == min(r['address'] for r in frame['source_reservations'])
        assert frame['address'] + frame['bytes'] == max(
            r['address'] + r['bytes'] for r in frame['source_reservations'])
        interface = dict(inputs=[n.id for n in source.nodes[:-1]], outputs=list(source.outputs),
                         workspace_bytes=package.description['workspace_bytes'], workspace_alignment=64)
        deployment = place(source, AtlasApplicationAdapter, rows, interface, tmp_path)
        assert deployment['allocations'][0]['address'] >= frame['address'] + frame['bytes']
        # The identical reservations without a compiler sequential proof still refuse.
        unproven = deepcopy(provider)
        unproven['reservations'] = frame['source_reservations']
        with pytest.raises(NativeCompilationError, match='reservations overlap within a call'):
            place(source, SimpleNamespace(deployment_requirements=lambda *_: unproven),
                  rows, interface, tmp_path)
        rows[0]['description'] = {**package.description, 'device_launch_schedule': {}}
        with pytest.raises(NativeCompilationError, match='sequential schedule differs'):
            AtlasApplicationAdapter.deployment_requirements(rows, tmp_path)

def test_standalone_v8_flat_buffers_and_length_checks(tmp_path):
    plan = dict(schema="atlas.native_tile_execution_plan.v8",
                scope="diagnostic_selected_standalone_core",
                input_addresses=[BASE, BASE + 2048], input_payload_bytes=[1024, 1024],
                output_addresses=[BASE + 4096], output_bytes=2048)
    publish(tmp_path, plan)
    assert [(r["address"], r["bytes"], r["role"]) for r in requirements(tmp_path)["reservations"]] == [
        (BASE, 1024, "input"), (BASE + 2048, 1024, "input"), (BASE + 4096, 2048, "output")]
    for field in ("input_addresses", "input_payload_bytes", "output_addresses"):
        publish(tmp_path, {**plan, field: []})
        with pytest.raises(NativeCompilationError, match="buffer lengths differ"):
            requirements(tmp_path)
    publish(tmp_path, plan)
    second = tmp_path / "segments/s"
    second.mkdir()
    for member in ("program.bin", "physical-program.json"):
        (second / member).write_bytes((tmp_path / "segments/r" / member).read_bytes())
    (second / "execution_plan.json").write_text(json.dumps({**plan,
        "input_addresses": [BASE, BASE + 2048, BASE + 4096],
        "input_payload_bytes": [1024, 1024, 2048],
        "output_addresses": [BASE + 6144]}))
    rows = regions()
    rows[0]["native_programs"].append("segments/s")
    assert [(r["address"], r["bytes"]) for r in requirements(tmp_path, rows)["reservations"]] == [
        (BASE, 1024), (BASE + 2048, 1024), (BASE + 4096, 2048), (BASE + 6144, 2048)]


def test_tiled_rhs_refill_groups_follow_native_n_outer_bounds(tmp_path):
    publish(tmp_path, dict(schema="atlas.native_tile_execution_plan.v8",
        scope="diagnostic_selected_standalone_core", input_addresses=[BASE, BASE + 2048],
        input_payload_bytes=[1024, 1024], output_addresses=[BASE + 4096], output_bytes=2048))
    rows = regions()
    rows[0]["inputs"] = ["source_a", "source_sx", "source_w", "source_sw"]
    rows[0]["description"].update(kind="already_quantized", input_node_ids=["a", "sx", "w", "sw"],
        precision_plan=dict(schema="atlas.already_quantized_contraction.v1",
            operand_ids=["a", "w"], batch_shape=[], shape=[1, 65, 33]))
    groups = requirements(tmp_path, rows)["refill_groups"]
    assert groups == [dict(context_index=0, operand="source_w", groups=[
        dict(first_launch=0, launch_count=2, source_offset=0, bytes=1056),
        dict(first_launch=2, launch_count=2, source_offset=1056, bytes=1056),
        dict(first_launch=4, launch_count=2, source_offset=2112, bytes=33)])]
    precision = rows[0]["description"]["precision_plan"]
    for changes in (dict(shape=[33, 65, 33]), dict(batch_shape=[2]),
                    dict(operand_permutation={}), dict(schema="atlas.already_quantized_epilogue.v1")):
        original = dict(precision)
        precision.update(changes)
        assert "refill_groups" not in requirements(tmp_path, rows)
        precision.clear()
        precision.update(original)
