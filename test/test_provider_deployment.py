"""Deployment serialization and opt-in fixed-address application contexts."""
import hashlib
import json
import subprocess

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.application_c_context import application_binding
from atlas_mlir_native.options import CodegenOptions, codegen_options
from merlin.compiler.semantic.target_binding import NativeCompilationError


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


def counted(base=BASE):
    return dict(schema="atlas.quantized_loop_execution.v1", input_addresses=[base, base + 2048],
                input_buffer_bytes=[1024, 1024], exponent_addresses=[base + 4096, base + 6144],
                exponent_buffer_bytes=[2048, 2048], output_addresses=[base + 8192], output_bytes=2048,
                status_address=base + 10240, status_bytes=2048)


def requirements(root, rows=None):
    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432,
            io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)):
        return AtlasApplicationAdapter.deployment_requirements(rows or regions(), root)


def test_counted_reservations_and_relative_exact_image(tmp_path):
    publish(tmp_path, counted())
    result = requirements(tmp_path)
    assert result["schema"] == "merlin.provider_deployment.v1"
    assert result["sys_clk_freq_hz"] == 100000000
    assert result["window"] == dict(address=BASE, bytes=786432)
    assert result["window_section"] == ".atlas_window"
    assert result["image_section"] == ".atlas_images"
    assert "parameter_section" not in result
    assert "PENGUIN26_UART_BAUDRATE" in result["platform_owned_requirements"]
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
    assert result["transport_timeout_seconds"] == 60
    assert result["transport_clock_hz_bound"] == 1_000_000_000
    assert result["host_allocation"]["measurement"] == "measured_sections"
    assert result["host_allocation"]["usable_bytes"] == 61440
    assert result["host_allocation"]["bytes"] == 65536
    assert result["host_allocation"]["declared_reservations"] == [dict(name="stack", bytes=8192),
                                                                  dict(name="heap", bytes=0)]
    assert result["host_allocation"]["reserved_bytes"] == 8192
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


def test_board_ddr_window_ends_ddr_after_a_host_at_its_base(tmp_path):
    from atlas_mlir_native.soc import SOC
    ddr = SOC.memory_region("ddr")
    for base, host in ((ddr.base, dict(address=ddr.base, bytes=ddr.capacity_bytes, usable_bytes=ddr.capacity_bytes,
                                       placement="after_window_data")),
                       (ddr.base + 28672, dict(address=ddr.base, bytes=28672, usable_bytes=28672))):
        publish(tmp_path / str(base), counted(base))
        window = ddr.end - base
        with codegen_options(CodegenOptions(io_memory_base=base, io_memory_bytes=window,
                io_memory_capacity_base=base, io_memory_capacity_bytes=window)):
            result = AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path / str(base))
        assert result["window"] == dict(address=base, bytes=window)
        assert {key: result["host_allocation"][key] for key in host} == host
        assert ("placement" in result["host_allocation"]) == (base == ddr.base)
        assert result["parameter_section"] == ".atlas_parameters"
        assert result["host_context_reservation_bytes"] == ddr.capacity_bytes
    with codegen_options(CodegenOptions(io_memory_base=ddr.base, io_memory_bytes=ddr.capacity_bytes // 2,
            io_memory_capacity_base=ddr.base, io_memory_capacity_bytes=ddr.capacity_bytes // 2)):
        assert AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path) is None


def test_resident_board_stages_in_a_capacity_prefix_and_reads_resident_operands_in_place(tmp_path):
    from atlas_mlir_native.soc import SOC
    ddr = SOC.memory_region("ddr")
    base, staging = ddr.base + 28672, 1 << 20
    plan = counted(base)
    arena = ddr.end - 4096
    plan.update(input_addresses=[base, arena], exponent_addresses=[base + 4096, arena + 2048],
                resident_bindings=[dict(address=arena, bytes=1024), dict(address=arena + 2048, bytes=2048)])
    publish(tmp_path, plan)
    options = dict(io_memory_base=base, io_memory_bytes=staging, io_memory_capacity_base=base,
                   io_memory_capacity_bytes=ddr.end - base)
    with codegen_options(CodegenOptions(application_residency=True, **options)):
        result = AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path)
    assert result["window"] == dict(address=base, bytes=staging) and result["resident_section"] == ".atlas_resident"
    assert all(row["address"] < base + staging for row in result["reservations"])
    assert len(result["reservations"]) == 4  # one input, one exponent, the output, and the status
    assert [row["upload_convention"] for row in result["images"]] == ["resident_template_bind"]
    # Without residency the same split is not a board geometry, and resident operands stay refused elsewhere.
    with codegen_options(CodegenOptions(**options)):
        assert AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path) is None
    with codegen_options(CodegenOptions(application_residency=True, **dict(options, io_memory_bytes=ddr.end - base))):
        assert "resident_section" in AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path)
    with pytest.raises(NativeCompilationError, match="does not support resident bindings"):
        requirements(tmp_path)


def test_a_resident_deployment_refused_by_its_source_replay_is_a_compile_refusal(monkeypatch):
    from atlas_mlir_native import resident_deployment

    calls = []

    def replay(*args, source_request_digest):
        calls.append((args, source_request_digest))
        raise ValueError("resident context differs from its source")

    monkeypatch.setattr(resident_deployment, "checked_resident_deployment", replay)
    with pytest.raises(NativeCompilationError, match="resident deployment: resident context differs") as refused:
        AtlasApplicationAdapter.admit_resident_deployment("regions", "sources", "root", "target",
                                                         source_request_digest="digest")
    assert refused.value.status == "unsupported_semantics"
    assert calls == [(("regions", "sources", "root", "target"), "digest")]

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


def vector_quantizer():
    from atlas_mlir_native.activation_quantize_device import prepare_plan
    from test_already_quantized import quantizer_source

    return prepare_plan(quantizer_source(count=1024), io_memory_base=BASE).record()


def test_vector_quantizer_reserves_entire_padded_frame_and_call_lifetimes(tmp_path):
    plan = vector_quantizer()
    publish(tmp_path, plan)
    rows = regions()
    rows[0]["native_programs"] = ["segments/quantizer/segments/body_000"]
    folder = tmp_path / rows[0]["native_programs"][0]
    folder.mkdir(parents=True)
    for member in ("execution_plan.json", "program.bin", "physical-program.json"):
        (folder / member).write_bytes((tmp_path / "segments/r" / member).read_bytes())
    # A later region may reuse the frame after the quantizer call completes.
    rows.append(dict(placement="accelerator", context_index=2,
                     native_programs=["segments/r"], description={}))
    result = requirements(tmp_path, rows)
    assert [(row["address"], row["bytes"], row["role"]) for row in result["reservations"][:3]] == [
        (BASE, 2048, "input"), (BASE + 2048, 1024, "code"), (BASE + 3072, 32, "exponent")]
    assert sum(row["bytes"] for row in result["reservations"][:3]) == plan["workspace_bytes"]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=index)
               and row["promotion_receipts"] == []
               for index, group in ((0, result["reservations"][:3]), (2, result["reservations"][3:]))
               for row in group)
    assert all(row["upload_convention"] == "counted_ensure_loaded" for row in result["images"])
    assert result["regions"][0]["launches_per_invocation"] == 1


@pytest.mark.parametrize("change", [
    {"input_address": None}, {"input_bytes": "2048"}, {"code_bytes": True},
    {"exponent_bytes": 0}, {"workspace_bytes": None}, {"engine": "unsupported_device_body"},
    {"launch_count": True}, {"launch_count": 2}, {"input_addresses": None},
    {"input_addresses": []}, {"input_payload_bytes": {}}, {"input_payload_bytes": [True]},
    {"output_addresses": [BASE]}, {"code_address": BASE + 2016},
    {"exponent_address": BASE + 3104}, {"output_bytes": 1024}, {"workspace_bytes": 3136},
    {"resident_bindings": [{"node_id": "x"}]}, {"private_staging": {"slot_addresses": [BASE]}},
])
def test_vector_quantizer_refuses_malformed_buffer_contract(tmp_path, change):
    publish(tmp_path, {**vector_quantizer(), **change})
    with pytest.raises(NativeCompilationError, match="activation quantizer buffer contract differs"):
        requirements(tmp_path)


@pytest.mark.parametrize("field", ["input_address", "code_bytes", "exponent_bytes", "workspace_bytes"])
def test_vector_quantizer_refuses_missing_buffer_fields(tmp_path, field):
    plan = vector_quantizer()
    del plan[field]
    publish(tmp_path, plan)
    with pytest.raises(NativeCompilationError, match="activation quantizer buffer contract differs"):
        requirements(tmp_path)


def test_vector_quantizer_retains_window_and_physical_image_checks(tmp_path):
    plan = vector_quantizer()
    shift = 786432 - 2048
    for field in ("input_address", "code_address", "exponent_address"):
        plan[field] += shift
    plan["input_addresses"] = [plan["input_address"]]
    plan["output_addresses"] = [plan["code_address"]]
    publish(tmp_path, plan)
    with pytest.raises(NativeCompilationError, match="outside selected window"):
        requirements(tmp_path)
    publish(tmp_path, vector_quantizer())
    (tmp_path / "segments/r/program.bin").write_bytes(b"\0\0\0\0")
    with pytest.raises(NativeCompilationError, match="differs from physical program"):
        requirements(tmp_path)


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
    # Routes chosen only because one counted image does not fit refuse for size, so a caller can split the region.
    for route, status in (("bounded_k32_tile_dispatch", "resource_limit"), ("checked_counted_row_blocks", "resource_limit"),
                          ("unknown_route", "unsupported_semantics")):
        rows[0]["description"]["execution_route"] = route
        with pytest.raises(NativeCompilationError, match="checked_counted_image") as caught:
            requirements(tmp_path, rows)
        assert caught.value.status == status


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


def test_host_only_binding_exists_only_for_onchip_deployment(tmp_path):
    from atlas_mlir_native.application import AtlasApplicationAdapter

    host = [regions()[1]]
    assert AtlasApplicationAdapter.application_binding("sample", host, interface()) is None
    contract = interface()
    contract["deployment"] = dict(provider=dict(allocation_alignment=64, host_context_reservation_bytes=4096),
        allocations=[dict(name=key, address=BASE + offset, bytes=size)
        for key, offset, size in (("device_scratch", 0, 64), ("conversion_scratch", 64, 64), ("workspace", 128, 512))])
    binding = AtlasApplicationAdapter.application_binding("sample", host, contract)
    assert "int sample_bind(" in binding.declarations and "_bind(&ctx->region_" not in binding.implementation
    # The host ABI the binding names is declared before it.
    assert binding.declarations.index("} atlas_host;") < binding.declarations.index("int sample_bind(")
    source = """
#include <stdint.h>
#include <stddef.h>
typedef struct { int active; } sample_driver_state;
typedef struct { void **items; size_t count; } sample_driver_contexts;
""" + binding.declarations + """
static int sample_invoke(sample_driver_state *s,void *p,size_t n,const float *const *in,
                         float *const *out,sample_driver_contexts *dispatch) {
  (void)s;(void)p;(void)n;(void)in;(void)out;return dispatch->count!=1;
}
""" + binding.implementation + """
int main(void) {
  sample_context ctx={0};atlas_host host={0};atlas_tile_memory_io io={0};
  const void *inputs[]={0};void *outputs[]={0};
  return sample_bind(&ctx,&host,io,1) || sample_run(&ctx,inputs,outputs);
}
"""
    path = tmp_path / "host.c"
    path.write_text(source)
    command = ["cc", "-std=c11", "-Wall", "-Wextra", "-Werror", str(path), "-o", str(tmp_path / "host")]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run([str(tmp_path / "host")], check=True)


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

@pytest.mark.parametrize("private_output", [False, True])
def test_real_exact_float_full_and_tail_placement(tmp_path, private_output):
    from copy import deepcopy
    from test_exact_float_device import request
    from atlas_mlir_native import exact_float_application as exact
    from merlin.compiler.build.deployment import place
    from merlin.compiler.semantic.target_binding import NativeCompilationError
    from types import SimpleNamespace
    from pathlib import Path

    source = request('bf16_add', 1025)
    options = CodegenOptions(application_deployment="onchip", io_memory_base=BASE, io_memory_bytes=786432,
        io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)
    with codegen_options(options):
        compiled = tmp_path / 'segments' / 'float'
        exact.compile_source(None, source, target_source=Path(__file__).resolve().parents[1],
                             destination=compiled, limits=None, private_output=private_output)
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


def test_real_nested_quantizer_full_and_tail_placement(tmp_path):
    from copy import deepcopy
    from pathlib import Path
    from types import SimpleNamespace
    from atlas_mlir_native import activation_quantize_application as quantizer
    from merlin.compiler.build.deployment import place
    from test_activation_quantize_device import dynamic_source

    source = dynamic_source(shape=(9, 1, 64), reduce="head")
    options = CodegenOptions(application_deployment="onchip", device_kv_quantizer=True,
        io_memory_base=BASE, io_memory_bytes=786432,
        io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)
    with codegen_options(options):
        compiled = tmp_path / "segments" / "quantizer"
        target = Path(__file__).resolve().parents[1]
        quantizer.compile_source(None, source, target_source=target, destination=compiled, limits=None)
        package = quantizer.prepare_c_package(source, compiled, "quantizer", target_source=target)
        rows = [dict(id="quantizer", placement="accelerator", context_index=0,
                     native_programs=list(package.programs), description=package.description)]
        provider = AtlasApplicationAdapter.deployment_requirements(rows, tmp_path)
        (frame,) = provider["reservations"]
        assert frame["sequential_images"]["total_launches"] == 2
        assert len(frame["source_reservations"]) == 6
        assert [row["role"] for row in frame["source_reservations"]] == ["input", "code", "exponent"] * 2
        assert all(row["lifetime"] == dict(scope="region_call", context_index=0)
                   for row in frame["source_reservations"])
        assert provider["raw_scratch_bytes"] == package.description["workspace_bytes"]
        assert provider["conversion_scratch_bytes"] == 1
        interface = dict(inputs=[source.nodes[0].id], outputs=list(source.outputs),
                         workspace_bytes=package.description["workspace_bytes"], workspace_alignment=64)
        deployment = place(source, AtlasApplicationAdapter, rows, interface, tmp_path)
        assert deployment["allocations"][0]["address"] >= frame["address"] + frame["bytes"]
        unproven = deepcopy(provider)
        unproven["reservations"] = frame["source_reservations"]
        with pytest.raises(NativeCompilationError, match="reservations overlap within a call"):
            place(source, SimpleNamespace(deployment_requirements=lambda *_: unproven),
                  rows, interface, tmp_path)
        rows[0]["description"] = {**package.description, "device_launch_schedule": {}}
        with pytest.raises(NativeCompilationError, match="activation quantizer sequential schedule differs"):
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
    rows[0]["description"]["launches_per_invocation"] = 2
    assert [(r["address"], r["bytes"]) for r in requirements(tmp_path, rows)["reservations"]] == [
        (BASE, 1024), (BASE + 2048, 1024), (BASE + 4096, 2048), (BASE + 6144, 2048)]


@pytest.mark.parametrize("inputs,sizes,output", [
    ([BASE], [2048], BASE + 1024),
    ([BASE, BASE + 512], [1024, 1024], BASE + 4096),
])
def test_standalone_v8_refuses_overlapping_buffers_within_image(tmp_path, inputs, sizes, output):
    publish(tmp_path, dict(schema="atlas.native_tile_execution_plan.v8",
        scope="diagnostic_selected_standalone_core", input_addresses=inputs,
        input_payload_bytes=sizes, output_addresses=[output], output_bytes=2048))
    with pytest.raises(NativeCompilationError, match="overlap within an image"):
        requirements(tmp_path)


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
    assert groups == [dict(context_index=0, operand="source_w", region_operand="w", groups=[
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


def reduction_plan():
    from atlas_mlir_native import reduction

    plan = reduction.prepare_plan(reduction.request_for((1, 1, 2048), (2,)))
    starts = [BASE + sum(plan.packed_sizes[:i]) for i in range(3)]
    return dict(schema="atlas.reduction_execution_plan.v1", input_addresses=starts[:2],
                input_buffer_bytes=list(plan.packed_sizes[:2]), output_addresses=starts[2:],
                output_bytes=plan.packed_sizes[2], output_physical_layouts=["row_column_zero_bf16_pair_tiles"],
                tiling=plan.record())


def test_reduction_reserves_data_seed_and_output(tmp_path):
    plan = reduction_plan()
    publish(tmp_path, plan)
    result = requirements(tmp_path)
    assert [(row["address"], row["bytes"], row["role"]) for row in result["reservations"]] == [
        (address, size, role) for address, size, role in zip(
            plan["input_addresses"] + plan["output_addresses"], plan["input_buffer_bytes"] + [plan["output_bytes"]],
            ("input", "input", "output"), strict=True)]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=0) for row in result["reservations"])
    publish(tmp_path, {**plan, "output_addresses": [BASE + 786432]})
    with pytest.raises(NativeCompilationError, match="outside selected window"):
        requirements(tmp_path)


def row_loop_plan(shape=(32, 65), budget=None):
    from atlas_mlir_native.row_loop import prepare_plan
    from test_tiled_softmax import source_request

    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432,
            io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)):
        plan = prepare_plan(source_request(shape, masked=True), instruction_budget=budget)
        return plan.execution_record(), plan.end


@pytest.mark.parametrize("shape,version", [((32, 65), 1), ((65, 64), 1), ((32, 65), 2), ((65, 64), 2)])
def test_row_loop_reserves_readonly_and_every_writable_group_frame(tmp_path, shape, version):
    from atlas_mlir_native.limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT

    plan, end = row_loop_plan(shape, None if version == 1 else 2 * DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT)
    assert plan["schema"] == f"atlas.row_loop_execution_plan.v{version}"
    publish(tmp_path, plan)
    rows = requirements(tmp_path)["reservations"]
    groups, stride, scratch = plan["groups"], plan["output_stride_bytes"], plan["scratch_bytes"]
    assert rows[0] == dict(name="segments/r/readonly", address=BASE, bytes=groups * plan["readonly_stride_bytes"],
                           role="input", lifetime=dict(scope="region_call", context_index=0), promotion_receipts=[])
    expected = []
    for group in range(groups):
        expected.append((plan["output_base"] + group * stride, stride - scratch, "output"))
        if scratch:
            expected.append((plan["scratch_address"] + group * stride, scratch, "private"))
    assert [(row["address"], row["bytes"], row["role"]) for row in rows[1:]] == expected
    # The reservations tile the complete row-loop frame without gaps or overlap.
    spans = sorted((row["address"], row["bytes"]) for row in rows)
    assert spans[0][0] == BASE and all(a + b == c for (a, b), (c, _) in zip(spans, spans[1:]))
    assert spans[-1][0] + spans[-1][1] == end


@pytest.mark.parametrize("change", [
    {"groups": 0}, {"groups": True}, {"readonly_stride_bytes": 0}, {"output_base": None},
    {"scratch_stride_bytes": 0}, {"scratch_bytes": -2048}, {"resident_bindings": [{"node_id": "x"}]},
    {"private_staging": {"slot_addresses": [BASE]}},
])
def test_row_loop_refuses_malformed_frame(tmp_path, change):
    plan, _ = row_loop_plan()
    plan = {**plan, **change}
    publish(tmp_path, plan)
    with pytest.raises(NativeCompilationError, match="row loop frame contract differs"):
        requirements(tmp_path)


@pytest.mark.parametrize("field,delta", [("output_base", 2048), ("scratch_address", 2048),
                                         ("scratch_bytes", 2048)])
def test_row_loop_refuses_inconsistent_frame_addresses(tmp_path, field, delta):
    plan, _ = row_loop_plan()
    plan[field] += delta
    publish(tmp_path, plan)
    with pytest.raises(NativeCompilationError, match="row loop frame contract differs"):
        requirements(tmp_path)


def test_deployment_and_board_options_derive_changed_soc_resources(tmp_path):
    from dataclasses import replace
    from atlas_mlir_native.options import BoardDescription
    from atlas_mlir_native.provider_deployment import SOC, deployment_requirements
    from atlas_mlir_native.memory_windows import dma_memory_windows, dma_span_in_window

    # A second declared map must change serialized capacities and placements.
    changed = replace(SOC, memory_regions=tuple(
        replace(region, base=region.base + 0x1000000, capacity_bytes=region.capacity_bytes * 2)
        if region.id in {"host_scratch", "ddr", "dma"} else
        replace(region, base=region.base + 2048, capacity_bytes=region.capacity_bytes - 2048)
        if region.id == "atlas_data" else
        replace(region, base=region.base + 4096, capacity_bytes=region.capacity_bytes // 2)
        if region.id == "atlas_imem" else region for region in SOC.memory_regions),
        resources=tuple(replace(resource, address=resource.address + 4096)
                        if resource.address is not None else resource for resource in SOC.resources))
    data = changed.memory_region("atlas_data")
    host = changed.memory_region("host_scratch")
    imem = changed.memory_region("atlas_imem")
    with codegen_options(CodegenOptions(io_memory_base=data.base, io_memory_bytes=data.capacity_bytes,
            io_memory_capacity_base=data.base, io_memory_capacity_bytes=data.capacity_bytes)):
        result = deployment_requirements([dict(placement="host", description={}, native_programs=[])],
                                         tmp_path, soc=changed)
    assert result["window"] == dict(address=data.base, bytes=data.capacity_bytes)
    assert result["host_allocation"]["address"] == host.base
    assert result["host_allocation"]["bytes"] == host.capacity_bytes
    assert result["host_map"] == dict(imem_address=imem.base, imem_bytes=imem.capacity_bytes,
                                      csr_address=changed.resource("atlas.csr").address)
    board = BoardDescription(soc=changed, host_reserved_bytes=4096, staging_bytes=2048)
    assert board.options()["io_memory_base"] == changed.memory_region("ddr").base + 4096
    assert board.options()["io_memory_capacity_bytes"] == changed.memory_region("ddr").capacity_bytes - 4096
    dma = changed.memory_region("dma")
    assert AtlasApplicationAdapter.resident_memory_geometry(soc=changed) == dict(
        dma_base=dma.base, dma_capacity=dma.capacity_bytes, alignment=2048)
    assert dma_memory_windows(changed) == tuple((region.base, region.end)
        for region in (dma, host, data))
    assert dma_span_in_window(host.base, host.capacity_bytes, soc=changed)
    assert dma_span_in_window(data.base, data.capacity_bytes, soc=changed)
    assert not dma_span_in_window(SOC.memory_region("dma").base, 2048, soc=changed)
    assert not dma_span_in_window(SOC.memory_region("host_scratch").base, 2048, soc=changed)
    assert not dma_span_in_window(SOC.memory_region("atlas_data").base, 2048, soc=changed)


def placed(root, change, parameters=None):
    """Merlin's placement of the counted fixture after `change` edits its published provider."""
    from types import SimpleNamespace
    from merlin.compiler.build.deployment import place

    publish(root, counted())
    provider = requirements(root)
    change(provider)
    tensor = SimpleNamespace(shape=(4, 4), dtype="bf16", record=lambda: dict(shape=[4, 4], dtype="bf16"))
    source = SimpleNamespace(node=lambda key: SimpleNamespace(type=tensor), digest=lambda: "0" * 64,
                             target_identity="atlas")
    rows = [dict(row, id=f"r{i}", context_index=i) for i, row in enumerate(regions())]
    contract = dict(interface(), workspace_alignment=64)
    if parameters is not None:
        contract["model_parameters"] = dict(payload_bytes=len(parameters),
                                            payload_sha256=hashlib.sha256(parameters).hexdigest())
    with codegen_options(CodegenOptions(application_deployment="onchip", io_memory_base=BASE, io_memory_bytes=786432,
            io_memory_capacity_base=BASE, io_memory_capacity_bytes=786432)):
        return place(source, SimpleNamespace(deployment_requirements=lambda *_: provider), rows, contract, root)


def test_host_after_window_data_shares_one_memory(tmp_path):
    from merlin.compiler.build.deployment import platform_files

    shared = lambda provider: provider["host_allocation"].update(
        address=BASE, bytes=786432, usable_bytes=786432, placement="after_window_data")
    deployment = placed(tmp_path, shared)
    host, window = deployment["host_allocation"], deployment["window"]
    used = -(-deployment["window_required_bytes"] // 16) * 16 + deployment["window_image_reserve_bytes"]
    assert host["address"] == BASE + -(-used // 64) * 64
    assert window == dict(address=BASE, bytes=host["address"] - BASE)
    assert host["bytes"] == host["usable_bytes"] == BASE + 786432 - host["address"]
    assert deployment["provider"]["window"] == window and deployment["provider"]["host_allocation"] == host
    script = platform_files(deployment)["handoff/chip.ld"].decode()
    assert f"HOST (rwx) : ORIGIN  =  {host['address']}, LENGTH  =  {host['bytes']}\n" in script
    assert f"WINDOW (rw) : ORIGIN  =  {BASE}, LENGTH  =  {window['bytes']} }}" in script
    with pytest.raises(NativeCompilationError, match="shares one complete window"):
        placed(tmp_path / "short", lambda provider: (shared(provider), provider["host_allocation"].update(
            bytes=786432 - 64, usable_bytes=786432 - 64)))
    with pytest.raises(NativeCompilationError, match="no room for the host stack and heap"):
        placed(tmp_path / "full", lambda provider: (shared(provider), provider["host_allocation"].update(
            reserved_bytes=786432, declared_reservations=[dict(name="stack", bytes=786432 - 4096),
                                                          dict(name="heap", bytes=4096)])))


def test_provider_parameter_section_ends_the_window_in_a_read_only_load(tmp_path):
    import shutil
    from merlin.compiler.build.deployment import platform_files

    weights = bytes(range(128))
    named = lambda provider: provider.update(parameter_section=".test_parameters")
    deployment = placed(tmp_path, named, weights)
    row = deployment["parameters"]
    offset = row["address"] - BASE
    assert offset + row["bytes"] == deployment["window_required_bytes"]
    files = platform_files(deployment, {"inputs/input_000.bin": b"\0" * 32, row["path"]: weights})
    source = files["handoff/window.S"].decode()
    assert (f'.incbin "window.bin",0,{offset}\n.section .test_parameters,"a",@progbits\n'
            f'.incbin "window.bin",{offset},{len(weights)}\n') in source
    script = files["handoff/chip.ld"].decode()
    assert "window PT_LOAD FLAGS(6); parameters PT_LOAD FLAGS(4); }" in script
    assert " .test_parameters : { KEEP(*(.test_parameters)) } >WINDOW :parameters\n" in script
    assert "__atlas_images_end = .; } >WINDOW :parameters" in script
    # The linker places the declared layout: fail, never skip, without the RISC-V toolchain.
    compiler, readelf = shutil.which("riscv64-unknown-elf-gcc"), shutil.which("riscv64-unknown-elf-readelf")
    assert compiler and readelf, "linker qualification requires the RISC-V toolchain on PATH"
    for name in ("chip.ld", "window.S", "window.bin"):
        (tmp_path / name).write_bytes(files["handoff/" + name])
    (tmp_path / "host.S").write_text('.section .text\n.global _start\n_start: .byte 0\n'
                                     '.section .atlas_images,"a",@progbits\n.word 0x73\n')
    elf = tmp_path / "layout.elf"
    subprocess.run([compiler, "-nostdlib", "-no-pie", "-Wl,--build-id=none", "-Wl,-T," + str(tmp_path / "chip.ld"),
                    "-Wa,-I," + str(tmp_path), str(tmp_path / "host.S"), str(tmp_path / "window.S"), "-o", str(elf)],
                   check=True, capture_output=True, text=True)
    loads = [line.split() for line in subprocess.run([readelf, "-lW", str(elf)], check=True, capture_output=True,
                                                      text=True).stdout.splitlines() if line.strip().startswith("LOAD")]
    flags = {int(fields[2], 16): fields[-2] for fields in loads}
    assert flags[BASE] == "RW" and flags[row["address"]] == "R"
    assert weights in elf.read_bytes()
    # Without parameters, or without a provider section, the window stays one writable load.
    assert b"test_parameters" not in platform_files(placed(tmp_path / "none", named))["handoff/window.S"]
    plain = placed(tmp_path / "plain", lambda provider: None, weights)
    assert b"parameters PT_LOAD" not in platform_files(plain)["handoff/chip.ld"]
    deployment["window_required_bytes"] += 64
    with pytest.raises(NativeCompilationError, match="read-only parameters must end the complete window data"):
        platform_files(deployment)
    with pytest.raises(NativeCompilationError, match="sections must differ"):
        placed(tmp_path / "same", lambda provider: provider.update(parameter_section=".atlas_window"))


def test_provider_publishes_the_deploy_clock_or_the_declared_host_clock(tmp_path):
    publish(tmp_path, counted())
    assert requirements(tmp_path)["sys_clk_freq_hz"] == 100000000
    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432, io_memory_capacity_base=BASE,
            io_memory_capacity_bytes=786432, host_clock_hz=500000000)):
        assert AtlasApplicationAdapter.deployment_requirements(regions(), tmp_path)["sys_clk_freq_hz"] == 500000000
    assert "host_clock_hz" not in CodegenOptions().identity_record()
    assert CodegenOptions(host_clock_hz=500000000).identity_record()["host_clock_hz"] == 500000000
    with pytest.raises(ValueError, match="host_clock_hz"):
        CodegenOptions(host_clock_hz=0)


@pytest.mark.parametrize("host", [0, 28672])
@pytest.mark.parametrize("kind", ["no_parameters", "state", "arena_too_large"])
def test_a_board_deploy_that_residency_cannot_apply_to_is_placed_exactly_as_without_it(tmp_path, monkeypatch, kind,
                                                                                       host):
    """The real adapter on the board geometry the driver selects: a declined application sees residency off."""
    from types import SimpleNamespace
    from merlin.compiler.build.deployment import place
    from merlin.compiler.schedule.storage import application_residency as residency
    from atlas_mlir_native.options import current_codegen_options
    from atlas_mlir_native.soc import SOC

    ddr = SOC.memory_region("ddr")
    # The measuring compile's window starts at the DDR base; the bound compile's starts after the measured host.
    base, size = ddr.base + host, ddr.capacity_bytes - host
    publish(tmp_path, counted(base))
    tensor = SimpleNamespace(shape=(4, 4), dtype="bf16", record=lambda: dict(shape=[4, 4], dtype="bf16"))
    source = SimpleNamespace(node=lambda key: SimpleNamespace(type=tensor), digest=lambda: "0" * 64,
                             target_identity="atlas")
    rows = [dict(row, id=f"r{i}", context_index=i) for i, row in enumerate(regions())]
    contract = dict(interface(), workspace_alignment=64)
    board = dict(application_deployment="onchip", io_memory_base=base, io_memory_bytes=size,
                 io_memory_capacity_base=base, io_memory_capacity_bytes=size)
    parameters, state = None, None
    if kind == "state":
        parameters, state = object(), object()
    if kind == "arena_too_large":
        # An arena as large as the whole DDR leaves no staging window after the host, or does not fit.
        entry = SimpleNamespace(offset=0, nbytes=4, name="w")
        parameters = SimpleNamespace(node_ids=("a",), entries=(entry,), check=lambda request: None)
        plan = SimpleNamespace(selected=(SimpleNamespace(id="device", placement="accelerator"),))
        region = SimpleNamespace(digest=lambda: "region", input_storages=(("a", "external"),))
        monkeypatch.setattr(residency, "application_plan", lambda *args: plan)
        monkeypatch.setattr(residency, "outline_semantic_regions",
                            lambda *args, **kwargs: SimpleNamespace(regions=(("device", region),)))
        monkeypatch.setattr(AtlasApplicationAdapter, "inventory_resident_inputs", staticmethod(
            lambda region, alternative: [dict(node_id="a", bytes=ddr.capacity_bytes, alignment=4096, layout={})]))
    admitted = []
    monkeypatch.setattr(AtlasApplicationAdapter, "admit_resident_deployment",
                        staticmethod(lambda *args, **kwargs: admitted.append(args)), raising=False)

    def deploy():
        provider = AtlasApplicationAdapter.deployment_requirements(rows, tmp_path)
        return provider, place(source, AtlasApplicationAdapter, rows, contract, tmp_path)

    with codegen_options(CodegenOptions(**board)):
        expected = deploy()
    with codegen_options(CodegenOptions(application_residency=True, **board)):
        with residency.residency_scope(source, AtlasApplicationAdapter, parameters, state):
            reason = residency.declined()
            assert residency.publication() is None
            assert current_codegen_options() == CodegenOptions(**board)
            actual = deploy()
    assert json.dumps(actual, sort_keys=True, default=str) == json.dumps(expected, sort_keys=True, default=str)
    assert actual[0]["window"] == dict(address=base, bytes=size) and "resident_section" not in actual[0]
    assert expected[1] is not None and not admitted
    assert reason == {"no_parameters": "the application has no external parameters to keep resident",
                      "state": "the application has persistent state inputs, which stay a host interface",
                      "arena_too_large": (f"the resident arena of {ddr.capacity_bytes} bytes leaves no staging window"
                                          if not host else f"the resident arena of {ddr.capacity_bytes} bytes does "
                                          f"not fit the declared capacity of {size} bytes")}[kind]
