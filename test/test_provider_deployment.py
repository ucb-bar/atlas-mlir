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
    with codegen_options(CodegenOptions(io_memory_base=BASE, io_memory_bytes=786432)):
        return AtlasApplicationAdapter.deployment_requirements(rows or regions(), root)


def test_counted_reservations_and_relative_exact_image(tmp_path):
    publish(tmp_path, counted())
    result = requirements(tmp_path)
    assert result["schema"] == "atlas.provider_deployment.v1"
    assert result["window"] == dict(address=BASE, bytes=786432)
    assert [row["bytes"] for row in result["reservations"]] == [1024, 1024, 2048, 2048, 2048, 2048]
    assert all(row["lifetime"] == dict(scope="region_call", context_index=0)
               and row["promotion_receipts"] == [] for row in result["reservations"])
    assert result["images"] == [dict(path="segments/r/program.bin", bytes=4,
        sha256=hashlib.sha256(b"\x73\0\0\0").hexdigest(), imem_offset_bytes=0,
        capacity_bytes=131072, upload_convention="counted_ensure_loaded")]
    assert (result["raw_scratch_bytes"], result["conversion_scratch_bytes"]) == (4096, 1024)


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
    assert tuple(hashlib.sha256(text.encode()).hexdigest() for text in
                 (binding.declarations, binding.implementation)) == expected


def fixed_binding():
    contract = interface()
    contract["deployment"] = dict(allocations=[dict(name=key, address=BASE + offset, bytes=size)
        for key, offset, size in (("device_scratch", 0, 4096), ("conversion_scratch", 4096, 1024),
                                 ("workspace", 8192, 512))])
    return application_binding("sample", regions(), contract)


def test_fixed_context_compiles_and_passes_buffer_sizes(tmp_path):
    binding = fixed_binding()
    assert binding.interface["host_context_reservation_bytes"] == 4096
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
    contract["deployment"] = dict(allocations=[dict(name="device_scratch", address=BASE, bytes=8)])
    with pytest.raises(ValueError, match="deployment allocation differs for device_scratch"):
        application_binding("sample", regions(), contract)
