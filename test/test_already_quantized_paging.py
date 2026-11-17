"""Exercise the published paged dispatcher with a synchronous mock device."""
import ctypes
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from atlas_mlir_native import already_quantized as native
from atlas_mlir_native.binding import io_memory_base
from test_already_quantized import source_request


@pytest.fixture
def exported(tmp_path):
    atlas = Path(__file__).resolve().parents[1]
    source = source_request(m=1, n=160, k=33, batch=())
    plan = native.prepare_source(source)
    compiled = tmp_path / "compiled"
    compiled.mkdir()
    (compiled / "already-quantized-plan.json").write_text(json.dumps(plan.record()))
    for role, body in native._native_bodies(plan):
        directory = compiled / role
        directory.mkdir()
        abi = native.AtlasNativeBinding.plan_boundary_abi(body)
        addresses = [io_memory_base()+abi["fixed_inputs"][node.id]*native.TILE_BYTES
                     for node in body.nodes if node.effect == "input"]
        records = {"manifest.json": dict(target_inventory_sha256=sha256(
            (atlas / "docs/selected-variant-inventory.json").read_bytes()).hexdigest()),
            "source-request.json": body.record(), "execution_plan.json": dict(
                input_addresses=addresses, output_addresses=[io_memory_base()+abi["fixed_outputs"][0]*native.TILE_BYTES],
                operation="mxu1_cap448_seeded_ordered"),
            "physical-program.json": dict(instructions=[dict(word_u32=0x00100073)])}
        for name, record in records.items():
            (directory / name).write_text(json.dumps(record))
        (directory / "program.bin").write_bytes((0x00100073).to_bytes(4, "little"))
    paging = dict(region_operand="w", slot=dict(bytes=2112), segments=[
        dict(id=i, operand_offset=first*1056, bytes=count*1056, first_group=first,
             group_count=count, first_launch=first*2, launch_count=count*2)
        for i, (first, count) in enumerate(((0, 2), (2, 2), (4, 1)))])

    def export(pages=paging):
        with patch.object(native, "verify_native_publication"):
            return native.prepare_c_package(source, compiled, "probe", target_source=atlas, paging=pages)
    return export, paging


def test_segment_partition_and_deployment_only_emission(exported):
    export, paging = exported
    ordinary, files = export(None)
    text = files["probe.c"].decode()
    assert "merlin_deployment_refill(" not in text and "rhs_offset" not in text and "paging_validation" not in ordinary
    record, files = export()
    text = files["probe.c"].decode()
    assert "after earlier starts" in record["paging_validation"]
    assert "which<1u" in text and "rhs_offset" in text
    assert text.count("extern int merlin_deployment_refill(unsigned);") == 1
    for key, change in (("operand_offset", 1), ("first_group", 1), ("first_launch", 1),
                        ("launch_count", 1), ("bytes", 1)):
        original = paging["segments"][1][key]
        paging["segments"][1][key] = change
        with pytest.raises(ValueError, match="partition"):
            export()
        paging["segments"][1][key] = original


def test_refills_validate_each_segment_before_use_and_leave_output_unchanged(exported, tmp_path):
    export, _ = exported
    _, files = export()
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (tmp_path / "shim.c").write_text('''#include "probe.h"
static atlas_host host; static probe_context region; static probe_source_context source;
static uint8_t raw[PROBE_WORKSPACE_BYTES],converted[PROBE_SOURCE_WORKSPACE_BYTES];
static unsigned starts,refills,bad,poison;static uint8_t *slot;static const uint8_t *weights;
static uint32_t read32(void *c,uintptr_t a) {(void)c;return a==0x40004u?starts:5u;}
static void write32(void *c,uintptr_t a,uint32_t v) {(void)c;if(a==0x40018u&&v==1u)++starts;}
static int put(void *c,uintptr_t a,const uint8_t *v,size_t n) {
 (void)c;(void)n;if(a==0x80000800u && v[0]!=(uint8_t)(8u+(starts/2u)*8u))++bad;return 0;
}
static int get(void *c,uintptr_t a,uint8_t *v,size_t n) {(void)c;(void)a;__builtin_memset(v,0,n);return 0;}
int merlin_deployment_refill(unsigned id) {
 if(host.active||id!=refills+1u)return 1;
 ++refills;__builtin_memcpy(slot,weights+id*2112u,id==2u?1056u:2112u);
 if(id==poison)slot[0]=1u;
 if(poison==3u && id==1u)slot[2111]=1u;
 return 0;
}
unsigned start_count(void) {return starts;}
unsigned refill_count(void) {return refills;}
unsigned bad_count(void) {return bad;}
unsigned active(void) {return host.active;}
int invoke(const void *a,const void *sx,void *w,const void *sw,void *out,const void *full,unsigned invalid) {
 starts=refills=bad=0;poison=invalid;slot=w;weights=full;
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read32,write32,0},(atlas_host_memory){1,0,0}))return -1;
 if(probe_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20u))return -2;
 if(probe_source_bind(&source,&region,converted,sizeof(converted)))return -3;
 const void *inputs[]={a,sx,w,sw};void *outputs[]={out};return probe_source_dispatch(&source,inputs,outputs);
}
''')
    # The fixed RHS address comes from the public tile ABI, not the paging schedule.
    body = native.prepare_source(source_request(m=1, n=160, k=33, batch=())).reset
    abi = native.AtlasNativeBinding.plan_boundary_abi(body)
    rhs_address = io_memory_base()+abi["fixed_inputs"]["w"]*native.TILE_BYTES
    shim = tmp_path / "shim.c"
    shim.write_text(shim.read_text().replace("0x80000800u", str(rhs_address)+"u"))
    subprocess.run(["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
        "probe.c", "runtime/atlas_host.c", "shim.c", "-o", "probe.so"], cwd=tmp_path, check=True, capture_output=True)
    lib = ctypes.CDLL(str(tmp_path / "probe.so"))
    lib.invoke.argtypes = [ctypes.c_void_p]*6+[ctypes.c_uint]
    a, sx, sw = np.zeros(33, np.uint8), np.zeros(1, np.int32), np.zeros(160, np.int32)
    full = np.repeat(np.arange(8, 48, 8, dtype=np.uint8), 1056)
    slot = np.zeros(full.size, np.uint8)
    for invalid, expected, starts, refills in ((99, 0, 10, 2), (0, 2, 0, 0), (1, 2, 4, 1), (3, 2, 4, 1)):
        slot[:] = 0
        slot[:2112] = full[:2112]
        if invalid == 0:
            slot[0] = 1
        output = np.full(320, 0xa5, np.uint8)
        status = lib.invoke(a.ctypes.data, sx.ctypes.data, slot.ctypes.data, sw.ctypes.data,
                            output.ctypes.data, full.ctypes.data, invalid)
        assert (status, lib.start_count(), lib.refill_count(), lib.bad_count(), lib.active()) == (
            expected, starts, refills, 0, 0)
        assert np.all(output == (0xa5 if expected else 0))
