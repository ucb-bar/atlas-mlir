'''Public stateful image assembly; fake MMIO never executes device arithmetic.'''
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from atlas_mlir_native import backend
from atlas_mlir_native.call import BF16Call
from atlas_mlir_native.policy import BF16_DRAM_PAIR_LAYOUT
from test_public_backend import _bf16_platform, _configured_transport, _system

from merlin import api
from merlin.compiler.artifact import Artifact, Compilation, Deployment
from merlin.compiler.backend.stateful import CRegionABI2
from merlin.compiler.build.contextentry import CImageContext, CImageEntry, ReservedImageInterval
from merlin.compiler.build.image import CImageLink
from merlin.compiler.request import CompileOptions
from merlin.semantic_compiler.search import SearchLimits
from merlin.semantic_compiler.snapshot import open_native_snapshot


@pytest.mark.parametrize("native", [False, True])
def test_selected_bf16_image_links_embedded_binary_and_runs_only_fake_transport(tmp_path, monkeypatch, native):
    compiler = shutil.which("gcc")
    values = tuple(os.environ.get(name) for name in (
        "ATLAS_SYSTEM_SNAPSHOT", "ATLAS_SYSTEM_TARGET_SOURCE", "ATLAS_SYSTEM_TOOL_DIRECTORY"))
    if not compiler or native and not all(values):
        pytest.skip("select actual host compiler and explicit native snapshot/source/tools")
    if native:
        snapshot, target_source, tools = Path(values[0]), Path(values[1]), Path(values[2])
        selected_snapshot = open_native_snapshot(snapshot)
    else:
        plan = {"schema": "atlas.native_tile_execution_plan.v8", "shape": [32, 32], "dtype": "bf16",
            "input_bytes": 2048, "output_bytes": 2048, "input_addresses": [0x90001800],
            "output_addresses": [0x90011800], "output_physical_layouts": ["bf16_pair_halves"],
            "boundary_domain_obligations": [{"input": "arg0", "required_domain": "finite-normal-or-zero-bf16",
                                              "status": "caller_premise_unchecked"}]}
        transport, _, _ = _configured_transport(tmp_path, monkeypatch, plan_override=plan)
        selected_snapshot, target_source, tools = transport.snapshot, transport.target_source, transport.tool_directory
    root = Path(__file__).resolve().parents[1]
    from merlin.common.paths import data_path

    platform, context = _bf16_platform(tmp_path)
    flags = ("-std=c11", "-O2", "-Wall", "-Wextra", "-Werror")
    call = BF16Call(context=context, compiler=Path(compiler), flags=flags, runtime=root / "runtime",
        headers=data_path("runtime", "c"), platform=platform, contract="contract.json",
        platform_header="platform.h", platform_initialize="test_platform_initialize", link_inputs=("platform.c",))
    provider = backend.AtlasBackend.configure(snapshot=selected_snapshot, target_source=target_source,
        tool_directory=tools, pipeline="mandatory", fixed_inputs=(("arg0", 3),), fixed_outputs=(35,),
        limits=SearchLimits(), call=call)
    startup_directory = tmp_path / "startup"
    startup_directory.mkdir()
    (startup_directory / "response.h").write_text("static const unsigned char response[2048]={0};\n")
    source = startup_directory / "main.c"
    source.write_text(r'''#include <stdint.h>
#include <string.h>
#include "entry.h"
#include "regions/0/call/platform/platform.h"
#include "response.h"
int main(void) {
 _Alignas(64) uint16_t input[1024]={0},output[1024];
 _Alignas(64) unsigned char arena[64];
 memset(output,0x5a,sizeof(output));
 test_platform_response(response,sizeof(response));
 model_buffer buffers[]={{input,sizeof(input)},{output,sizeof(output)}}; model_state state={0};
 if(image_run(&state,arena,sizeof(arena),buffers,2) || memcmp(input,output,sizeof(input)) ||
    state.active || state.completed!=1 || test_platform_starts()!=1) return 1;
 test_platform_mode(1);
 if(image_run(&state,arena,sizeof(arena),buffers,2)!=MODEL_KERNEL_ERROR || !state.active) return 2;
 unsigned effects=test_platform_effects(); state=(model_state){0};
 if(image_run(&state,arena,sizeof(arena),buffers,2)!=MODEL_BUSY || test_platform_effects()!=effects) return 3;
 return 0;
}
''')
    products = (Artifact.from_file(source, kind="c-source", role="entry"),
                Artifact.from_file(startup_directory / "response.h", kind="c-header", role="test-response"))
    deployment = Deployment("fake-mmio-host", "linux", "x86_64", "sysv", 1)
    startup = Compilation(startup_directory, "caller.fake-mmio", products[0].sha256, products, deployment=deployment)
    startup.write()
    contract = json.loads((platform.directory / "contract.json").read_text())
    reservations = tuple(ReservedImageInterval("physical", contract[name]["address"], contract[name]["bytes"])
                         for name in ("input", "output")) + (
        ReservedImageInterval("physical", contract["map"]["imem_base"], contract["map"]["imem_capacity_bytes"]),
        ReservedImageInterval("physical", contract["map"]["csr_base"], 0x1c))
    image_flags = (*flags, "-no-pie", "-I" + str(tmp_path / "compiled-image/regions/0/call/sdk"))
    def admit(binding, region, image, options):
        # Original caller approval is scoped to the reviewed fake endpoint, not hardware facts.
        assert binding.platform is platform and binding.context == context
        assert image.startup is startup and image.deployment == deployment
        assert image.compiler == Path(compiler).resolve() and image.flags == image_flags
        assert options == {"scope": "fake MMIO only; no device arithmetic or hardware execution"}
        assert CRegionABI2.from_compilation(region).context == context
        assert region.metadata["device_pre_reservations"] == contract
        assert region.metadata["physical_layout_contract"] == BF16_DRAM_PAIR_LAYOUT
        assert image.context_entry.reservations == reservations
    entry = CImageEntry(contexts=(CImageContext(context, platform, "contract.json",
        "int(context*,void*,size_t)", "contents_not_retained", "host_state_only"),), symbol="image_run",
        validate_platform=admit, options={"scope": "fake MMIO only; no device arithmetic or hardware execution"},
        reservations=reservations, entry_state_bytes=128, entry_state_pre_reserved=True)
    image = CImageLink(compiler=Path(compiler), flags=image_flags, startup=startup, link_inputs=("main.c",),
                       deployment=deployment, context_entry=entry)
    original = (root / "test/examples/native_bf16_relu_linalg.mlir").read_text()
    result = api.compile(original, backend=_system(provider, tmp_path, context=context, image=image),
        output=tmp_path / "compiled-image",
        options=CompileOptions(parameters={"merlin.emit": "c-dispatch", "merlin.entry": "model"}))
    assert result.metadata["stage"] == "linked_image" and result.deployment == deployment
    subprocess.run([str(result.directory / "image/program.elf")], check=True, capture_output=True, text=True)
    report = json.loads((result.directory / "image/link.json").read_text())
    assert report["placement_execution"] == "not_observed"
    assert report["context_entry"]["error_lifetime"] == "persistent_entry_context_workspace; no_retry_or_reset"
    assert (result.directory / "regions/0/program.bin").read_bytes()
    assert not any(a.kind == "llvm-ir" for a in result.artifacts)
