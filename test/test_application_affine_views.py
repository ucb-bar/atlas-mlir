"""Run the exported C driver and its actual launches through the Rust model."""
from dataclasses import replace
import ctypes
import json
import os

import pytest

from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import DRAM_BASE, TARGET_IDENTITY
from atlas_mlir_native.options import codegen_options, current_codegen_options
from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
from merlin.semantic_compiler.codegen import CodegenOptions as MerlinOptions, codegen_options as merlin_options
from merlin.semantic_compiler.model import IndexMap, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.search import SearchLimits
from test_already_quantized import _policy
from test_looped_mnk import ROOT, build_snapshot


@pytest.mark.skipif(not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')),
                    reason='select Rust simulator and Atlas tools')
def test_affine_alias_removes_a_real_launch_and_its_transfers(tmp_path):
    from rust_model import Application

    tensor = TensorType((32, 32, 1), 'bf16', 'raw-bit-pattern')
    x = SemanticNode('x', 'input', (), tensor, effect='input')
    a = SemanticNode('a', 'copy', ('x',), tensor)
    b = SemanticNode('b', 'transpose', ('a',), replace(tensor, shape=(1, 32, 32)), index_maps=(
        IndexMap(3, ((0, 1, 0), (0, 0, 1), (1, 0, 0)), (0, 0, 0)), pointwise_identity_maps(3, 1)[0]))
    out = SemanticNode('out', 'copy', ('b',), b.type)
    source = KernelRequest((x, a, b, out), ('out',), ('external',), (('x', 'external'),),
                           TARGET_IDENTITY, 'diagnostic')
    adapter = AtlasApplicationAdapter(_policy(), original_source_sha256='0'*64)
    snapshot = build_snapshot(tmp_path)
    results = []
    for enabled in (False, True):
        destination = tmp_path / str(enabled)
        with merlin_options(MerlinOptions(application_buffer_views=enabled)), codegen_options(
                replace(current_codegen_options(), io_memory_bytes=1048576)):
            manifest = compile_semantic_application(
                source, adapter=adapter, snapshot=snapshot, target_source=ROOT, destination=destination,
                limits=SearchLimits(), name='layout_chain', capacity=65536)
            assert verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
                target_source=ROOT, artifact=destination)['status'] == 'checked'
        launches = 2 if enabled else 3
        assert len(manifest['segments']) == launches
        memory = json.loads((destination / 'program.json').read_text())['memory_plan']
        assert (memory['offsets']['a'] == memory['offsets']['b']) == enabled
        application = Application(destination / 'c-package', 'layout_chain', destination / 'runtime',
                                  base=DRAM_BASE-64, size=8192)
        for invocation in range(2):
            raw = bytes((i*37 + i//64*11 + invocation*113) % 256 for i in range(2048))
            argument = ctypes.create_string_buffer(raw)
            result = ctypes.create_string_buffer(bytes([0x7a])*2112)
            application.invoke([argument], [result])
            assert result.raw[:2048] == argument.raw[:2048] == raw
            assert result.raw[2048:2112] == bytes([0x7a])*64
            assert application.memory[:64] == bytes([0xa5])*64
            assert application.memory[4160:] == bytes([0xa5])*(8192-4160)
            assert len(application.stats) == launches*(invocation+1)
        assert application.upload_bytes == application.download_bytes == launches*2*2048
        for stats in application.stats:
            if 'dram_read_bytes' in stats:
                assert stats['dram_read_bytes'] == stats['dram_write_bytes'] == 2048
        results.append(sum(row['cycles'] for row in application.stats))
    assert results[1] < results[0]
