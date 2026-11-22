#!/usr/bin/env python3
"""Compare token-group dispatch batching on an existing lifted capture and input.

Measures generated application C, including real launches and host transfers.
Input bytes are loaded only after both applications have compiled and replayed.
"""
import argparse
import ctypes
from dataclasses import replace
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile-inputs', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='raw bytes for the single BF16 input')
    parser.add_argument('--pipeline', choices=('mandatory', 'optimized'), default='mandatory')
    parser.add_argument('--io-memory-bytes', type=int, default=1048576)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')):
        parser.error('select ATLAS_RUST_NPU_MODEL and ATLAS_OOT_BIN_DIR')

    from atlas_mlir_native.application import AtlasApplicationAdapter
    from atlas_mlir_native.binding import DRAM_BASE
    from atlas_mlir_native.options import codegen_options, current_codegen_options
    from merlin.compile.semantic_application import compile_semantic_application, verify_semantic_application
    from merlin.frontends.compile_inputs import load_compile_inputs, load_lifted_argument_bytes
    from merlin.semantic_compiler.linalg_bridge import translate_linalg_text
    from merlin.semantic_compiler.search import SearchLimits
    from tools.profile_quantized_panels import revision
    from test_looped_mnk import build_snapshot
    from rust_model import Application

    policy = json.loads((args.compile_inputs / 'precision-policy.json').read_text())
    staged = load_compile_inputs(args.compile_inputs)
    parameters = load_lifted_argument_bytes(args.compile_inputs, staged)
    adapter = AtlasApplicationAdapter({**policy, 'activation_quantizer': 'device'},
                                     original_source_sha256=policy['source_sha256'])
    source = translate_linalg_text(staged.program, entry=staged.entry, target_identity=policy['target_identity'],
        lowering_policy='diagnostic', argument_constants=parameters,
        source_operation_translators=adapter.source_operation_translators).request
    inputs = [node for node in source.nodes if node.effect == 'input']
    if len(inputs) != 1 or inputs[0].type.dtype != 'bf16':
        parser.error('this evaluator requires exactly one runtime BF16 input')
    if not any(node.op == 'atlas.quantize_act.codes' and dict(node.attrs).get('reduce') == 'token'
               and dict(node.attrs).get('mode') == 'dynamic' for node in source.nodes):
        parser.error('capture has no dynamic token quantizer')
    widths = {'bf16': 2, 'f32': 4, 'i32': 4, 'i8': 1, 'u8': 1}
    sizes = [prod(source.node(key).type.shape) * widths[source.node(key).type.dtype] for key in source.outputs]
    args.out.mkdir(parents=True, exist_ok=False)
    options = replace(current_codegen_options(), io_memory_bytes=args.io_memory_bytes,
                      native_pipeline=args.pipeline, device_kv_quantizer=True,
                      execution_instruction_limit=1000000)
    artifacts = []
    with codegen_options(options):
        snapshot = build_snapshot(args.out)
        for batching in (False, True):
            destination = args.out / ('batched' if batching else 'baseline')
            with codegen_options(replace(options, dynamic_quantizer_group_batching=batching)):
                compile_semantic_application(source, adapter=adapter, snapshot=snapshot, target_source=ROOT,
                    destination=destination, limits=SearchLimits(), name='capture', capacity=args.io_memory_bytes)
                assert verify_semantic_application(source, adapter=adapter, snapshot=snapshot,
                    target_source=ROOT, artifact=destination)['status'] == 'checked'
            artifacts.append(destination)
    raw = args.input.read_bytes()
    assert len(raw) == prod(inputs[0].type.shape)*2, 'input extent differs from captured signature'
    (args.out / 'input.bin').write_bytes(raw)
    results, payloads = [], []
    for destination in artifacts:
        app = Application(destination / 'c-package', 'capture', destination / 'runtime',
                          base=DRAM_BASE-64, size=args.io_memory_bytes+128)
        argument = ctypes.create_string_buffer(raw)
        outputs = [ctypes.create_string_buffer(bytes([0x7a])*(size+64)) for size in sizes]
        app.invoke([argument], outputs)
        assert argument.raw[:len(raw)] == raw
        assert app.memory[:64] == app.memory[-64:] == bytes([0xa5])*64
        payload = []
        for index, (output, size) in enumerate(zip(outputs, sizes, strict=True)):
            assert output.raw[size:size+64] == bytes([0x7a])*64
            payload.append(output.raw[:size])
            (destination / f'output-{index}.bin').write_bytes(payload[-1])
        payloads.append(payload)
        results.append(dict(policy=destination.name, dispatches=len(app.stats),
            upload_bytes=app.upload_bytes, download_bytes=app.download_bytes,
            output_sha256=[sha256(value).hexdigest() for value in payload],
            **{key: sum(row[key] for row in app.stats) for key in
               ('cycles', 'total_instructions', 'dram_read_bytes', 'dram_write_bytes')}))
    assert payloads[0] == payloads[1], 'public output bytes differ'
    binary = Path(os.environ['ATLAS_RUST_NPU_MODEL']).resolve()
    tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
    report = dict(source_digest=source.digest(), input_sha256=sha256(raw).hexdigest(),
        exact_public_output_match=True, measurements=results, options=options.identity_record(),
        instruction_limit=1000000, pack_layout='register_rows',
        provenance=dict(atlas=revision(ROOT), merlin=revision(Path(os.environ['MERLIN_NATIVE_ROOT'])),
            simulator=revision(binary.parent), simulator_binary_sha256=sha256(binary.read_bytes()).hexdigest(),
            compiler_tools={name: sha256((tools / name).read_bytes()).hexdigest() for name in ('atlas-opt','atlas-emit')},
            evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest()),
        scope='existing lifted capture; matched public bytes; device cycles exclude host and image loading time')
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({**report, 'provenance': 'see comparison.json'}, indent=2))


if __name__ == '__main__':
    main()
