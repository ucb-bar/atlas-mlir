#!/usr/bin/env python3
"""Measure an authored partial-output DMA hypothesis, without changing codegen.

The common input load supplies VMEM in the physical BF16 register layout. Compare a prefix
store with serial or queued stores of its occupied register halves. This probes
transport cost only: no contraction, restoration, publication, or board claim.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not (os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR')):
        parser.error('select ATLAS_RUST_NPU_MODEL and ATLAS_OOT_BIN_DIR')
    from atlas_mlir_native.binding import DRAM_BASE, RTL_REVISION
    from test_delay_insertion import program
    from verification_support import constant
    from rust_model import execute
    from tools.profile_quantized_panels import revision

    args.out.mkdir(parents=True, exist_ok=False)
    tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
    binary = Path(os.environ['ATLAS_RUST_NPU_MODEL']).resolve()
    initial = bytearray([0xa5]) * 8192
    initial[:2048] = bytes((i * 37 + 11) % 256 for i in range(2048))
    results = []
    for rows, columns in ((1, 17), (3, 32), (16, 17), (24, 32), (31, 32), (32, 32)):
        prefix = 1024 + rows * 32
        for policy in ('prefix', 'serial_halves', 'queued_halves'):
            spans = [(0, prefix)] if policy == 'prefix' else [(0, rows * 32), (1024, rows * 32)]
            ops = [*constant(6, 0)]
            ops += [('dma_config', f'channel = {channel} : i32, base_reg = 6 : i32') for channel in (0, 1)]
            ops += [*constant(4, 0x2000), *constant(7, DRAM_BASE), *constant(9, 2048),
                    ('dma', 'direction = "load", channel = 0 : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32'),
                    ('dma_wait', 'channel = 0 : i32')]
            for index, (offset, size) in enumerate(spans):
                channel = index if policy == 'queued_halves' else 0
                ops += [*constant(4, 0x2000 + offset // 4),
                        *constant(7, DRAM_BASE + 4096 + offset), *constant(9, size),
                        ('dma', f'direction = "store", channel = {channel} : i32, reg = 4 : i32, dram = 7 : i32, size = 9 : i32')]
                if policy != 'queued_halves':
                    ops.append(('dma_wait', f'channel = {channel} : i32'))
            if policy == 'queued_halves':
                ops += [('dma_wait', f'channel = {channel} : i32') for channel in range(len(spans))]
            ops.append(('trap', 'kind = "ecall"'))
            source = program(ops)
            compiled = subprocess.run([str(tools / 'atlas-opt'), '--insert-atlas-delays',
                '--verify-atlas-machine-stream', '--verify-atlas-timing', '-'],
                input=source, text=True, capture_output=True, timeout=30)
            if compiled.returncode:
                raise RuntimeError(compiled.stderr)
            emitted = subprocess.run([str(tools / 'atlas-emit'), '--program-json', '-'],
                input=compiled.stdout, text=True, capture_output=True, timeout=30)
            if emitted.returncode:
                raise RuntimeError(emitted.stderr)
            physical = json.loads(emitted.stdout)
            destination = args.out / f'{rows}x{columns}-{policy}'
            actual, stats = execute(physical, initial, DRAM_BASE, destination)
            expected = bytearray(initial)
            for offset, size in spans:
                expected[4096+offset:4096+offset+size] = initial[offset:offset+size]
            assert actual == expected, 'transport bytes or untouched memory differ'
            logical = b''.join(actual[4096+(c//16)*1024+r*32+(c%16)*2:
                                     4096+(c//16)*1024+r*32+(c%16)*2+2]
                               for r in range(rows) for c in range(columns))
            wanted = b''.join(initial[(c//16)*1024+r*32+(c%16)*2:
                                     (c//16)*1024+r*32+(c%16)*2+2]
                              for r in range(rows) for c in range(columns))
            assert logical == wanted
            assert stats['dram_read_bytes'] == 2048
            assert stats['dram_write_bytes'] == sum(size for _, size in spans)
            (destination / 'source.mlir').write_text(source)
            (destination / 'machine.mlir').write_text(compiled.stdout)
            (destination / 'physical-program.json').write_text(emitted.stdout)
            results.append(dict(rows=rows, columns=columns, policy=policy,
                spans=spans, cycles=stats['cycles'], dispatches=1,
                dram_read_bytes=stats['dram_read_bytes'], dram_write_bytes=stats['dram_write_bytes'],
                total_instructions=stats['total_instructions'],
                logical_output_sha256=sha256(logical).hexdigest(),
                machine_sha256=sha256(compiled.stdout.encode()).hexdigest()))
    report = dict(scope=__doc__, measurements=results,
        input_sha256=sha256(initial).hexdigest(), numerics='rtl', pack_layout='register_rows',
        rtl_revision=RTL_REVISION, exact_transport_and_guard_match=True,
        provenance=dict(atlas=revision(ROOT), simulator=revision(binary.parent),
            simulator_binary_sha256=sha256(binary.read_bytes()).hexdigest(),
            compiler_tools={name: sha256((tools / name).read_bytes()).hexdigest() for name in ('atlas-opt', 'atlas-emit')},
            evaluator_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
