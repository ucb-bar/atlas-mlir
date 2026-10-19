#!/usr/bin/env python3
"""Generate raw BF16 and continuation fixtures for the existing AtlasCore test."""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_mxu_spilling import (
    mxu_diamond,
    mxu_diamond_inputs,
    mxu_inputs,
    mxu_loop,
    mxu_loop_inputs,
    mxu_pressure,
)
from test_virtual_spilling import compile_program, run


def write_fixture(root, name, source, rewritten, machine, binary, tiles, outputs, control=None):
    fixture = root / name
    fixture.mkdir()
    for filename, data in (('source.mlir', source), ('spilled.mlir', rewritten), ('machine.mlir', machine)):
        (fixture / filename).write_text(data)
    (fixture / 'program.bin').write_bytes(binary)
    (fixture / 'expected.bin').write_bytes(b''.join(outputs))
    inputs = [(0x90000000+i*2048, tile) for i, tile in enumerate(tiles)]
    if control is not None:
        inputs.append((0x90010000, control.to_bytes(4, 'little') + bytes(1020)))
    digest = hashlib.sha256(binary).hexdigest()
    properties = ['schema=atlas.rtl_tile_fixture.v1', f'name={name}', f'input_count={len(inputs)}',
                  f'output_count={len(outputs)}', f'program_sha256={digest}', 'output.address=0x90020000']
    for index, (address, payload) in enumerate(inputs):
        (fixture / f'input-{index}.bin').write_bytes(payload)
        properties.append(f'input.{index}.address={address:#x}')
    (fixture / 'manifest.properties').write_text('\n'.join(properties) + '\n')
    return {'name': name, 'words': len(binary)//4, 'sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not exist; fixtures require fresh output')
    args.out.mkdir(parents=True)
    records = []
    for unit in (0, 1):
        for continuation, reset, iterations in ((False, False, 0), (True, False, 0),
                                                (True, True, 0), (True, False, 1),
                                                (True, False, 3)):
            source = mxu_loop(unit, iterations) if iterations else mxu_pressure(unit, continuation=continuation, reset=reset)
            rewritten = run(source, '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
            if rewritten.returncode:
                raise RuntimeError(rewritten.stderr)
            machine, program = compile_program(source)
            binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in program['instructions'])
            tiles, outputs = mxu_inputs(continuation, reset=reset)
            if iterations:
                tiles, outputs = mxu_loop_inputs(iterations)
            for group in range(1 if continuation else 64):
                name = f'mxu_spill_u{unit}_' + (f'loop_{iterations}' if iterations else 'reset' if reset else 'continuation' if continuation else f'bits_{group}')
                if not continuation:
                    # The oldest accumulator is the spill victim. Its 64*1024
                    # cells cover every 16-bit pattern through VMEM, not merely
                    # through push/pop. Other slots use rotated patterns.
                    tiles = [b''.join(struct.pack('<H', (group*1024+i*8192+j) & 0xffff)
                                     for j in range(1024)) for i in range(3)]
                    outputs = tiles
                records.append(write_fixture(args.out, name, source, rewritten.stdout, machine, binary, tiles, outputs))
    source = mxu_diamond()
    rewritten = run(source, '--spill-atlas-virtual', '--verify-atlas-virtual-stream')
    if rewritten.returncode:
        raise RuntimeError(rewritten.stderr)
    machine, program = compile_program(source)
    binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in program['instructions'])
    for take in (0, 1, 2, 3):
        tiles, outputs = mxu_diamond_inputs(take)
        records.append(write_fixture(args.out, f'mxu_spill_diamond_{take}', source, rewritten.stdout,
                                     machine, binary, tiles, outputs, control=take))
    (args.out / 'fixtures.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps({'fixtures': len(records), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
