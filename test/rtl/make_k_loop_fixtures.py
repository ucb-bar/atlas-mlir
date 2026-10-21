#!/usr/bin/env python3
"""Build compact K-loop fixtures for the existing standalone AtlasCore test."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_bounded_dma_addresses import loop_inputs, loop_source
from test_virtual_spilling import compile_program


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must be fresh')
    args.out.mkdir(parents=True)
    records = []
    for count in (1, 2, 3, 7):
        source = loop_source(count)
        machine, emitted = compile_program(source)
        binary = b''.join(row['word_u32'].to_bytes(4, 'little') for row in emitted['instructions'])
        inputs, output = loop_inputs(count)
        name = f'k_loop_{count}'
        root = args.out / name
        root.mkdir()
        (root / 'source.mlir').write_text(source)
        (root / 'machine.mlir').write_text(machine)
        (root / 'program.bin').write_bytes(binary)
        (root / 'expected.bin').write_bytes(output)
        properties = ['schema=atlas.rtl_tile_fixture.v1', f'name={name}',
                      f'input_count={len(inputs)}', 'output_count=1',
                      f'program_sha256={hashlib.sha256(binary).hexdigest()}',
                      'output.address=0x90020000']
        for i, (address, data) in enumerate(inputs):
            (root / f'input-{i}.bin').write_bytes(data)
            properties.append(f'input.{i}.address={address:#x}')
        (root / 'manifest.properties').write_text('\n'.join(properties) + '\n')
        records.append({'name': name, 'k_tiles': count, 'words': len(binary)//4})
    (args.out / 'fixtures.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
