"""Run a compiler-emitted program in the explicitly selected Rust model."""
import json
import os
import subprocess


def execute(physical, memory, base, destination):
    destination.mkdir(parents=True)
    (destination / 'input.bin').write_bytes(memory)
    bundle = dict(schema_version=1,
                  words=[row['word_u32'] for row in physical['instructions']],
                  config=dict(numerics='rtl', dram_base=base, dram_size=len(memory)),
                  memory_regions=[dict(address=base, file='input.bin')])
    program = destination / 'program.json'
    program.write_text(json.dumps(bundle)+'\n')
    state = destination / 'state'
    result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
                             '--compute', 'cpu', '--max-cycles', '100000000',
                             '--dump-state', str(state)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    stats = json.loads(result.stdout)
    assert stats['completed'], stats
    first = base - base % 4096
    pages = []
    for address in range(first, base + len(memory), 4096):
        path = state / f'dram_{address:016x}.bin'
        pages.append(path.read_bytes() if path.exists() else bytes(4096))
    actual = b''.join(pages)[base-first:base-first+len(memory)]
    return actual, stats
