"""Assemble and exactly deduplicate kernels from a completed Smol policy replay.

Produces kernel assets and descriptors for whole-host generation. It does not
execute a device, create a host ELF, or qualify a complete FullBundle.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

from atlas_path_guard import checked
from atlas_program import assembly
from run_model_tile import configure_model, model_binding


def safe_path(value):
    path = Path(value)
    if any(word in part.lower() for part in path.parts for word in ('hammer', 'vlsi')):
        raise ValueError('restricted path component')
    return checked(path)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require_finite_trajectory(arrays):
    """BF16 captures contain raw uint16 words, rather than integer values."""
    for name, value in arrays.items():
        finite = (not np.any((value & np.uint16(0x7f80)) == np.uint16(0x7f80))) if value.dtype == np.uint16 else np.isfinite(value).all()
        if not finite:
            raise ValueError('Nonfinite captured trajectory: ' + name)


def split_initial(initial, expected, row):
    shape = row.get('physical_mkn')
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(x) is not int or x <= 0 or x % 32 for x in shape)):
        raise ValueError('Require positive physical M/K/N multiples of32')
    m, k, n = shape
    a_size, w_size, output_size = m*k, k*n, m*n*2
    offset = a_size + w_size + 256
    total = offset + output_size + 256
    if total > 64*1024*1024 or row.get('output_offset') != offset or row.get('dram_bytes') != total:
        raise ValueError('Initial DRAM geometry differs')
    if (len(initial) != total or len(expected) != output_size
            or initial[a_size+w_size:offset] != b'\xa5'*256
            or initial[offset:offset+output_size] != b'\xcd'*output_size
            or initial[-256:] != b'Z'*256):
        raise ValueError('Initial DRAM extent or guard bytes differ')
    resident_b = k*n <= 256*1024
    resident_a = not resident_b and m*k <= 256*1024
    loads = m*k*(n//32)+k*n if resident_b else m*k+k*n*(m//32) if resident_a else (m//32)*(n//32)*(2*k//32*1024)
    if (row.get('resident_a') is not resident_a or row.get('resident_b') is not resident_b
            or row.get('requested_dma') != {'load_bytes': loads, 'store_bytes': output_size}):
        raise ValueError('Projection residence or DMA descriptor differs')
    return initial[:a_size], initial[a_size:a_size+w_size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=safe_path, required=True)
    parser.add_argument('--model-root', type=safe_path, required=True)
    parser.add_argument('--out', type=safe_path, required=True)
    args = parser.parse_args()
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    raw = args.replay.read_bytes()
    replay = json.loads(raw)
    ref = replay.get('projection_reference', {})
    if (replay.get('status') != 'PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY'
            or replay.get('input_state_rng_bookends_equal') is not True
            or ref.get('source_bookends_equal') is not True
            or ref.get('fresh_model_executed') is not False
            or len(ref.get('calls', [])) != 1421 or not 1 <= len(ref.get('kernels', [])) <= 20000):
        raise ValueError('Require complete genuine-predecessor Atlas arithmetic replay')
    pins = {str(args.replay): sha(raw), str(safe_path(__file__)): sha(safe_path(__file__).read_bytes())}

    def merge_pin(path, digest):
        key = str(path)
        if key in pins and pins[key] != digest:
            raise ValueError('Source pin conflict: ' + key)
        pins[key] = digest

    for record in [*replay['source_bookends'], *ref['source_bookends']]:
        path = safe_path(record['path'])
        merge_pin(path, record['sha256'])
    tools_root = safe_path(__file__).parent
    for name, module in tuple(sys.modules.items()):
        if name.startswith('atlas_') or name == 'run_model_tile':
            path = safe_path(module.__file__)
            if path.parent != tools_root:
                raise ValueError('Exporter helper imported outside selected tools: ' + name)
            merge_pin(path, sha(path.read_bytes()))
    for path, expected in pins.items():
        if sha(safe_path(path).read_bytes()) != expected:
            raise ValueError('Replay source changed before export: ' + path)
    trajectory = replay['trajectory']
    trajectory_path = safe_path(trajectory['path'])
    if sha(trajectory_path.read_bytes()) != trajectory['sha256']:
        raise ValueError('Replay trajectory payload changed')
    merge_pin(trajectory_path, trajectory['sha256'])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        require_finite_trajectory({name: archive[name] for name in archive.files})
    args.out.mkdir(parents=True, exist_ok=False)
    assets = args.out / 'assets'
    assets.mkdir()
    model_before_configuration = model_binding(args.model_root)
    _, _, assemble, _ = configure_model(args.model_root)
    model_before = model_binding(args.model_root)
    for key in ('revision', 'tracked_source_diff_exit', 'curve_sha256'):
        if model_before_configuration[key] != model_before[key]:
            raise ValueError('Model binding changed during assembler configuration')
    for name, digest in model_before_configuration['imported_source_sha256'].items():
        if model_before['imported_source_sha256'].get(name) != digest:
            raise ValueError('Previously imported model source changed during configuration')
    emitted = {}

    def asset(data, suffix):
        name = 'assets/' + sha(data) + suffix
        if name not in emitted:
            safe_path(args.out / name).write_bytes(data)
            emitted[name] = len(data)
        return name

    kernels = []
    for index, row in enumerate(ref['kernels']):
        if row.get('index') != index:
            raise ValueError('Kernel order differs')
        files = {}
        for name in ('program.S', 'initial-dram.bin', 'expected-output.bin'):
            record = row['files'][name]
            path = safe_path(record['path'])
            content = path.read_bytes()
            if sha(content) != record['sha256'] or len(content) != record['bytes']:
                raise ValueError('Replay payload changed: ' + name)
            files[name] = content
            merge_pin(path, record['sha256'])
        a, w = split_initial(files['initial-dram.bin'], files['expected-output.bin'], row)
        m, k, n = row['physical_mkn']
        canonical = assembly(m, k, n, row['output_offset'], row['output_rescale_exponent'],
                             resident_a=row['resident_a'], resident_b=row['resident_b'],
                             pretranspose_resident_b=row['resident_b']).encode()
        if canonical != files['program.S']:
            raise ValueError('Replay assembly differs from pinned recipe')
        words = assemble(io.StringIO(canonical.decode())).assemble()
        if not words or len(words) > 8192 or words[-1] != 0x73 or 0x73 in words[:-1] or 0x100073 in words:
            raise ValueError('Program extent or terminal instruction differs')
        binary = struct.pack('<' + 'I'*len(words), *words)
        descriptor = {key: row[key] for key in ('index', 'physical_mkn', 'output_offset', 'requested_dma',
                     'logical_mkn', 'call', 'module', 'logical_row_start', 'a_scale', 'b_scale', 'output_rescale_exponent')}
        descriptor.update(source=asset(canonical, '.S'), original=asset(binary, '.bin'),
                          activation=asset(a, '.bin'), weights=asset(w, '.bin'),
                          expected=asset(files['expected-output.bin'], '.bin'),
                          total_dram_bytes=len(files['initial-dram.bin']), dram_base=0x100000000,
                          initial_sha256=sha(files['initial-dram.bin']))
        kernels.append(descriptor)
    model_after = model_binding(args.model_root)
    if model_before != model_after:
        raise ValueError('Assembler model binding changed during export')
    for path, expected in pins.items():
        if sha(safe_path(path).read_bytes()) != expected:
            raise ValueError('Source or replay payload changed during export: ' + path)
    report = {'status': 'PREPARED_SMOL_FULL_KERNELS_NOT_EXECUTED', 'schema': 'atlas.full_kernel_assets.v1',
              'workload': 'smol-full', 'kernel_count': len(kernels), 'kernels': kernels,
              'replay_receipt': {'path': str(args.replay), 'sha256': sha(raw)},
              'source_sha256': pins, 'source_bookends_equal': True,
              'assembler_binding': model_after, 'assembler_binding_before_configuration': model_before_configuration,
              'fresh_model_executed': False,
              'finite_trajectory_verified': True, 'trajectory': trajectory,
              'unique_asset_count': len(emitted), 'unique_asset_bytes': sum(emitted.values())}
    manifest = safe_path(args.out / 'kernels.json')
    manifest.write_text(json.dumps(report, indent=2)+'\n')
    files = [*sorted(emitted), 'kernels.json']
    inventory = ''.join(sha(safe_path(args.out / name).read_bytes()) + '  ' + name + '\n' for name in files)
    safe_path(args.out / 'SHA256SUMS').write_text(inventory)
    print(json.dumps({key: report[key] for key in ('status', 'kernel_count', 'unique_asset_count', 'unique_asset_bytes')}), flush=True)


if __name__ == '__main__':
    main()
