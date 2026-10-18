"""Source-bound complete-K 32xK by Kx32 Atlas virtual exports.

The unchanged retained Fraction oracle defines tensor-pow2 encoding, ordered
MXU0 recurrence and final rescale. native_C never supplies expected bytes.
This module imports neither Torch nor npu_model. No redundant ops are injected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import atlas_sota_arithmetic as oracle

ORACLE_SHA256 = '7976cb9496e31fe9fcfa2bcbbaa2a84c9015c5c471b6c5fef71f5ee94aacc8b0'
INPUT_BASE = 0x90000000
BF16 = '!atlas.virtual_bf16'
FP8 = '!atlas.virtual_fp8'
STATE = '!atlas.virtual_state'


def safe_path(value) -> Path:
    path = Path(value).absolute()
    if any('hammer' in part.lower() or 'vlsi' in part.lower() for part in path.parts):
        raise ValueError('prohibited path component')
    for prefix in (*reversed(path.parents), path):
        if prefix.is_symlink():
            raise ValueError('symlink paths are not accepted')
    return path


def sha(path) -> str:
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_arrays(arrays, source_dtype):
    required = {'A', 'B', 'native_C'}
    optional = {'A_bf16', 'B_bf16', 'native_C_bf16'}
    if not required <= arrays.keys() or arrays.keys() - required - optional:
        raise ValueError('unsupported operand archive members')
    a, b, c = (arrays[name] for name in ('A', 'B', 'native_C'))
    if (a.ndim != 2 or b.ndim != 2 or a.shape[0] != 32 or b.shape[1] != 32 or
            not 32 <= a.shape[1] <= 4096 or a.shape[1] % 32 or b.shape[0] != a.shape[1] or c.shape != (32, 32)):
        raise ValueError('require complete-K 32xK by Kx32, K multiple32 in [32,4096]')
    if source_dtype not in ('float32', 'bfloat16'):
        raise ValueError('unsupported source dtype')
    for name in required:
        value = arrays[name]
        if value.dtype != np.dtype('float32') or not np.isfinite(value).all():
            raise ValueError('source arrays must be finite float32 views')
        raw = arrays.get(name + '_bf16')
        if source_dtype == 'bfloat16' and name in ('A', 'B'):
            if raw is None or np.any(value.view(np.uint32) & 65535):
                raise ValueError('BF16 operands require authenticated raw words')
        if raw is not None and (raw.dtype != np.dtype('uint16') or raw.shape != value.shape or
                                not np.array_equal(raw.astype(np.uint32) << 16, value.view(np.uint32))):
            raise ValueError('raw BF16 words do not authenticate source view')
    return a.shape[1]


def load_source(input_path, manifest_path, *, npz_sha256, manifest_sha256):
    input_path, manifest_path = safe_path(input_path), safe_path(manifest_path)
    if sha(input_path) != npz_sha256 or sha(manifest_path) != manifest_sha256:
        raise ValueError('source identity mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema_version') != 1 or manifest.get('npz_sha256') != npz_sha256 or not manifest.get('source'):
        raise ValueError('manifest does not bind the source archive')
    with zipfile.ZipFile(input_path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        allowed = {name + '.npy' for name in ('A', 'B', 'native_C', 'A_bf16', 'B_bf16', 'native_C_bf16')}
        if len(names) != len(set(names)) or set(names) - allowed or sum(m.file_size for m in members) > 32 * 1024 * 1024:
            raise ValueError('unbounded/unsupported NPZ archive')
    with np.load(input_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    k = validate_arrays(arrays, manifest.get('source_dtype'))
    site = manifest.get('site', {})
    if site.get('k_range') != [0, k] or not (site.get('complete_k') or site.get('full_k_reduction')):
        raise ValueError('capture does not declare this complete K range')
    return arrays, manifest


def encode_inputs(arrays):
    """Exact retained tensor-pow2 recipe over each whole captured operand."""
    encoded, conversions = {}, {}
    for name in ('A', 'B'):
        value = arrays[name]
        bf16 = np.array([oracle.f32_to_bf16_bits(float(v)) for v in value.flat], dtype=np.uint16).reshape(value.shape)
        exponent = oracle.tensor_scale_exponent(bf16.flat)
        packed = np.array([oracle.pack_bf16(int(v), 127 + exponent) for v in bf16.flat], dtype=np.uint8).reshape(value.shape)
        encoded[name] = packed
        conversions[name] = {'scale_exponent': exponent, 'scale_code': 127 + exponent,
                             'bf16_sha256': hashlib.sha256(bf16.tobytes()).hexdigest(),
                             'fp8_sha256': hashlib.sha256(packed.tobytes()).hexdigest()}
    return encoded, conversions


def expected_output(encoded, exponent):
    raw = [[oracle.mxu0_dot(encoded['A'][row], encoded['B'][:, col]) for col in range(32)] for row in range(32)]
    result = [[oracle.rescale_bf16(bits, exponent) for bits in row] for row in raw]
    return oracle.output_halves(result), raw


def signed_i32(value):
    if not 0 <= value < 1 << 32:
        raise ValueError('address is outside u32')
    return value - (1 << 32) if value >= 1 << 31 else value


def virtual_source(k, exponent, output_address):
    if not 32 <= k <= 4096 or k % 32 or not -126 <= exponent <= 127:
        raise ValueError('unsupported complete-K/rescale domain')
    if output_address % 1024 or output_address < INPUT_BASE + 2 * (k // 32) * 1024 + 32 or output_address + 2048 > 1 << 32:
        raise ValueError('output must be disjoint and ABI-aligned within selected u32 memory')
    tiles = k // 32
    lines = ['module {', f'  func.func @sota_tile() -> {STATE} attributes {{atlas.input_dram_base = {INPUT_BASE} : i64, atlas.output_dram_base = {output_address} : i64}} {{',
             f'    %s0 = "atlas.virtual_start"() : () -> {STATE}',
             '    %tile_bytes = arith.constant 1024 : i32',
             '    %output_bytes = arith.constant 2048 : i32']
    state = '%s0'
    for tile in range(tiles):
        for name, offset in (('a', tile * 1024), ('b', (tiles + tile) * 1024)):
            lines += [f'    %{name}_addr{tile} = arith.constant {signed_i32(INPUT_BASE + offset)} : i32',
                      f'    %s_{name}{tile}, %{name}{tile} = "atlas.virtual_dma_load_fp8"({state}, %{name}_addr{tile}, %tile_bytes) : ({STATE}, i32, i32) -> ({STATE}, {FP8})']
            state = f'%s_{name}{tile}'
        lines += [f'    %bt{tile} = "atlas.virtual_xlu_transpose"(%b{tile}) : ({FP8}) -> {FP8}',
                  f'    %s_w{tile}, %w{tile} = "atlas.virtual_mxu_load_weight"({state}, %bt{tile}) {{unit = 0 : i32}} : ({STATE}, {FP8}) -> ({STATE}, !atlas.virtual_mxu_weight<0>)']
        if tile == 0:
            lines.append(f'    %s_mac{tile}, %acc{tile} = "atlas.virtual_mxu_reset"(%s_w{tile}, %a{tile}, %w{tile}) : ({STATE}, {FP8}, !atlas.virtual_mxu_weight<0>) -> ({STATE}, !atlas.virtual_mxu_acc<0>)')
        else:
            lines.append(f'    %s_mac{tile}, %acc{tile} = "atlas.virtual_mxu_accumulate"(%s_w{tile}, %a{tile}, %w{tile}, %acc{tile-1}) : ({STATE}, {FP8}, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> ({STATE}, !atlas.virtual_mxu_acc<0>)')
        state = f'%s_mac{tile}'
    lines += [f'    %s_read, %raw = "atlas.virtual_mxu_readout_bf16"({state}, %acc{tiles-1}) : ({STATE}, !atlas.virtual_mxu_acc<0>) -> ({STATE}, {BF16})',
              f'    %factor = "atlas.virtual_vpu_splat"() {{bits = {(exponent + 127) << 7} : i32}} : () -> {BF16}',
              f'    %scaled = "atlas.virtual_vpu_binary"(%raw, %factor) {{kind = "mul"}} : ({BF16}, {BF16}) -> {BF16}',
              f'    %out_addr = arith.constant {signed_i32(output_address)} : i32',
              f'    %s_out = "atlas.virtual_dma_store_bf16"(%s_read, %scaled, %out_addr, %output_bytes) : ({STATE}, {BF16}, i32, i32) -> {STATE}',
              f'    return %s_out : {STATE}', '  }', '}']
    return '\n'.join(lines) + '\n'


def materialize(input_path, manifest_path, output, *, name, npz_sha256, manifest_sha256):
    output = safe_path(output)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('invalid workload name')
    if sha(Path(oracle.__file__)) != ORACLE_SHA256:
        raise ValueError('independent oracle source identity mismatch')
    arrays, source_manifest = load_source(input_path, manifest_path, npz_sha256=npz_sha256, manifest_sha256=manifest_sha256)
    encoded, conversions = encode_inputs(arrays)
    exponent = sum(v['scale_exponent'] for v in conversions.values())
    wanted, raw_expected = expected_output(encoded, exponent)
    k = arrays['A'].shape[1];tiles = k // 32
    inputs = []
    blob = bytearray()
    for operand in ('A', 'B'):
        for tile in range(tiles):
            value = encoded[operand][:, tile*32:(tile+1)*32] if operand == 'A' else encoded[operand][tile*32:(tile+1)*32, :]
            data = value.tobytes()
            inputs.append({'operand': operand, 'k_chunk': tile, 'format': 'fp8', 'address': INPUT_BASE + len(blob), 'length': 1024, 'sha256': hashlib.sha256(data).hexdigest()})
            blob.extend(data)
    base = INPUT_BASE - 64
    output_address = INPUT_BASE + len(blob) + 1024
    initial = b'G' * 64 + bytes(blob) + b'\xa5' * 1024 + b'\xcd' * 2048 + b'\x5a' * 64
    expected = bytearray(initial);offset = output_address - base;expected[offset:offset+2048] = wanted
    text = virtual_source(k, exponent, output_address)
    if sha(input_path) != npz_sha256 or sha(manifest_path) != manifest_sha256:
        raise ValueError("source inputs changed during materialization")
    output.mkdir(parents=True, exist_ok=False)
    (output/'source.mlir').write_text(text)
    (output/'initial.bin').write_bytes(initial);(output/'expected.bin').write_bytes(expected);(output/'expected-output.bin').write_bytes(wanted)
    np.savez(output/'encoded.npz', **encoded)
    fixtures = []
    for n in range(2):
        fixtures.append({'workload_id': name, 'case': name + f'-repeat{n+1}', 'base': base, 'length': len(initial),
                         'initial': str(output/'initial.bin'), 'expected': str(output/'expected.bin'),
                         'initial_sha256': hashlib.sha256(initial).hexdigest(), 'expected_sha256': hashlib.sha256(expected).hexdigest(),
                         'inputs': inputs, 'outputs': [{'index': 0, 'address': output_address, 'length': 2048, 'sha256': hashlib.sha256(wanted).hexdigest()}]})
    save(output/'fixtures.json', fixtures)
    receipt = {'status': 'PASS_MATERIALIZATION_ONLY', 'workload_id': name, 'shape': [32,k,32], 'recipe': 'atlas_mxu0_fp8_tensor_pow2_bf16acc_v1',
               'npz': str(safe_path(input_path)), 'npz_sha256': npz_sha256, 'manifest': str(safe_path(manifest_path)), 'manifest_sha256': manifest_sha256,
               'source_sha256': sha(output/'source.mlir'), 'adapter_sha256': sha(Path(__file__)), 'oracle_sha256': ORACLE_SHA256,
               'conversion': conversions, 'output_rescale_exponent': exponent, 'expected_output_sha256': hashlib.sha256(wanted).hexdigest(),
               'target_oracle': 'independent Fraction MXU0 ordered recurrence plus retained output rescale', 'native_C_used_as_oracle': False,
               'site': source_manifest['site'], 'inputs': inputs, 'outputs': fixtures[0]['outputs'], 'fixtures': str(output/'fixtures.json'),
               'scope': 'One complete-K single output tile, target derivative of captured native operands; not whole model or native arithmetic equivalence.'}
    save(output/'materialization.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    for name in ('input', 'manifest', 'npz-sha256', 'manifest-sha256', 'name', 'out'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.input,args.manifest,args.out,name=args.name,npz_sha256=args.npz_sha256,manifest_sha256=args.manifest_sha256),indent=2))

if __name__ == '__main__':main()
