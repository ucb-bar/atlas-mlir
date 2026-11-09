"""Checked reduction layout bridges retain the original consumer guard."""
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from atlas_mlir_native.adjacent_reduction_scale import alternatives, compile_group, export_group, prepare
from atlas_mlir_native.binding import DRAM_BASE
from merlin.semantic_compiler.model import SemanticNode, pointwise_identity_maps
from merlin.xdsl_dialects.lowering.global_plan import BufferRepresentation, CycleInterval, RegionAlternative, ValueRepresentation
from test_adjacent_square_reduce import case as prefix_case


def case(operand=0, shape=(3, 7), escape=False):
    source, _, choices = prefix_case(shape)
    reduced = source.nodes[-1]
    other = SemanticNode('other', 'input', (), reduced.type, effect='input')
    inputs = (reduced.id, other.id) if operand == 0 else (other.id, reduced.id)
    divided = SemanticNode('divided', 'f32_div', inputs, reduced.type,
        (('numeric_semantics', 'ieee754-f32-div-rne-v1'),),
        index_maps=pointwise_identity_maps(len(reduced.type.shape), 2))
    source = replace(source, nodes=(*source.nodes, other, divided),
        input_storages=(*source.input_storages, (other.id, 'external')),
        outputs=(divided.id, reduced.id) if escape else (divided.id,),
        output_storages=('external', 'external') if escape else ('external',))
    ports = lambda keys: tuple(BufferRepresentation(key, ValueRepresentation('host_memory', 'dense', 'f32')) for key in keys)
    last = RegionAlternative('r2', (2,), 'source', 'accelerator', CycleInterval.unknown('test'),
                             ports(inputs), ports((divided.id,)), metadata=(('kind', 'pointwise_math'),))
    return source, SimpleNamespace(), (*choices, last)


def fixture(group, data, seed, other):
    from atlas_mlir_native.adjacent_square_reduce import input_spec
    from atlas_mlir_native.layout import pack_bf16_pair_halves
    from atlas_mlir_native.recip_lane import vrecip_bf16_bounded_bits
    from atlas_mlir_native.reduction import input_spec as seed_spec
    from atlas_mlir_native.reference import _round_bf16_product, evaluate_seeded_bf16_reduction
    from merlin.semantic_compiler.pointwise_tiling import materialize_pointwise_input, assemble_pointwise_output
    from merlin.semantic_compiler.reduction_tiling import materialize_reduction_inputs
    from merlin.semantic_compiler.reference import TensorValue

    def physical(raw):
        return b''.join(pack_bf16_pair_halves(raw[at:at + 2048]) for at in range(0, len(raw), 2048))
    first, second = group['prefix']['plans']
    codes = np.frombuffer(input_spec().encode(data), dtype='<u2')
    seed_codes = np.frombuffer(seed_spec().encode(seed), dtype='<u2')
    squared = np.array([0 if code & 0x7fff == 0 else _round_bf16_product(int(code), int(code)) for code in codes], dtype='<u2')
    reduction_request = group['prefix']['precisions'][1].request
    def reduce(seed_values):
        args = tuple(TensorValue(node.type, tuple(int(code) for code in values))
                     for node, values in zip(reduction_request.nodes[:2], (squared, seed_values), strict=True))
        return evaluate_seeded_bf16_reduction(reduction_request.nodes[-1], args).elements
    reduced, unseeded = reduce(seed_codes), reduce(np.zeros(len(seed_codes), dtype='<u2'))
    reduction_raw = bytearray()
    for start in range(0, len(reduced), 32):
        tile = np.zeros((32, 32), dtype='<u2')
        count = min(32, len(reduced) - start)
        tile[:count, :] = np.array(unseeded[start:start + count], dtype='<u2')[:, None]
        tile[:count, 0] = reduced[start:start + count]
        reduction_raw.extend(tile.tobytes())
    spec = group['precision'].operand_specs[group['operand']]
    admitted = [any(low <= code <= high for low, high in spec.admitted_code_ranges) for code in reduced]
    pad = int.from_bytes(group['plan'].input_pad(group['operand']), 'little')
    transformed = np.array([code if keep else pad for code, keep in zip(reduced, admitted, strict=True)], dtype='<u2')
    other_codes = np.frombuffer(group['precision'].operand_specs[1 - group['operand']].encode(other), dtype='<u2')
    transformed_raw = materialize_pointwise_input(group['plan'], group['operand'], transformed.tobytes())
    other_raw = materialize_pointwise_input(group['plan'], 1 - group['operand'], other_codes.tobytes())
    operands = [None, None]
    operands[group['operand']] = np.frombuffer(transformed_raw, dtype='<u2')
    operands[1 - group['operand']] = np.frombuffer(other_raw, dtype='<u2')
    result_codes = []
    for left, denominator in zip(*operands, strict=True):
        reciprocal = vrecip_bf16_bounded_bits(int(denominator))
        result_codes.append((int(left) ^ reciprocal) & 0x8000 if left & 0x7fff == 0
                            else _round_bf16_product(int(left), reciprocal))
    result_raw = np.array(result_codes, dtype='<u2').tobytes()
    logical = np.frombuffer(assemble_pointwise_output(group['plan'], result_raw), dtype='<u2')
    wanted = (logical.astype('<u4') << 16).tobytes()
    _, packed_seed = materialize_reduction_inputs(second, squared.tobytes(), seed_codes.tobytes())
    payloads = {group['data']: physical(materialize_pointwise_input(first, 0, codes.tobytes())),
                group['seed']: physical(packed_seed), group['other']: physical(other_raw)}
    initial = bytearray([0xa5]) * (group['arena_bytes'] + 2048)
    for key, payload in payloads.items():
        at = group['addresses'][key] - DRAM_BASE
        initial[at:at + len(payload)] = payload
    expected = bytearray(initial)
    status = int(not all(admitted))
    for key, payload in zip(group['mutable'],
            (physical(materialize_pointwise_input(first, 0, squared.tobytes())), physical(bytes(reduction_raw)),
             physical(transformed_raw), physical(result_raw), status.to_bytes(4, 'little') + bytes(2044)), strict=True):
        at = group['addresses'][key] - DRAM_BASE
        expected[at:at + len(payload)] = payload
    return initial, bytes(expected), wanted, status


class ReductionScaleTest(unittest.TestCase):
    def test_both_operand_orders_select_the_actual_removed_guard(self):
        for operand in (0, 1):
            source, adapter, choices = case(operand)
            enlarged = [row for row in alternatives(adapter, source, choices)
                        if dict(row.metadata)['kind'] == 'adjacent_reduction_scale']
            self.assertEqual(len(enlarged), 1)
            (group,) = adapter._reduction_scale_groups.values()
            self.assertEqual(group['operand'], operand)
            self.assertEqual(group['proof']['consumer_conversion'], group['precision'].operand_specs[operand].record())
            self.assertEqual(group['proof']['pad_code'], 0 if operand == 0 else 0x3f80)
            source, adapter, choices = case(operand, escape=True)
            self.assertEqual(len(alternatives(adapter, source, choices)), 3)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'), 'select native tools')
    def test_two_page_transition_composes_with_default_initialization_and_bounds(self):
        from atlas_mlir_native.adjacent_reduction_scale import transition_source
        from atlas_mlir_native.binding import AtlasNativeBinding, NATIVE_CHECKS, NATIVE_PIPELINES, _run
        from atlas_mlir_native.pointwise import virtual_program
        from atlas_mlir_native.reduction import virtual_program as reduction_virtual
        from atlas_mlir_native.region_composition import rematerialize_scalar_constants
        from atlas_mlir_native.resident_graph import compose_fragments
        source, _, _ = case(shape=(1, 64, 32))
        group = prepare(source)
        addresses, mutable = group['addresses'], group['mutable']
        first, second = group['prefix']['plans']
        catalog = {descriptor.computation: descriptor.name for descriptor in AtlasNativeBinding.profile().descriptors}
        sources = [virtual_program(first, (addresses[group['data']],), addresses[mutable[0]], catalog['mul_bf16_zero']),
            reduction_virtual(second, (addresses[mutable[0]], addresses[group['seed']]), addresses[mutable[1]]),
            transition_source(group),
            virtual_program(group['plan'], (addresses[group['transition']], addresses[group['other']]),
                addresses[mutable[3]], [catalog[node.op] for node in group['plan'].body_request().nodes if node.effect == 'pure'])]
        tools = Path(os.environ['ATLAS_OOT_BIN_DIR'])
        virtual = rematerialize_scalar_constants(compose_fragments(sources, tools=tools, boundaries_lowered=True))
        machine = _run([str(tools / 'atlas-opt'), *NATIVE_PIPELINES['mandatory'], '-'], input_text=virtual)
        _run([str(tools / 'atlas-opt'), *NATIVE_CHECKS, '--verify-atlas-timing', '-'], input_text=machine)

    @unittest.skipUnless(os.environ.get('ATLAS_OOT_BIN_DIR'), 'select native tools')
    def test_native_c_roundtrip_internal_rejection_and_original_model(self):
        from test_looped_mnk import ROOT, build_snapshot
        from merlin.semantic_compiler.search import SearchLimits
        from test_pointwise_loop import check_source_c_call
        groups = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = build_snapshot(root)
            for operand in (0, 1):
                source, _, _ = case(operand)
                group = prepare(source)
                compiled = root / ('compiled' + str(operand))
                compile_group(group, snapshot, target_source=ROOT, destination=compiled, limits=SearchLimits())
                binary = (compiled / 'program.bin').read_bytes()
                region = export_group(group, compiled, 'pointwise', target_source=ROOT)
                package = root / ('package' + str(operand)); package.mkdir()
                for name, payload in region.files.items():
                    path = package / name; path.parent.mkdir(exist_ok=True, parents=True); path.write_bytes(payload)
                with (package / 'pointwise.c').open('a') as handle:
                    handle.write('\n#include "runtime/atlas_host.c"\n')
                data = (((np.arange(21) % 7) - 3) / 2).astype('<f4').reshape(3, 7)
                data[0, :2] = [0.0, -0.0]
                seed, other = np.array([0.5, -0.0, 1.0], dtype='<f4'), np.ones(3, dtype='<f4')
                initial, expected, wanted, status = fixture(group, data, seed, other)
                self.assertEqual(status, 0)
                bad = data.copy(); bad[0, 1] = 2.0**40
                check_source_c_call(self, package, initial, binary, lambda memory: expected,
                                    [data, seed, other], wanted, invalid_values=([bad, seed, other],))
                bad_data, bad_seed = data.copy(), seed.copy()
                if operand == 0:
                    bad_data.fill(2.0**31)
                else:
                    bad_data.fill(0); bad_seed.fill(-1)
                rejected_initial, rejected_expected, _, rejected_status = fixture(group, bad_data, bad_seed, other)
                self.assertEqual(rejected_status, 1)
                self._reject_after_launch(package, binary, [bad_data, bad_seed, other], rejected_initial, rejected_expected)
                groups.append(dict(kind='reduction_operand_' + str(operand), binary=binary.hex(), cases=[
                    dict(label='valid', memory=initial.hex(), expected_memory=expected.hex()),
                    dict(label='internal_guard_rejected', memory=rejected_initial.hex(), expected_memory=rejected_expected.hex())]))
            if os.environ.get('ATLAS_RUN_NPU_MODEL') == '1':
                spec, receipt = root / 'model-spec.json', root / 'receipt.json'
                spec.write_text(json.dumps(groups))
                process = subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],
                    str(Path(__file__).with_name('device_boundary_model.py')), str(spec), str(receipt), str(ROOT)],
                    text=True, capture_output=True, timeout=180)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                if os.environ.get('ATLAS_REDUCTION_SCALE_RECEIPT'):
                    Path(os.environ['ATLAS_REDUCTION_SCALE_RECEIPT']).write_bytes(receipt.read_bytes())

    def _reject_after_launch(self, package, binary, values, initial, expected):
        client = ctypes.CDLL(str(package / 'client.so'))
        read_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_size_t)
        write_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32)
        memory_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)
        memory, words, errors = bytearray(initial), {}, []
        polls, launches = None, 0
        def read(_, address):
            nonlocal polls
            if address == 0x40004: return len(words)
            if polls is None: return 5
            polls += 1
            return 0 if polls == 1 else 5
        def write(_, address, word):
            nonlocal polls, launches
            if 0x20000 <= address < 0x40000: words[(address - 0x20000) // 4] = word
            else:
                if (address, word) != (0x40018, 1): errors.append('CSR')
                if b''.join(words[i].to_bytes(4, 'little') for i in range(len(words))) != binary: errors.append('binary')
                if memory != initial: errors.append('input packing')
                memory[:] = expected; polls = 0; launches += 1
        def put(_, address, source, count):
            at = address - DRAM_BASE; memory[at:at + count] = ctypes.string_at(source, count); return 0
        def get(_, address, destination, count):
            at = address - DRAM_BASE; ctypes.memmove(destination, bytes(memory[at:at + count]), count); return 0
        callbacks = read_type(read), write_type(write), memory_type(put), memory_type(get)
        client.call.argtypes = [read_type, write_type, memory_type, memory_type,
                               ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        inputs = (ctypes.c_void_p * len(values))(*(value.ctypes.data for value in values))
        output = ctypes.create_string_buffer(bytes([0xa7]) * 44)
        self.assertEqual(client.call(*callbacks, inputs, ctypes.addressof(output) + 16), 2)
        self.assertEqual(launches, 1)
        self.assertEqual(errors, [])
        self.assertEqual(memory, expected)
        self.assertEqual(output.raw[:-1], bytes([0xa7]) * 44)


if __name__ == '__main__':
    unittest.main()
