"""One native counted image, checked address bindings, and original-model replay.

This suite deliberately has no environment skips: run with the selected native
tools and the Python 3.14 model environment. The 2/22-trip cases are synthetic
quantized contractions, not a complete model or a timing measurement.
"""
from copy import deepcopy
import ctypes
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.options import codegen_options
from atlas_mlir_native import quantized_loop
from atlas_mlir_native.quantized_resident import pack_resident_input
from atlas_mlir_native.resident_loop_image import ResidentLoopImage, _checked_constant_sites
from merlin.compiler.semantic.search.exploration import SearchLimits
from test_vmem_window import source_request
from test_looped_mnk import build_snapshot
from test_quantized_resident import options


ROOT = Path(__file__).resolve().parents[1]
SOPHIALAKE_BASE = 0x80000000
SOPHIALAKE_BYTES = 0x40000000


def moved_options(source, selected, trip):
    bindings = deepcopy(selected.resident_bindings)
    for row in bindings[source.digest()].values():
        row['address'] += (trip + 1) * 8192
    return replace(selected, resident_bindings=bindings)


def fixture_values(trip):
    """Small integer operands keep every MXU partial exactly BF16 representable."""
    a = np.full((3, 33), 0x38, np.uint8)
    a[1::2] = 0x40
    w = np.full((5, 33), 0x38, np.uint8)
    w[(trip % 5), :] = 0x40
    sx = np.array(trip % 3 - 1, dtype='<i4')
    sw = np.array([-1, 0, 1, 0, -1], dtype='<i4').reshape(5, 1)
    return dict(a=a, sx=sx, w=w, sw=sw)


def expected_tile(values):
    # Independent integer arithmetic and IEEE binary32 packing. These fixtures
    # need no model numeric helper, compiler reference, or deferred rounding.
    decode = {0x38: 1, 0x40: 2}
    result = bytearray(2048)
    for r in range(3):
        for c in range(5):
            total = sum(decode[int(values['a'][r, k])] * decode[int(values['w'][c, k])]
                        for k in range(33))
            value = np.float32(total * 2.0 ** (int(values['sx']) + int(values['sw'][c, 0])))
            word = int(value.view(np.uint32))
            if word & 0xffff:
                raise AssertionError('fixture arithmetic is not exactly BF16 representable')
            at = (c // 16) * 1024 + (r * 16 + c % 16) * 2
            result[at:at+2] = (word >> 16).to_bytes(2, 'little')
    return bytes(result)


class ResidentLoopImageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='atlas-resident-image-')
        cls.root = Path(os.environ.get('ATLAS_RESIDENT_IMAGE_RECEIPTS', cls.temporary.name))
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.source = source_request(3, 5, 33, batch=())
        cls.selected = replace(options(cls.source, ('w', 'sw')),
                               io_memory_capacity_base=SOPHIALAKE_BASE,
                               io_memory_capacity_bytes=SOPHIALAKE_BYTES)
        cls.compiled = cls.root / 'compiled'
        with codegen_options(cls.selected):
            cls.snapshot = build_snapshot(cls.root / 'native')
            with patch.object(quantized_loop, 'compile_source', wraps=quantized_loop.compile_source) as compile_once:
                quantized_loop.compile_source(cls.snapshot, cls.source, target_source=ROOT,
                    destination=cls.compiled, limits=SearchLimits())
                if compile_once.call_count != 1:
                    raise AssertionError('fixture must compile exactly one native counted image')
            cls.image = ResidentLoopImage(cls.source, cls.compiled)
        cls.template = (cls.compiled / 'program.bin').read_bytes()
        cls.program = json.loads((cls.compiled / 'physical-program.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def bind(self, trip):
        selected = moved_options(self.source, self.selected, trip)
        with codegen_options(selected), \
             patch.object(type(self.snapshot), 'select', side_effect=AssertionError('binding repeated selection')), \
             patch.object(quantized_loop, 'compile_source', side_effect=AssertionError('binding repeated compilation')), \
             patch.object(quantized_loop, '_run', side_effect=AssertionError('binding ran native tools')):
            bound = self.image.bind(self.source)
        self.assertNotEqual(bound.binary, self.template)
        self.assertEqual(bound.execution_plan['input_addresses'][1],
                         selected.resident_bindings[self.source.digest()]['w']['address'])
        self.assertEqual(bound.execution_plan['exponent_addresses'][1],
                         selected.resident_bindings[self.source.digest()]['sw']['address'])
        self.assertEqual(bound.receipt['binary_sha256'], sha256(bound.binary).hexdigest())
        self.assertEqual(bound.receipt['template_sha256'], sha256(self.template).hexdigest())
        self.assertEqual(bound.receipt['source_digest'], self.source.digest())
        relocated = {i for row in bound.receipt['relocations'] for i in row['word_indices']}
        changed = set()
        self.assertEqual(len(bound.binary), len(self.template))
        self.assertEqual(len(bound.physical_program['instructions']), len(self.program['instructions']))
        for index, (original, current) in enumerate(zip(self.program['instructions'],
                                                      bound.physical_program['instructions'], strict=True)):
            old, new = deepcopy(original), deepcopy(current)
            delta = old['word_u32'] ^ new['word_u32']
            if delta:
                changed.add(index)
                self.assertIn(index, relocated)
                self.assertIn(old['operation'], ('atlas.upper', 'atlas.alu_imm'))
                mask = 0xfffff000 if old['operation'] == 'atlas.upper' else 0xfff00000
                self.assertEqual(delta & ~mask, 0, 'binding changed a non-immediate instruction bit')
                for row in (old, new):
                    row.pop('word_u32'); row.pop('word_hex')
                    row['fields'].pop('immediate')
            self.assertEqual(old, new)
        self.assertTrue(changed)
        self.assertEqual(b''.join(row['word_u32'].to_bytes(4, 'little')
                                 for row in bound.physical_program['instructions']), bound.binary)
        return bound

    def test_template_and_binding_refuse_changed_source_options_and_publication(self):
        with codegen_options(self.selected):
            self.assertEqual(self.image.bind(self.source).binary, self.template)
            with self.assertRaises(Exception):
                self.image.bind(source_request(3, 6, 33, batch=()))
        for field, value in (('native_pipeline', 'optimized'), ('quantized_operand_reuse', 'row-cache')):
            with self.subTest(field=field), codegen_options(replace(self.selected, **{field: value})):
                with self.assertRaises(Exception):
                    self.image.bind(self.source)
        with codegen_options(replace(self.selected, native_pipeline='optimized')):
            with self.assertRaisesRegex(ValueError, 'compilation pipeline differs'):
                ResidentLoopImage(self.source, self.compiled)
        for field, limit in (('execution_instruction_limit', 1), ('program_size_limit_bytes', 4)):
            with self.subTest(constructor_budget=field), codegen_options(replace(self.selected, **{field: limit})):
                with self.assertRaises(Exception):
                    ResidentLoopImage(self.source, self.compiled)
        malformed = deepcopy(self.selected.resident_bindings)
        malformed[self.source.digest()]['w']['address'] += 1
        with codegen_options(replace(self.selected, resident_bindings=malformed)):
            with self.assertRaises(Exception):
                self.image.bind(self.source)
        with tempfile.TemporaryDirectory(prefix='atlas-resident-tamper-') as temporary:
            damaged = Path(temporary) / 'compiled'
            shutil.copytree(self.compiled, damaged)
            payload = bytearray((damaged / 'program.bin').read_bytes())
            payload[0] ^= 1
            (damaged / 'program.bin').write_bytes(payload)
            with codegen_options(self.selected), self.assertRaises(Exception):
                ResidentLoopImage(self.source, damaged)

    def test_partial_address_recognition_refuses_an_independent_scalar_reordering(self):
        layout = json.loads((self.compiled / 'execution_plan.json').read_text())
        addresses = {row['node_id']: row['address']
                     for row in layout['resident_bindings']}
        sites = _checked_constant_sites(self.program, addresses)
        self.assertGreaterEqual(len(sites['sw']), 2)
        upper, lower = sites['sw'][0]
        self.assertEqual(lower, upper + 1)
        following = self.program['instructions'][lower + 1]
        self.assertEqual(following['operation'], 'atlas.upper')
        self.assertNotEqual(following['fields']['dst'], self.program['instructions'][lower]['fields']['dst'])
        reordered = deepcopy(self.program)
        reordered['instructions'][lower], reordered['instructions'][lower + 1] = (
            reordered['instructions'][lower + 1], reordered['instructions'][lower])
        # The independent size LUI can execute between an address LUI/ADDI.
        # Another complete sw pair remains, so an any-match check is unsafe.
        with self.assertRaisesRegex(ValueError, 'unaccounted resident LUI'):
            _checked_constant_sites(reordered, addresses)
        # A second resident base equal to sw's rounded high must not disguise
        # that separated sw LUI as a complete upper-only materialization.
        collision = dict(addresses, high_aligned=(addresses['sw'] + 0x800) & ~0xfff)
        with self.assertRaisesRegex(ValueError, 'nonadjacent resident LUI/ADDI'):
            _checked_constant_sites(reordered, collision)
        missing_provenance = deepcopy(self.program)
        missing_provenance['instructions'][sites['w'][0][0]]['fields'].pop('atlas.virtual_cfg_source')
        with self.assertRaisesRegex(ValueError, 'resident LUI lacks virtual source provenance'):
            _checked_constant_sites(missing_provenance, addresses)

    def test_relocated_2_and_22_trip_images_execute_exactly_and_last_weight_mutation_is_observed(self):
        import torch
        model = Path(os.environ.get('ATLAS_NPU_MODEL_ROOT', os.environ['NPU_MODEL_ROOT']))
        sys.path[:0] = [str(model), str(ROOT), str(ROOT / 'tools'),
                       str(ROOT.parent / 'merlin/examples/atlas/whole-model/unchanged_fx')]
        import functional_runtime
        from functional_runtime import FunctionalModelRuntime, PhysicalFunctionalInterpreter
        torch.set_num_threads(1)
        # The callback runtime backs only the staged/loaded prefix. Configure
        # the original sparse interpreter with the entire physical board DDR
        # window, rather than narrowing its architectural map to that prefix.
        executed_maps = []
        testcase = self

        class SophiaLakeInterpreter(PhysicalFunctionalInterpreter):
            def __init__(self, config, **kwargs):
                super().__init__(replace(config, dram_base=SOPHIALAKE_BASE,
                                         dram_size=SOPHIALAKE_BYTES), **kwargs)

            def run(self, *args, **kwargs):
                result = super().run(*args, **kwargs)
                state = result.state
                testcase.assertEqual(state.cfg.dram_base, SOPHIALAKE_BASE)
                testcase.assertEqual(state.cfg.dram_size, SOPHIALAKE_BYTES)
                testcase.assertEqual(state.dram.size, SOPHIALAKE_BASE + SOPHIALAKE_BYTES)
                # Read both valid physical edges and reject each adjacent byte.
                state.read_dram(SOPHIALAKE_BASE, 1)
                state.read_dram(SOPHIALAKE_BASE + SOPHIALAKE_BYTES - 1, 1)
                with testcase.assertRaises(AssertionError):
                    state.read_dram(SOPHIALAKE_BASE - 1, 1)
                with testcase.assertRaises(AssertionError):
                    state.read_dram(SOPHIALAKE_BASE + SOPHIALAKE_BYTES, 1)
                executed_maps.append(dict(dram_base=state.cfg.dram_base,
                    dram_size=state.cfg.dram_size, dram_backing=type(state.dram).__name__,
                    sparse_pages=len(state.dram._pages), page_bytes=state.dram.PAGE_BYTES,
                    num_x_registers=state.cfg.num_x_registers,
                    num_m_registers=state.cfg.num_m_registers, vmem_size=state.cfg.vmem_size))
                return result

        self.assertEqual(self.selected.io_memory_capacity_base, SOPHIALAKE_BASE)
        self.assertEqual(self.selected.io_memory_capacity_bytes, SOPHIALAKE_BYTES)
        for trips in (2, 22):
            with self.subTest(trips=trips):
                bound = [self.bind(trip) for trip in range(trips)]
                values = [fixture_values(trip) for trip in range(trips)]
                immutable = [(row['address'], pack_resident_input(self.source, row['node_id'],
                              data[row['node_id']].tobytes()))
                             for image, data in zip(bound, values, strict=True)
                             for row in image.execution_plan['resident_bindings']]
                plans = {sha256(image.binary).hexdigest(): image.execution_plan for image in bound}
                runtime = FunctionalModelRuntime(ROOT, model, [image.binary for image in bound],
                    self.root / f'model-{trips}', memory_bytes=self.selected.io_memory_bytes,
                    memory_base=DRAM_BASE, image_plans=plans, trace_payloads=False,
                    immutable_loads=immutable, capacity_base=self.selected.io_memory_capacity_base,
                    capacity_bytes=self.selected.io_memory_capacity_bytes)

                def execute(image, data):
                    plan = image.execution_plan
                    for key, address in (('a', plan['input_addresses'][0]),
                                         ('sx', plan['exponent_addresses'][0])):
                        payload = pack_resident_input(self.source, key, data[key].tobytes())
                        staging = ctypes.create_string_buffer(payload)
                        self.assertEqual(runtime.put(None, address, ctypes.addressof(staging), len(payload)), 0)
                    for index in range(len(image.binary) // 4):
                        runtime.write(None, 0x20000 + index * 4,
                                      int.from_bytes(image.binary[index*4:index*4+4], 'little'))
                    with patch.object(functional_runtime, 'PhysicalFunctionalInterpreter', SophiaLakeInterpreter):
                        runtime.write(None, 0x40018, 1)
                    self.assertEqual(runtime.read(None, 0x40008), 0)
                    self.assertEqual(runtime.read(None, 0x40008), 5, runtime.errors)
                    at = plan['status_address'] - runtime.base
                    self.assertEqual(runtime.memory[at:at+4], bytes(4))
                    at = plan['output_addresses'][0] - runtime.base
                    actual = bytes(runtime.memory[at:at+plan['output_bytes']])
                    self.assertEqual(actual, expected_tile(data), f'EXACT mismatch in {trips}-trip replay')
                    return actual

                for image, data in zip(bound, values, strict=True):
                    before = [(address, bytes(runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)]))
                              for address, payload in immutable]
                    execute(image, data)
                    for address, payload in before:
                        self.assertEqual(bytes(runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(payload)]), payload)
                # Execute a controlled change to the last logical weight in the
                # last relocated allocation. A decorative receipt cannot pass.
                last = bound[-1]
                mutated = {key: value.copy() for key, value in values[-1].items()}
                mutated['w'][-1, -1] = 0x40 if mutated['w'][-1, -1] == 0x38 else 0x38
                self.assertNotEqual(expected_tile(mutated), expected_tile(values[-1]))
                address = last.execution_plan['input_addresses'][1]
                packed = pack_resident_input(self.source, 'w', mutated['w'].tobytes())
                runtime.memory[address-DRAM_BASE:address-DRAM_BASE+len(packed)] = packed
                execute(last, mutated)
                record = runtime.finish()
                self.assertFalse(record['errors'])
                self.assertEqual(len(record['launches']), trips + 1)
                self.assertEqual(len(record['immutable_initial_loads']), trips * 2)
                maps = executed_maps[-(trips + 1):]
                self.assertEqual(len(maps), trips + 1)
                self.assertTrue(all(row['sparse_pages'] * row['page_bytes'] < 1048576 for row in maps))
                self.assertTrue(all(row['address'] < DRAM_BASE + self.selected.io_memory_bytes
                                    for row in record['transfers'] if row['operation'] == 'write'))
                receipt = dict(schema='atlas.resident_loop_image_model_test.v1', trips=trips,
                    compile_source_calls=1, exact=True, executed_last_weight_mutation=True,
                    resident_initial_loads=len(immutable), bindings=[image.receipt for image in bound],
                    functional_instructions=[row['instructions'] for row in record['launches']],
                    backend=record['backend'], numerics=record['numerics'],
                    physical_memory_map=maps, host_backing_bytes=record['host_memory_bytes'])
                (self.root / f'resident-image-{trips}.json').write_text(json.dumps(receipt, indent=2) + '\n')
                print('RESIDENT_IMAGE_EXACT', trips, receipt['functional_instructions'], flush=True)


if __name__ == '__main__':
    unittest.main()
