"""Ordinary virtual activation quantizer source and compilation checks."""
from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from atlas_mlir_native.activation_quantize_device import prepare_plan, virtual_program
from test_already_quantized import quantizer_source


ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('ATLAS_OOT_BIN_DIR', ROOT / 'build/bin'))


def dynamic_source(shape=(2, 5, 35), reduce='head_block', rule='nc', rows=3, dtype='bf16'):
    source = quantizer_source(dtype=dtype, count=1)
    attrs = dict(source.nodes[1].attrs)
    attrs.pop('static_exp')
    attrs.update(mode='dynamic', semantic_kind='dynamic_activation_quantizer', reduce=reduce, rule=rule, rows=rows)
    exp_shape = () if reduce == 'tensor' else (*shape[:-2], 1, 1) if reduce == 'head' else (*shape[:-1], 1)
    x, code, exp = source.nodes
    return replace(source, nodes=(replace(x, type=replace(x.type, shape=shape)),
                                 replace(code, type=replace(code.type, shape=shape), attrs=tuple(sorted(attrs.items()))),
                                 replace(exp, type=replace(exp.type, shape=exp_shape), attrs=tuple(sorted(attrs.items())))))


def host_reference(source, raw):
    from atlas_mlir_native.static_quantize import emit_c as static
    from atlas_mlir_native.dynamic_quantize import emit_c as dynamic
    plan = prepare_plan(source)
    emit = static if plan.attrs['mode'] == 'static' else dynamic
    with tempfile.TemporaryDirectory(prefix='atlas-quant-host-') as directory:
        _, files = emit(source, 'reference')
        directory = Path(directory)
        for name, data in files.items():
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_bytes(data)
        subprocess.run([shutil.which('cc'), '-shared', '-fPIC', '-O2', str(directory / 'reference.c'),
                        '-o', str(directory / 'reference.so')], check=True, capture_output=True)
        library = ctypes.CDLL(str(directory / 'reference.so'))
        function = library.reference
        function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
        input_ = ctypes.create_string_buffer(raw)
        codes = ctypes.create_string_buffer(plan.count)
        exps = ctypes.create_string_buffer(int(np.prod(plan.exponents.type.shape)) * 4)
        outputs = [codes if key == plan.codes.id else exps for key in source.outputs]
        pointers = (ctypes.c_void_p * 2)(*(ctypes.addressof(value) for value in outputs))
        assert function(None, (ctypes.c_void_p * 1)(ctypes.addressof(input_)), pointers) == 0
        return codes.raw, exps.raw


@unittest.skipUnless((BIN/'atlas-opt').is_file(), 'build Atlas tools')
class QuantizerVirtualPipelineTest(unittest.TestCase):
    def test_counted_loop_source_has_no_machine_resources(self):
        source = dynamic_source(shape=(1, 64), reduce='tensor')
        text = virtual_program(prepare_plan(source))
        for required in ('^group(', '^lane(', 'virtual_dma_load_bf16', 'virtual_vpu_reduce', 'virtual_raw_pack_fp8', 'virtual_slot_load', 'cf.cond_br'):
            self.assertIn(required, text)
        for forbidden in ('"atlas.delay"', '"atlas.dma_wait"', '"atlas.alu_reg"', 'virtual_activation_quantize'):
            self.assertNotIn(forbidden, text)

    def test_full_tiles_and_indexed_counted_lanes_keep_program_bounded(self):
        text = virtual_program(prepare_plan(dynamic_source(shape=(8, 64, 64), reduce='head')))
        self.assertIn('arith.constant 2048 : i32', text)
        self.assertIn('arith.constant 1024 : i32', text)
        self.assertIn('^lane(', text)
        self.assertEqual(text.count('"atlas.virtual_slot_load"'), 3)
        self.assertEqual(text.count('"atlas.virtual_slot_store"'), 1)
        single = virtual_program(prepare_plan(dynamic_source(shape=(1, 64, 64), reduce='head')))
        self.assertEqual(len(text.splitlines()), len(single.splitlines()))

    def test_regular_compiler_runs_all_source_contracts(self):
        for source in (quantizer_source(count=2048), dynamic_source(shape=(1, 4096), reduce='tensor')):
            result = subprocess.run([str(BIN/'atlas-opt'), '--atlas-compile', '-'],
                                    input=virtual_program(prepare_plan(source)), text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for contract in ('virtual_dma_contract', 'virtual_mxu_contract', 'virtual_tile_contract',
                             'virtual_cfg_contract', 'virtual_source_memory_contract', 'virtual_buffer_contract'):
                self.assertIn(contract, result.stdout)

    def test_unsupported_bodies_are_explicit(self):
        for source in (quantizer_source(count=35), quantizer_source(dtype='f32',count=32)):
            with self.assertRaisesRegex(ValueError, 'aligned BF16 group'):
                virtual_program(prepare_plan(source))

    def test_raw_pack_scale_and_scratch_mutations_fail_normal_contracts(self):
        result = subprocess.run([str(BIN/'atlas-opt'), '--atlas-compile', '-'],
                                input=virtual_program(prepare_plan(quantizer_source(count=64))),
                                text=True, capture_output=True, check=True)
        original = result.stdout
        store = next(line for line in original.splitlines() if '"atlas.scalar_store"' in line)
        scale = next(line for line in original.splitlines() if 'kind = "seld"' in line)
        for before, after in ((store, re.sub(r'src = \d+ : i32', 'src = 0 : i32', store)),
                              (scale, scale.replace('offset = 0 : i32', 'offset = 4 : i32'))):
            changed = original.replace(before, after, 1)
            self.assertNotEqual(changed, original)
            checked = subprocess.run([str(BIN/'atlas-opt'), '--verify-atlas-generated-schedule', '-'],
                                     input=changed, text=True, capture_output=True)
            self.assertNotEqual(checked.returncode, 0)
            self.assertIn('contract', checked.stderr)


if __name__ == '__main__':
    unittest.main()
