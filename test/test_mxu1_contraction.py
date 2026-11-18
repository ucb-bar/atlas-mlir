"""Source-bound MXU1 QK-style conversion, batch/tail C dispatch, and model bytes."""
from __future__ import annotations

import ctypes as ct
from dataclasses import replace
import hashlib
import json
from math import prod
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.mxu1_contraction import POLICY, prepare_source, compile_source, prepare_c_package
from merlin.semantic_compiler.model import ConstantBinding, IndexMap, KernelRequest, SemanticNode, TensorType

ROOT = Path(__file__).resolve().parents[1]


def source_request(m=1, n=33, k=64, batch=()):
    rank = len(batch)+2
    dims = rank+1
    axes = tuple(tuple(int(i == j) for j in range(dims)) for i in range(dims))
    prefix = axes[:rank-2]
    maps = tuple(IndexMap(dims, (*prefix, *pair), (0,)*rank)
                 for pair in ((axes[-3], axes[-1]), (axes[-1], axes[-2]), (axes[-3], axes[-2])))
    a = SemanticNode('left', 'input', (), TensorType((*batch, m, k), 'f32', 'raw-bit-pattern'), effect='input')
    b = SemanticNode('right', 'input', (), TensorType((*batch, k, n), 'f32', 'raw-bit-pattern'), effect='input')
    seed = SemanticNode('zero', 'constant', (), TensorType((*batch, m, n), 'f32', 'raw-bit-pattern'), effect='constant')
    result = SemanticNode('output', 'source_f32_matmul_unqualified', ('left', 'right', 'zero'), seed.type,
                          (("reduction_order", "unresolved"), ("numeric_semantics", "source-f32-mul-add-unqualified-v1"),
                           ("scalar_add_order", "accumulator-product")), index_maps=(*maps, maps[-1]))
    return KernelRequest((a, b, seed, result), ('output',), ('external',), (('left', 'external'), ('right', 'external')),
                         TARGET_IDENTITY, 'diagnostic', constants=(ConstantBinding('zero', 'f32-le-raw', '00000000'*(prod(batch)*m*n)),))


def policy_for(source, exponents=(3, 2)):
    return dict(policy=POLICY, source_request_digest=source.digest(), provenance_sha256='12'*32,
                operand_encode_exponents=list(exponents))


class MXU1SourcePlanTest(unittest.TestCase):
    def test_dense_batched_maps_scales_and_round_before_flush(self):
        import numpy as np
        source = source_request(batch=(2,))
        plan = prepare_source(source, policy_for(source))
        self.assertEqual(plan.batch_count, 2)
        self.assertEqual(plan.restore_exponent, -5)
        self.assertEqual([node.op for node in plan.body.nodes if node.effect == 'pure'], ['matmul', 'matmul_accumulate'])
        spec = plan.operand_specs[0]
        values = np.array([[0, -0., 448/8, -448/8, (2**-6)*(1-2**-5)/8]], np.float32)
        self.assertEqual(spec.encode(values).tolist(), [[0, 0, 126, 254, 8]])
        self.assertEqual(spec.underflow_rounding, 'round-normal-significand-then-flush')

    def test_policy_source_seed_geometry_and_broadcast_refusals(self):
        source = source_request()
        for wrong in (policy_for(source)|{'provenance_sha256':'bad'}, policy_for(source)|{'source_request_digest':'00'*32},
                      policy_for(source)|{'operand_encode_exponents':[33, 0]}):
            with self.assertRaises(ValueError):
                prepare_source(source, wrong)
        for k in (31, 65, 288):
            current = source_request(k=k)
            with self.assertRaises(ValueError):
                prepare_source(current, policy_for(current))
        bad_seed = replace(source, constants=(ConstantBinding('zero', 'f32-le-raw', '00000080'*33),))
        with self.assertRaisesRegex(ValueError, 'positive-zero'):
            prepare_source(bad_seed, policy_for(bad_seed))
        batch = source_request(batch=(2,))
        wrong = replace(batch.node('right'), type=replace(batch.node('right').type, shape=(1, 64, 33)))
        changed = replace(batch, nodes=tuple(wrong if node.id == 'right' else node for node in batch.nodes))
        with self.assertRaisesRegex(ValueError, 'identical dense batches'):
            prepare_source(changed, policy_for(changed))


@unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'), 'select native tools')
class MXU1SourceNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
        from merlin.semantic_compiler.search import SearchLimits
        retained = os.environ.get('ATLAS_MXU1_SOURCE_OUT')
        cls.temporary = None if retained else tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR'))
        cls.work = Path(retained or cls.temporary.name)
        cls.work.mkdir(exist_ok=True)
        cls.snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=cls.work/'snapshot',
            crate=packaged_egg_bridge_path(), cargo_target_dir=Path(os.environ.get('ATLAS_MXU1_CARGO_TARGET', cls.work/'cargo')),
            source_revision='source-bound-model47-mxu1')
        cls.cases = []
        for m, n, batch in ((1, 33, ()), (3, 33, (2,))):
            source = source_request(m=m, n=n, batch=batch)
            policy = policy_for(source)
            case = cls.work/('dense' if not batch else 'batch2')
            compile_source(cls.snapshot, source, policy, target_source=ROOT, destination=case,
                           limits=SearchLimits(candidate_nodes=256, wall_timeout_s=120))
            record, files = prepare_c_package(source, case, 'layer', policy=policy, target_source=ROOT)
            package = case/'c-package'
            package.mkdir()
            for filename, payload in files.items():
                path = package/filename
                path.parent.mkdir(exist_ok=True, parents=True)
                path.write_bytes(payload)
            cls.cases.append((source, policy, case, package, record))

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def test_c_package_native_and_rv64_compile_and_tamper_refusal(self):
        for source, policy, case, package, record in self.cases:
            self.assertEqual(record['launches_per_call'], prepare_source(source, policy).launches_per_call)
            for compiler, flags, output in ((shutil.which('cc'), ['-fPIC'], 'native.o'),
                    (str(Path(os.environ['ATLAS_LLVM_BIN'])/'clang'), ['--target=riscv64-unknown-elf', '-march=rv64gcv', '-mabi=lp64d', '-ffreestanding', '-Wno-gcc-install-dir-libstdcxx'], 'rv64.o')):
                subprocess.run([compiler, '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', *flags, '-c', 'layer.c', '-o', output], cwd=package, check=True, capture_output=True)
            with self.assertRaises(ValueError):
                prepare_c_package(source, case, 'changed', policy=policy|{'operand_encode_exponents':[0, 0]}, target_source=ROOT)

    @unittest.skipUnless(os.environ.get('ATLAS_MXU1_MODEL') == '1', 'select original47')
    def test_generated_dense_and_batched_tail_calls_exact_atomic_outputs(self):
        for source, policy, case, package, record in self.cases:
            self.execute(source, policy, case, package, record)

    def execute(self, source, policy, case, package, record):
        import numpy as np
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        from atlas_mlir_native.reference import mxu1_cap448_lane
        from test_mxu1_cap448 import authenticate_model, MODEL_PIN
        plan = prepare_source(source, policy)
        before = authenticate_model(Path(os.environ['NPU_MODEL_ROOT']))
        torch.set_num_threads(1)
        (package/'shim.c').write_text('''#include "layer.h"
static atlas_host host;
static layer_context region;
static layer_source_context source;
static uint8_t raw[LAYER_WORKSPACE_BYTES], converted[LAYER_SOURCE_WORKSPACE_BYTES];
int bind(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get) {
  if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},
     (atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0})) return -1;
  if(layer_bind(&region,&host,(atlas_tile_memory_io){put,get,0},raw,sizeof(raw),20)) return -1;
  return layer_source_bind(&source,&region,converted,sizeof(converted));
}
int call(const float *a,const float *b,float *out) {
 const float *inputs[]={a,b}; return layer_source_run(&source,inputs,out,0);
}
int dispatch(const void *a,const void *b,void *out) {
 const void *inputs[]={a,b};void *outputs[]={out};return layer_source_dispatch(&source,inputs,outputs);
}
''')
        subprocess.run([shutil.which('cc'), '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-shared', '-fPIC',
                        'layer.c', 'runtime/atlas_host.c', 'shim.c', '-o', 'layer.so'], cwd=package, check=True, capture_output=True)
        lib = ct.CDLL(str(package/'layer.so'))
        READ = ct.CFUNCTYPE(ct.c_uint32, ct.c_void_p, ct.c_size_t)
        WRITE = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_size_t, ct.c_uint32)
        MEMORY = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t)
        lib.bind.argtypes = [READ, WRITE, MEMORY, MEMORY]
        lib.call.argtypes = lib.dispatch.argtypes = [ct.c_void_p]*3
        inventory = json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
        binary = (case/'program.bin').read_bytes()
        decoded = decode_binary(inventory, binary)
        dram = bytearray([0xa5])*(32*2048)
        imem = {}
        control = dict(starts=0, running=False, polls=0, fail_read_at=None)
        errors, cycles = [], []
        output_offset = record['device_output_address']-0x90000000
        def read(_, address):
            if address == 0x40004: return control['starts']
            if address != 0x40008:
                errors.append('bad CSR read');return 3
            if not control['running']: return 5
            control['polls'] += 1
            if control['polls'] == 1: return 0
            control['running'] = False
            return 5 if not errors else 3
        def write(_, address, word):
            if 0x20000 <= address < 0x40000:
                if address == 0x20000: imem.clear()
                imem[(address-0x20000)//4] = word;return
            if address != 0x40018 or word != 1:
                errors.append('bad launch');return
            control.update(starts=control['starts']+1, running=True, polls=0)
            try:
                assert b''.join(imem[i].to_bytes(4,'little') for i in range(len(imem))) == binary
                initial = bytes(dram)
                device = InstantiableProgram([model_instruction(row, inventory, isa) for row in decoded['instructions']])
                device.dram_base = 0x90000000
                device.memory_regions = [(0, torch.tensor(list(initial), dtype=torch.uint8))]
                config = DefaultHardwareConfig()
                config.arch_state_config = replace(config.arch_state_config, numerics='rtl', randomize_init=True, init_seed=217)
                simulation = Simulation(config, LoggerConfig(filename=str(package/f"model-{control['starts']}.json")), device, verbose=False)
                state = simulation.core.arch_state
                try:
                    state.vmem.fill_(0xcd)
                    simulation.run(max_cycles=100000)
                    assert state.halted and simulation.get_stats().runtime_errors == 0
                    actual = bytes(state.read_dram(device.dram_base, len(dram)).tolist())
                    assert actual[:output_offset] == initial[:output_offset]
                    assert actual[output_offset+2048:] == initial[output_offset+2048:]
                    dram[:] = actual
                    cycles.append(simulation.get_stats().cycles)
                finally: state.close()
            except Exception as error: errors.append(repr(error))
        def put(_, address, value, count):
            if control['running']: errors.append('early input overwrite');return 1
            offset = address-0x90000000
            dram[offset:offset+count] = ct.string_at(value,count);return 0
        def get(_, address, value, count):
            if control['running']: errors.append('early readback');return 1
            if control['starts'] == control['fail_read_at']: return 1
            offset = address-0x90000000
            ct.memmove(value,bytes(dram[offset:offset+count]),count);return 0
        callbacks = READ(read), WRITE(write), MEMORY(put), MEMORY(get)
        self.assertEqual(lib.bind(*callbacks),0)
        a = (np.arange(plan.batch_count*plan.m*plan.k,dtype=np.float32)%13-6).reshape(source.node('left').type.shape)/8
        b = (np.arange(plan.batch_count*plan.k*plan.n,dtype=np.float32)%11-5).reshape(source.node('right').type.shape)/8
        qa,qb = (spec.encode(value).reshape(plan.batch_count,*value.shape[-2:]) for spec,value in zip(plan.operand_specs,(a,b),strict=True))
        expected = []
        for batch in range(plan.batch_count):
            for row in range(plan.m):
                for col in range(plan.n):
                    value = 0
                    for kt in range(plan.k//32):
                        value = mxu1_cap448_lane(tuple(map(int,qa[batch,row,kt*32:(kt+1)*32])),
                                                  tuple(map(int,qb[batch,kt*32:(kt+1)*32,col])),value)
                    if value:
                        value=(value&0x807f)|((((value>>7)&255)+plan.restore_exponent)<<7)
                    expected.append(value<<16)
        wanted = struct.pack('<'+'I'*len(expected),*expected)
        output = (ct.c_uint8*(len(wanted)+32))(*([0xcd]*(len(wanted)+32)))
        pointer = ct.addressof(output)+16
        for call in (lib.call,lib.dispatch):
            self.assertEqual(call(a.ctypes.data,b.ctypes.data,pointer),0,errors)
            self.assertEqual(bytes(output)[16:-16],wanted)
            self.assertEqual(bytes(output)[:16]+bytes(output)[-16:],b'\xcd'*32)
        launches = control['starts']
        bad = a.copy();bad.flat[0] = np.nan
        preserved = bytes(output)
        self.assertEqual(lib.call(bad.ctypes.data,b.ctypes.data,pointer),2)
        self.assertEqual(control['starts'],launches)
        self.assertEqual(bytes(output),preserved)
        self.assertNotEqual(lib.call(a.ctypes.data,b.ctypes.data,a.ctypes.data),0)
        self.assertEqual(control['starts'],launches)
        ct.memset(pointer,0xcd,len(wanted))
        control['fail_read_at'] = control['starts']+2
        self.assertEqual(lib.call(a.ctypes.data,b.ctypes.data,pointer),3)
        self.assertEqual(bytes(output)[16:-16],b'\xcd'*len(wanted))
        self.assertEqual(bytes(output)[:16]+bytes(output)[-16:],b'\xcd'*32)
        self.assertFalse(errors, errors)
        after = authenticate_model(Path(os.environ['NPU_MODEL_ROOT']))
        self.assertTrue(all(after.get(name)==value for name,value in before.items()))
        receipt = dict(status='PASS',model_commit=MODEL_PIN,numerics='rtl',model_source_sha256=after,
            shape=[plan.batch_count,plan.m,plan.n,plan.k],program_sha256=hashlib.sha256(binary).hexdigest(),
            program_words=len(binary)//4,launches=len(cycles),model_cycles=cycles,final_elements_checked=len(expected),
            output_bytes_per_success=len(wanted),full_output_exact=True,device_non_output_guards_preserved=True,
            caller_guards_preserved=True,late_failure_output_unchanged=True,nonfinite_and_overlap_refused_before_launch=True)
        (case/'model-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    unittest.main()
