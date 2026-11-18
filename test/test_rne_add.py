"""Compile pair-row RNE addition and execute its generated C publication path."""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from atlas_mlir_native.binding import AtlasNativeBinding, DRAM_BASE, TARGET_IDENTITY
from atlas_mlir_native.rne_add import RNE_ADD_POLICY, compile_source, prepare_c_package, prepare_source
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

ROOT = Path(__file__).resolve().parents[1]
MODEL_PIN = "b97b2f5a97e432141c4ef8504bec396651d77a8f"


def source(shape=(1, 35)):
    type_ = TensorType(shape, "f32", "raw-bit-pattern")
    return KernelRequest((SemanticNode("a", "input", (), type_, effect="input"),
        SemanticNode("b", "input", (), type_, effect="input"),
        SemanticNode("sum", "f32_add", ("a", "b"), type_,
            (("numeric_semantics", "ieee754-f32-add-rne-v1"),),
            index_maps=pointwise_identity_maps(len(shape), 2))), ("sum",), ("external",),
        (("a", "external"), ("b", "external")), TARGET_IDENTITY, "diagnostic")


def rounded_add(a, b):
    """Exact integer sum and one RNE step, independent of device reduction."""
    def units(code):
        exponent = (code >> 7) & 255
        value = ((code & 127) | (128 if exponent else 0)) << (exponent-1 if exponent else 0)
        return -value if code & 32768 else value
    value = units(a) + units(b)
    if not value:
        return (a & b & 32768) if not ((a | b) & 32767) else 0
    magnitude, sign = abs(value), 32768 if value < 0 else 0
    shift = max(0, magnitude.bit_length()-8)
    high = magnitude >> shift
    if shift:
        low, half = magnitude & ((1 << shift)-1), 1 << (shift-1)
        high += low > half or low == half and high & 1
    if high == 256:
        high >>= 1
        shift += 1
    return sign | ((shift+1) << 7) | (high-128) if high >= 128 else sign | high


class RNEAddPolicyTest(unittest.TestCase):
    def test_source_accounting_and_fixed_batch(self):
        precision = prepare_source(source(), policy=RNE_ADD_POLICY)
        self.assertEqual(precision.count, 35)
        self.assertEqual(precision.request.nodes[0].type.shape, (32, 2))
        self.assertEqual(precision.record()["launch_count"], 2)
        self.assertEqual(precision.reduction_source.source_identity, source().digest())
        with self.assertRaisesRegex(ValueError, "explicitly selected"):
            prepare_source(source(), policy="old-chopped-add")
        bad = source()
        bad = dataclasses.replace(bad, nodes=(*bad.nodes[:-1], dataclasses.replace(bad.nodes[-1], op="f32_sub")))
        with self.assertRaisesRegex(ValueError, "accounting"):
            prepare_source(bad, policy=RNE_ADD_POLICY)

    def test_rounding_discriminators_and_signed_zero_contract(self):
        self.assertEqual(rounded_add(0x3f80, 0x3bc0), 0x3f81)
        self.assertEqual(rounded_add(0x3f80, 0x3b80), 0x3f80)
        self.assertEqual(rounded_add(0x3f81, 0x3b80), 0x3f82)
        self.assertEqual(rounded_add(0xbf80, 0xbbc0), 0xbf81)
        self.assertEqual(rounded_add(0x3f80, 0xbf80), 0)
        self.assertEqual(rounded_add(0x8000, 0x8000), 0x8000)
        self.assertEqual(rounded_add(0x8000, 0), 0)


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"), "select native tools")
class RNEAddNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.target_binding import verify_native_publication
        retained = os.environ.get("ATLAS_RNE_ADD_TEST_OUT")
        cls.temporary = None if retained else tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        cls.work = Path(retained or cls.temporary.name)
        cls.work.mkdir(exist_ok=True)
        snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=cls.work/"snapshot",
            crate=packaged_egg_bridge_path(), cargo_target_dir=Path(os.environ.get("ATLAS_RNE_ADD_CARGO_TARGET", str(cls.work/"cargo"))),
            source_revision="generic-rne-add-test")
        cls.captured = source()
        cls.precision = prepare_source(cls.captured, policy=RNE_ADD_POLICY)
        cls.compiled = cls.work/"compiled"
        manifest = compile_source(snapshot, cls.captured, policy=RNE_ADD_POLICY, target_source=ROOT,
            destination=cls.compiled, limits=SearchLimits(candidate_nodes=256))
        verify_native_publication(cls.compiled, manifest, engine="merlin_native",
            request_digest=cls.precision.request.digest(), target_identity=TARGET_IDENTITY)
        cls.record, files = prepare_c_package(cls.captured, cls.compiled, "pointwise", policy=RNE_ADD_POLICY,
            target_source=ROOT, runtime_linkage="embedded")
        cls.package = cls.work/"package"
        cls.package.mkdir()
        for name, data in files.items():
            path = cls.package/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (cls.package/"client.c").write_text('''#include "pointwise.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const float *const *inputs,float *output) {
  atlas_host host; pointwise_context raw; pointwise_source_context source;
  unsigned char scratch[POINTWISE_WORKSPACE_BYTES];
  _Alignas(4) unsigned char conversion[POINTWISE_SOURCE_WORKSPACE_BYTES];
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
      pointwise_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8) ||
      pointwise_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
  return pointwise_source_run(&source,inputs,output,0);
}''')
        result = subprocess.run([shutil.which("cc"), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
            "-shared", "-fPIC", "pointwise.c", "client.c", "-o", "client.so"], cwd=cls.package, capture_output=True)
        if result.returncode:
            raise RuntimeError(result.stderr.decode())
        cls.binary = (cls.compiled/"program.bin").read_bytes()
        cls.execution = json.loads((cls.compiled/"execution_plan.json").read_text())

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def test_generated_c_tails_refusals_and_atomic_output(self):
        import numpy as np
        palette = ((0x3f80,0x3bc0),(0x3f80,0x3b80),(0x3f81,0x3b80),(0xbf80,0xbbc0),
                   (0x3f80,0xbf80),(0x8000,0),(0x8000,0x3f80),(0x2000,0xa000),(0x5f00,0xdf00))
        pairs = [palette[i % len(palette)] for i in range(35)]
        arrays = [(np.array([pair[j] for pair in pairs], dtype="<u2").astype("<u4") << 16).view("<f4") for j in (0,1)]
        expected = b"".join(struct.pack("<I", rounded_add(a,b) << 16) for a,b in pairs)
        self._call(arrays, expected)
        invalid = [value.copy() for value in arrays]
        invalid[0][-1] = invalid[1][-1] = -0.0
        self._call(invalid, None)
        for value in (np.nan, np.inf, 2.0**-64, 2.0**64):
            invalid = [item.copy() for item in arrays]
            invalid[0][-1] = value
            self._call(invalid, None)
        self._call(arrays, None, fail_second=True)

    def _call(self, arrays, expected, *, fail_second=False):
        library = ctypes.CDLL(str(self.package/"client.so"))
        rd = ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_size_t)
        wr = ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32)
        mem = ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t)
        memory, words, errors, calls = bytearray([0xa5])*(128*2048), {}, [], []
        state = {"polls":None,"fault":False}
        def read(_, address):
            if address == 0x40004:
                return len(words)
            if address != 0x40008:
                errors.append("unexpected CSR read");return 3
            if state["polls"] is None:
                return 5
            state["polls"] += 1
            return 0 if state["polls"] == 1 else 3 if state["fault"] else 5
        def write(_, address, value):
            if 0x20000 <= address < 0x40000:
                words[(address-0x20000)//4] = value
                return
            try:
                self.assertEqual((address,value),(0x40018,1))
                self.assertEqual(b"".join(words[i].to_bytes(4,"little") for i in range(len(words))),self.binary)
                calls.append(True)
                state["fault"] = fail_second and len(calls) == 2
                if not state["fault"]:
                    memory[:] = self._execute(memory)
                state["polls"] = 0
            except Exception as error:
                errors.append(repr(error));state["fault"]=True;state["polls"]=0
        def transfer(address,pointer,size,put):
            at=address-DRAM_BASE
            if not 0 <= at <= len(memory)-size:
                return 1
            if put: memory[at:at+size]=ctypes.string_at(pointer,size)
            else: ctypes.memmove(pointer,bytes(memory[at:at+size]),size)
            return 0
        callbacks=(rd(read),wr(write),mem(lambda _,a,p,n:transfer(a,p,n,True)),mem(lambda _,a,p,n:transfer(a,p,n,False)))
        library.call.argtypes=[rd,wr,mem,mem,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
        inputs=(ctypes.c_void_p*2)(*(value.ctypes.data for value in arrays))
        before=[value.tobytes() for value in arrays]
        output=ctypes.create_string_buffer(bytes([0xa7])*172)
        result=library.call(*callbacks,inputs,ctypes.addressof(output)+16)
        self.assertEqual(errors,[])
        self.assertEqual([value.tobytes() for value in arrays],before)
        self.assertEqual(output.raw[:16],bytes([0xa7])*16)
        self.assertEqual(output.raw[156:172],bytes([0xa7])*16)
        if expected is None:
            self.assertEqual(result,4 if fail_second else 2)
            self.assertEqual(output.raw[16:156],bytes([0xa7])*140)
            self.assertEqual(len(calls),2 if fail_second else 0)
        else:
            self.assertEqual(result,0)
            self.assertEqual(output.raw[16:156],expected)
            self.assertEqual(len(calls),2)

    def _execute(self, memory):
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        addresses=self.execution["input_addresses"]
        first=addresses[0]-DRAM_BASE
        seed=addresses[1]-DRAM_BASE
        self.assertEqual(memory[seed:seed+2048],bytes(2048))
        expected=bytearray(memory)
        codes=[]
        for row in range(32):
            a,b=struct.unpack_from("<HH",memory,first+row*32)
            codes.extend([rounded_add(a,b)]*32)
        raw=pack_bf16_pair_halves(struct.pack("<1024H",*codes))
        out=self.execution["output_addresses"][0]-DRAM_BASE
        expected[out:out+2048]=raw
        if os.environ.get("ATLAS_RNE_ADD_MODEL") == "1":
            import torch
            from npu_model.configs import isa_definition as isa
            from npu_model.configs.hardware.default import DefaultHardwareConfig
            from npu_model.logging import LoggerConfig
            from npu_model.simulation import Simulation
            from npu_model.software.program import InstantiableProgram
            from tools.check_program_decode import decode_binary
            from tools.model_instructions import model_instruction
            self.assertEqual(subprocess.check_output(["git","-C",os.environ["NPU_MODEL_ROOT"],"rev-parse","HEAD"],text=True).strip(),MODEL_PIN)
            torch.set_num_threads(1)
            inventory=json.loads((ROOT/"docs/selected-variant-inventory.json").read_text())
            program=InstantiableProgram([model_instruction(row,inventory,isa) for row in decode_binary(inventory,self.binary)["instructions"]])
            program.dram_base=DRAM_BASE
            program.memory_regions=[(0,torch.tensor(list(memory),dtype=torch.uint8))]
            config=DefaultHardwareConfig()
            config.arch_state_config=dataclasses.replace(config.arch_state_config,numerics="rtl",randomize_init=True,init_seed=631)
            sim=Simulation(config,LoggerConfig(filename=str(self.work/"model.json")),program,verbose=False)
            arch=sim.core.arch_state
            try:
                arch.vmem.fill_(0xcd);sim.run(max_cycles=100000)
                self.assertTrue(arch.halted)
                self.assertEqual(sim.get_stats().runtime_errors,0)
                actual=bytes(arch.read_dram(DRAM_BASE,len(memory)).tolist())
                self.assertEqual(actual,bytes(expected))
                receipt=dict(status="PASS",model_commit=MODEL_PIN,numerics="rtl",program_sha256=hashlib.sha256(self.binary).hexdigest(),
                    program_words=len(self.binary)//4,cycles=sim.get_stats().cycles,output_bytes=2048,full_arena_bytes=len(memory))
                with (self.work/"model-receipts.jsonl").open("a") as file:
                    file.write(json.dumps(receipt)+"\n")
                return actual
            finally:
                arch.close()
        return expected
