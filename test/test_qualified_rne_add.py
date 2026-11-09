"""Independent integer RNE oracle versus the selected primitive expression DAG."""

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.qualified_rne_add import input_spec, prepare_plan, prepare_source
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.reference import TensorValue, evaluate_graph


def source_request(dtype="bf16", shape=(32, 32), *, repeated=False):
    tensor = TensorType(shape, dtype, "raw-bit-pattern")
    inputs = (SemanticNode("a", "input", (), tensor, effect="input"),)
    if not repeated:
        inputs += (SemanticNode("b", "input", (), tensor, effect="input"),)
    result = SemanticNode("sum", dtype + "_add", ("a", "a" if repeated else "b"), tensor,
                          (("numeric_semantics", f"ieee754-{dtype}-add-rne-v1"),),
                          index_maps=pointwise_identity_maps(len(shape), 2))
    return KernelRequest((*inputs, result), (result.id,), ("external",),
                         tuple((node.id, "external") for node in inputs), TARGET_IDENTITY, "diagnostic")


def integer_rne_add(left, right):
    """Exact integer-aligned BF16 sum, then one independent ties-even rounding."""
    def component(code):
        magnitude = code & 0x7FFF
        return ((128 + (magnitude & 127)) * (-1 if code & 0x8000 else 1),
                (magnitude >> 7) - 134) if magnitude else (0, -133)

    a, ae = component(left)
    b, be = component(right)
    exponent = min(ae, be)
    total = (a << (ae - exponent)) + (b << (be - exponent))
    if not total:
        return 0x8000 if left == right == 0x8000 else 0
    sign = 0x8000 if total < 0 else 0
    magnitude = abs(total)
    shift = magnitude.bit_length() - 8
    if shift > 0:
        significand, remainder = divmod(magnitude, 1 << shift)
        half = 1 << (shift - 1)
        significand += remainder > half or remainder == half and bool(significand & 1)
    else:
        significand = magnitude << -shift
    if significand == 256:
        significand = 128
        shift += 1
    return sign | ((exponent + shift + 134) << 7) | (significand - 128)


def evaluate_body(body, pairs):
    """Interpret selected primitive nodes only; no composite RNE callback exists."""
    from atlas_mlir_native.reference import (
        evaluate_bf16_binary_rne, evaluate_bf16_binary_zero, evaluate_bf16_max,
        evaluate_bf16_recip, evaluate_bf16_tanh,
    )

    callbacks = {
        (op, "raw-bit-pattern"): callback for ops, callback in (
            (("add_bf16_rne", "sub_bf16_rne"), evaluate_bf16_binary_rne),
            (("mul_bf16_zero",), evaluate_bf16_binary_zero),
            (("max_bf16",), evaluate_bf16_max),
            (("recip_bf16_bounded",), evaluate_bf16_recip),
            (("tanh_bf16",), evaluate_bf16_tanh),
        ) for op in ops
    }
    expected_inputs = [node for node in body.nodes if node.effect == "input"]
    tensor = body.nodes[-1].type
    padded = tuple(pairs) + ((0, 0),) * (1024 - len(pairs))
    inputs = {node.id: TensorValue(tensor, tuple(pair[i] for pair in padded))
              for i, node in enumerate(expected_inputs)}
    constants = {
        binding.node_id: TensorValue(tensor, tuple(int.from_bytes(bytes.fromhex(binding.data_hex)[i:i + 2], "little")
                                                  for i in range(0, len(binding.data_hex) // 2, 2)))
        for binding in body.constants
    }
    return evaluate_graph(body, inputs, constants=constants, primitives=callbacks)[0].elements[:len(pairs)]


class RneAddTest(unittest.TestCase):
    def test_source_precision_is_explicit_and_domain_is_bounded(self):
        import numpy as np

        for dtype in ("f32", "bf16"):
            precision = prepare_source(source_request(dtype, (3, 7)))
            self.assertEqual(precision.record()["arithmetic"], "ieee754-bf16-add-rne-v1")
            self.assertEqual(prepare_plan(precision.request).tile_shape, (32, 32))
            self.assertTrue(precision.record()["qualification"].startswith(
                "conditional_source_equivalence" if dtype == "bf16" else "diagnostic_precision_change"))
        input_spec().encode(np.array([0, -0.0, 2.0**-32, -(2.0**32)], dtype=np.float32))
        for value in (2.0**-33, 2.0**33, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                input_spec().encode(np.array([value], dtype=np.float32))
        source = source_request()
        for node in (replace(source.nodes[-1], op="bf16_sub"),
                     replace(source.nodes[-1], attrs=(("numeric_semantics", "chop"),))):
            with self.assertRaises(ValueError):
                prepare_source(replace(source, nodes=(*source.nodes[:-1], node)))
        repeated = prepare_source(source_request(repeated=True))
        self.assertEqual(len(repeated.operand_specs), 1)
        self.assertEqual(len([node for node in prepare_plan(repeated.request).body_request().nodes
                              if node.effect == "input"]), 1)

    def test_body_uses_only_existing_provider_admitted_primitives(self):
        from atlas_mlir_native.binding import _bf16_tile_request

        body = prepare_plan(prepare_source(source_request()).request).body_request()
        self.assertTrue(_bf16_tile_request(body))
        self.assertNotIn("rne_add_bf16", {node.op for node in body.nodes})
        self.assertEqual({node.op for node in body.nodes if node.effect == "pure"},
                         {"add_bf16_rne", "sub_bf16_rne", "mul_bf16_zero", "max_bf16",
                          "recip_bf16_bounded", "tanh_bf16"})

    def test_pointwise_requires_explicit_add_policy_and_preserves_streaming(self):
        from atlas_mlir_native.pointwise import (
            is_pointwise_request, prepare_execution_plan, prepare_source as pointwise_source,
        )
        from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY

        for dtype in ("f32", "bf16"):
            source = source_request(dtype, (128, 2048))
            precision = pointwise_source(source, rne_add_policy=RNE_ADD_POLICY)
            self.assertTrue(is_pointwise_request(precision.request))
            full, device, streaming = prepare_execution_plan(precision.request)
            self.assertIsNotNone(streaming)
            self.assertEqual(full.body_request().digest(), device.body_request().digest())
            with self.assertRaisesRegex(ValueError, "RNE add precision policy"):
                pointwise_source(source, rne_add_policy="chop")
        self.assertEqual(pointwise_source(source_request("f32")).request.nodes[-1].op,
                         "add_bf16_rne")

    def test_edge_cases_ties_cancellation_and_signed_zero(self):
        pairs = [(a, b) for a in (0, 0x8000) for b in (0, 0x8000)]
        for exponent in (-32, -1, 0, 1, 31):
            base = (exponent + 127) << 7
            half_ulp = (exponent - 8 + 127) << 7
            for fraction in (0, 1, 2, 63, 64, 126, 127):
                code = base | fraction
                pairs += [(code, half_ulp), (code, half_ulp - 1), (code, half_ulp + 1),
                          (code | 0x8000, half_ulp | 0x8000),
                          (code, code | 0x8000), (code, (code - 1) | 0x8000),
                          (code | 0x8000, code - 1), (code, 0), (0x8000, code)]
        # Tiny tie operands below the stated input floor are excluded.
        pairs = [(a, b) for a, b in pairs if all(
            not code & 0x7FFF or 0x2F80 <= code & 0x7FFF <= 0x4F80 for code in (a, b))]
        body = prepare_plan(prepare_source(source_request()).request).body_request()
        actual = evaluate_body(body, pairs)
        self.assertEqual(actual, tuple(integer_rne_add(a, b) for a, b in pairs))
        self.assertEqual(actual[:4], (0, 0, 0, 0x8000))

    def test_random_bounded_pairs_and_exponent_gap_extremes(self):
        rng = random.Random(812749)
        codes = [0, 0x8000, *(sign | code for sign in (0, 0x8000)
                            for code in range(0x2F80, 0x4F81))]
        pairs = [(rng.choice(codes), rng.choice(codes)) for _ in range(4096)]
        pairs += [(a, b) for a in (0x2F80, 0x2FFF, 0x4F00, 0x4F80, 0xAF80, 0xCFFF, 0xCF80)
                  for b in (0x2F80, 0x2FFF, 0x4F00, 0x4F80, 0xAF80, 0xCFFF, 0xCF80)]
        body = prepare_plan(prepare_source(source_request()).request).body_request()
        for offset in range(0, len(pairs), 1024):
            chunk = pairs[offset:offset + 1024]
            with self.subTest(offset=offset):
                actual = evaluate_body(body, chunk)
                for i, ((a, b), code) in enumerate(zip(chunk, actual, strict=True)):
                    self.assertEqual(code, integer_rne_add(a, b), (offset + i, hex(a), hex(b), hex(code)))

    def test_every_significand_at_and_around_midpoint(self):
        pairs = []
        for exponent in (-24, -1, 0, 24, 31):
            for fraction in range(128):
                code = ((exponent + 127) << 7) | fraction
                half = (exponent - 8 + 127) << 7
                for delta in (-1, 0, 1):
                    smaller = half + delta
                    if smaller < 0x2F80:
                        continue
                    for left_sign in (0, 0x8000):
                        for right_sign in (0, 0x8000):
                            pairs.append((code | left_sign, smaller | right_sign))
        body = prepare_plan(prepare_source(source_request()).request).body_request()
        for offset in range(0, len(pairs), 1024):
            chunk = pairs[offset:offset + 1024]
            actual = evaluate_body(body, chunk)
            for (a, b), code in zip(chunk, actual, strict=True):
                self.assertEqual(code, integer_rne_add(a, b), (hex(a), hex(b), hex(code)))

    def test_opposite_sign_residual_above_half_must_not_become_an_even_tie(self):
        # Real prefill residual pairs: the exact positive residual has nine
        # significand bits and chopping it incorrectly manufactured a tie.
        pairs = []
        for a, b in ((0xB8FF, 0x3D09), (0x387F, 0xBCC9)):
            pairs.extend(((a, b), (b, a), (a ^ 0x8000, b ^ 0x8000), (b ^ 0x8000, a ^ 0x8000)))
        body = prepare_plan(prepare_source(source_request()).request).body_request()
        actual = evaluate_body(body, pairs)
        for (a, b), observed in zip(pairs, actual, strict=True):
            self.assertEqual(observed, integer_rne_add(a, b), (hex(a), hex(b), hex(observed)))

    @unittest.skipUnless(os.environ.get("ATLAS_RNE_ADD_COMPILE") == "1", "select targeted native RNE add compile")
    def test_native_compilation_and_source_c_package(self):
        from atlas_mlir_native.binding import AtlasNativeBinding
        from atlas_mlir_native.pointwise import plan_abi, prepare_source as pointwise_source
        from atlas_mlir_native.pointwise_c import prepare_c_package
        from atlas_mlir_native.qualified_rne_add import RNE_ADD_POLICY
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import ROOT, build_snapshot

        def check(root):
            root.mkdir(parents=True, exist_ok=True)
            snapshot = build_snapshot(root)
            source = source_request()
            precision = pointwise_source(source, rne_add_policy=RNE_ADD_POLICY)
            abi = plan_abi(precision.request)
            compiled = root / "compiled"
            manifest = AtlasNativeBinding.compile(
                snapshot, precision.request, fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]), target_source=ROOT,
                destination=compiled,
                limits=SearchLimits(candidate_nodes=256, candidates=64, orders_per_candidate=4,
                                    wall_timeout_s=180),
            )
            self.assertEqual(manifest["engine"], "merlin_native")
            pure = [node for node in prepare_plan(precision.request).body_request().nodes if node.effect == "pure"]
            self.assertEqual(len(manifest["selected_body_descriptor"]), len(pure))
            record, files = prepare_c_package(source, compiled, "pointwise", target_source=ROOT,
                                               runtime_linkage="embedded", rne_add_policy=RNE_ADD_POLICY)
            self.assertEqual(record["input_node_ids"], ["a", "b"])
            self.assertEqual(record["precision_plan"]["output_conversion"], "bf16-bit-preserving")
            package = root / "package"
            package.mkdir()
            for name, data in files.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (root / "source-c-package.json").write_text(json.dumps(record, indent=2) + "\n")

        selected = os.environ.get("ATLAS_RNE_ADD_ARTIFACT_ROOT")
        if selected:
            check(Path(selected))
        else:
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as directory:
                check(Path(directory))

    @unittest.skipUnless(os.environ.get("ATLAS_RNE_ADD_MODEL") == "1", "select original47 generated-C execution")
    def test_original47_model_through_generated_bf16_source_c(self):
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        from atlas_mlir_native.pointwise import prepare_plan as pointwise_plan
        from merlin.semantic_compiler.pointwise_tiling import materialize_pointwise_input

        model_root = os.environ.get("ATLAS_NPU_MODEL_ROOT", os.environ.get("NPU_MODEL_ROOT"))
        artifact_root = os.environ.get("ATLAS_RNE_ADD_ARTIFACT_ROOT")
        if not model_root or not artifact_root:
            self.skipTest("set ATLAS_NPU_MODEL_ROOT or NPU_MODEL_ROOT and ATLAS_RNE_ADD_ARTIFACT_ROOT")
        root = Path(artifact_root)
        request = KernelRequest.from_record(json.loads((root / "compiled/source-request.json").read_text()))
        plan = pointwise_plan(request)
        execution = json.loads((root / "compiled/execution_plan.json").read_text())
        rng = random.Random(62184)
        codes = [0, 0x8000, *(sign | code for sign in (0, 0x8000) for code in range(0x2F80, 0x4F81))]
        pairs = [(a, b) for a in (0, 0x8000) for b in (0, 0x8000)]
        pairs += [(0x3F80, 0x3B80), (0x3F81, 0x3B80), (0xBF81, 0xBB80), (0x3F80, 0xBF80)]
        pairs += [(rng.choice(codes), rng.choice(codes)) for _ in range(1024 - len(pairs))]
        initial = bytearray([0xA5]) * 262144
        for i, address in enumerate(execution["input_addresses"]):
            raw = b"".join(pair[i].to_bytes(2, "little") for pair in pairs)
            (root / f"model-input-{i}.bin").write_bytes(raw)
            payload = pack_bf16_pair_halves(materialize_pointwise_input(plan, i, raw))
            at = address - 0x90000000
            initial[at:at + len(payload)] = payload
        wanted = b"".join(integer_rne_add(a, b).to_bytes(2, "little") for a, b in pairs)
        (root / "model-wanted-source.bin").write_bytes(wanted)
        (root / "model-initial.bin").write_bytes(initial)
        at = execution["output_addresses"][0] - 0x90000000
        initial[at:at + 2048] = pack_bf16_pair_halves(wanted)
        (root / "model-wanted-arena.bin").write_bytes(initial)
        environment = os.environ.copy()
        compiler_root = Path(__file__).resolve().parents[1]
        environment["PYTHONPATH"] = os.pathsep.join((model_root, str(compiler_root), str(compiler_root / "tools")))
        result = subprocess.run(
            [os.environ.get("MODEL_PYTHON", sys.executable),
             "-", str(root), model_root, str(compiler_root)],
            input=_ORIGINAL47_CHECK, text=True, capture_output=True, env=environment, timeout=180,
        )
        (root / "original47.stdout.log").write_text(result.stdout)
        (root / "original47.stderr.log").write_text(result.stderr)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads((root / "original47-receipt.json").read_text())
        self.assertEqual(receipt["runtime_errors"], 0)
        self.assertEqual(receipt["source_bytes_compared"], 2048)


_ORIGINAL47_CHECK = r'''
import ctypes, dataclasses, hashlib, json, subprocess, sys
from pathlib import Path
import torch
import npu_model
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.logging import LoggerConfig
from npu_model.simulation import Simulation
from npu_model.software.program import InstantiableProgram
from tools.check_program_decode import decode_binary
from tools.model_instructions import model_instruction
root, model_root, compiler_root = map(Path, sys.argv[1:])
revision = subprocess.check_output(['git','-C',str(model_root),'rev-parse','HEAD'], text=True).strip()
assert revision.startswith('47b8599'), revision
assert Path(npu_model.__file__).resolve().is_relative_to(model_root.resolve())
torch.set_num_threads(1)
binary = (root/'compiled/program.bin').read_bytes()
inventory = json.loads((compiler_root/'docs/selected-variant-inventory.json').read_text())
decoded = decode_binary(inventory,binary)
program = InstantiableProgram([model_instruction(row,inventory,isa) for row in decoded['instructions']])
program.dram_base = 0x90000000
initial = (root/'model-initial.bin').read_bytes()
wanted_arena = (root/'model-wanted-arena.bin').read_bytes()
wanted = (root/'model-wanted-source.bin').read_bytes()
package = root/'package'
(package/'client.c').write_text("""#include "pointwise.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const uint8_t *const *inputs,uint8_t *output) {
  atlas_host host; pointwise_context raw; pointwise_source_context source;
  unsigned char scratch[POINTWISE_WORKSPACE_BYTES], conversion[POINTWISE_SOURCE_WORKSPACE_BYTES];
  if (atlas_host_init(&host,(atlas_host_map){0x20000,0x40000,131072},
      (atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
      pointwise_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8) ||
      pointwise_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
  return pointwise_source_run(&source,inputs,output,0);
}
""")
subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                'pointwise.c','client.c','-o','client.so'], cwd=package, check=True)
client = ctypes.CDLL(str(package/'client.so'))
rd_type = ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_void_p,ctypes.c_size_t)
wr_type = ctypes.CFUNCTYPE(None,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_uint32)
mem_type = ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t)
memory, words, errors, receipt = bytearray([0xA5])*len(initial), {}, [], {}
polls = None
def read32(_,address):
    global polls
    if address==0x40004: return len(words)
    assert address==0x40008, address
    if polls is None: return 5
    polls += 1
    return 0 if polls==1 else 3 if errors else 5
def write32(_,address,word):
    global polls
    try:
        if 0x20000<=address<0x40000:
            words[(address-0x20000)//4] = word
            return
        assert (address,word)==(0x40018,1), (address,word)
        assert b''.join(words[i].to_bytes(4,'little') for i in range(len(words)))==binary
        assert bytes(memory)==initial, 'generated source packing differs'
        program.memory_regions=[(0,torch.tensor(list(memory),dtype=torch.uint8))]
        config=DefaultHardwareConfig()
        config.arch_state_config=dataclasses.replace(config.arch_state_config,numerics='rtl',
                                                     randomize_init=True,init_seed=81)
        sim=Simulation(config,LoggerConfig(filename=str(root/'original47-model.json')),program,verbose=False)
        state=sim.core.arch_state
        try:
            state.vmem.fill_(0xCD)
            sim.run(max_cycles=500000)
            assert state.halted
            assert sim.get_stats().runtime_errors==0
            observed=bytes(state.read_dram(0x90000000,len(memory)).tolist())
            assert observed==wanted_arena, 'generated RNE device output or arena guards differ'
            memory[:]=observed
            receipt.update(model_revision=revision,numerics='rtl',cycles=sim.get_stats().cycles,
                           runtime_errors=sim.get_stats().runtime_errors,arena_bytes_compared=len(memory),
                           program_sha256=hashlib.sha256(binary).hexdigest())
        finally:
            state.close()
        polls=0
    except Exception as error:
        errors.append(repr(error))
def transfer(address,pointer,size,write):
    at=address-0x90000000
    if not 0<=at<=len(memory)-size: return 1
    if write: memory[at:at+size]=ctypes.string_at(pointer,size)
    else: ctypes.memmove(pointer,bytes(memory[at:at+size]),size)
    return 0
callbacks=(rd_type(read32),wr_type(write32),
           mem_type(lambda _,a,p,n:transfer(a,p,n,True)),
           mem_type(lambda _,a,p,n:transfer(a,p,n,False)))
client.call.argtypes=[rd_type,wr_type,mem_type,mem_type,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
values=[ctypes.create_string_buffer((root/f'model-input-{i}.bin').read_bytes()) for i in range(2)]
inputs=(ctypes.c_void_p*2)(*(ctypes.addressof(value) for value in values))
output=ctypes.create_string_buffer(bytes([0xA7])*(len(wanted)+32))
result=client.call(*callbacks,inputs,ctypes.addressof(output)+16)
assert not errors, errors
assert result==0, result
assert output.raw[16:16+len(wanted)]==wanted
assert output.raw[:16]==bytes([0xA7])*16
assert output.raw[16+len(wanted):-1]==bytes([0xA7])*16
assert all(value.raw[:-1]==(root/f'model-input-{i}.bin').read_bytes() for i,value in enumerate(values))
receipt.update(source_bytes_compared=len(wanted),source_abi='bf16-le-raw',host_output_guards='pass',
               input_preservation='pass',source_output_sha256=hashlib.sha256(wanted).hexdigest())
(root/'original47-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


if __name__ == "__main__":
    unittest.main()
