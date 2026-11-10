from dataclasses import replace
import ctypes
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from hashlib import sha256
from atlas_mlir_native.exact_float_device import prepare_plan,machine_program,OPS,MAX_LEAF_ELEMENTS,compile_source,prepare_c_package
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.limits import DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT
from merlin.semantic_compiler.model import TensorType,pointwise_identity_maps
from merlin.tests.runtime.test_bf16_arithmetic_c import request as binary_request

def request(operation,count=35,shared=False):
    source=binary_request((count,),"bf16_add",shared)
    width,ow,policy=OPS[operation]
    binary=operation in ("bf16_add","bf16_sub")
    operands=source.nodes[:-1] if binary else source.nodes[:1]
    operands=tuple(replace(n,type=TensorType((count,),"bf16" if width==2 else "f32","raw-bit-pattern")) for n in operands)
    out=replace(source.nodes[-1],op=operation,inputs=source.nodes[-1].inputs if binary else (operands[0].id,),
                type=TensorType((count,),"bf16" if ow==2 else "f32","raw-bit-pattern"),
                attrs=(("numeric_semantics",policy),),index_maps=pointwise_identity_maps(1,2 if binary else 1))
    return replace(source,target_identity=TARGET_IDENTITY,lowering_policy="diagnostic",nodes=(*operands,out),input_storages=tuple((n.id,"external") for n in operands))

def host_reference(source,raw):
    from merlin.compile.c_float_casts import emit_float_cast_c
    from merlin.compile.c_float_arithmetic import emit_float_arithmetic_c
    emitter=emit_float_cast_c if source.nodes[-1].op in ("bf16_to_f32","f32_to_bf16") else emit_float_arithmetic_c
    with tempfile.TemporaryDirectory(prefix="atlas-float-reference-") as path:
        path=Path(path);_,files=emitter(source,"reference")
        for name,data in files.items():
            (path/name).parent.mkdir(parents=True,exist_ok=True);(path/name).write_bytes(data.encode() if isinstance(data,str) else data)
        subprocess.run([shutil.which("cc"),"-shared","-fPIC","-O2",str(path/"reference.c"),"-o",str(path/"reference.so")],check=True,capture_output=True)
        lib=ctypes.CDLL(str(path/"reference.so"));fn=lib.reference
        fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
        inputs=[ctypes.create_string_buffer(data) for data in raw]
        p=prepare_plan(source);out=ctypes.create_string_buffer(bytes([0xa5])*(p.count*p.output_width))
        status=fn(None,(ctypes.c_void_p*len(inputs))(*(ctypes.addressof(a) for a in inputs)),(ctypes.c_void_p*1)(ctypes.addressof(out)))
        return out.raw[:-1],status

def scalar_execute(source, raw, max_instructions=DEFAULT_FUNCTIONAL_INSTRUCTION_LIMIT):
    from float_isa_c import execute
    result=execute(source,raw)
    if result[2]>max_instructions:raise RuntimeError("default scalar instruction limit exhausted")
    return result


class ExactFloatPortableTest(unittest.TestCase):
    def test_directed_boundaries_ties_and_seeded_pairs(self):
        rng=random.Random(731)
        edges=[0,1,2,127,128,129,0x3b80,0x3b81,0x3f80,0x3f81,0x3f82,0x7f7f,0x7f80]
        edges += [v|32768 for v in edges]
        pairs=[(a,b) for a in edges for b in edges]
        pairs += [(rng.randrange(65536),rng.randrange(65536)) for _ in range(1024)]
        for operation in ("bf16_add","bf16_sub"):
            valid=[(a,b) for a,b in pairs if (a&32767)<=32640 and (b&32767)<=32640 and not ((a&32767)==(b&32767)==32640 and ((a^b^(32768 if operation=="bf16_sub" else 0))&32768))]
            for start in range(0,len(valid),128):
                values=valid[start:start+128];source=request(operation,len(values));raw=[b"".join(pair[i].to_bytes(2,"little") for pair in values) for i in (0,1)]
                observed,status,_=scalar_execute(source,raw);expected,host_status=host_reference(source,raw)
                self.assertEqual(status,0);self.assertEqual(host_status,0)
                self.assertEqual(observed,expected,(operation,values,observed.hex(),expected.hex()))
    def test_casts_negations_and_invalid_late_lane_preserve_output(self):
        bf16=[0,0x8000,1,127,128,0x3f80,0x7f7f,0x7f80,0xff80]
        f32=[0,0x80000000,1,0x007fffff,0x00800000,0x3f887fff,0x3f888000,0x3f888001,0x3f898000,0x3fff8000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000]
        for operation in ("bf16_to_f32","f32_to_bf16","bf16_neg","f32_neg"):
            width=OPS[operation][0];values=(bf16 if width==2 else f32)*3
            if operation.endswith("_neg"):values += [0x7fc1,0xffff] if width==2 else [0x7f800001,0xffffffff]
            raw=[b"".join(v.to_bytes(width,"little") for v in values)];source=request(operation,len(values))
            self.assertEqual(scalar_execute(source,raw)[:2],host_reference(source,raw))
        for operation in ("bf16_to_f32","f32_to_bf16","bf16_add","bf16_sub"):
            p=prepare_plan(request(operation,35));raw=[bytes(34*p.width)+(0x7fc1 if p.width==2 else 0x7f800001).to_bytes(p.width,"little")]+([bytes(70)] if len(p.operands)>1 else [])
            observed,status,_=scalar_execute(p.source,raw)
            self.assertNotEqual(status,0);self.assertEqual(observed,bytes([0xa5])*len(observed))
    def test_pair_row_full_normal_input_and_checked_result_domain(self):
        accepted=[(0x0100,0x0100),(0x0100,0x0080),
                  (0x7f00,0xfe80),(0x7f00,0x7e00)]*8
        rejected=[(0x0081,0x0080),(0x0101,0x0100),
                  (0x7f7f,0xff7f),(0x7f7f,0x7f7f)]*8
        for operation in ("bf16_add","bf16_sub"):
            source=request(operation,32)
            for pairs in (accepted,rejected):
                raw=[b"".join(pair[k].to_bytes(2,"little") for pair in pairs) for k in (0,1)]
                self.assertEqual(scalar_execute(source,raw)[:2],host_reference(source,raw))

    def test_shared_inputs(self):
        for operation in ("bf16_add","bf16_sub"):
            source=request(operation,35,shared=True);raw=[b"\x80\x3f"*35]
            self.assertEqual(scalar_execute(source,raw)[:2],host_reference(source,raw))


ROOT=Path(__file__).resolve().parents[1]
BIN=Path(os.environ.get('ATLAS_OOT_BIN_DIR',ROOT/'build/bin'))

@unittest.skipUnless((BIN/'atlas-opt').is_file(),'select built Atlas tools')
class ExactFloatTypedTest(unittest.TestCase):
    def lower(self,source):
        from atlas_mlir_native.exact_float_device import virtual_program
        r=subprocess.run([str(BIN/'atlas-opt'),'--atlas-compile-schedule','--atlas-compile-unscheduled','-'],input=virtual_program(prepare_plan(source)),text=True,capture_output=True,timeout=30)
        self.assertEqual(r.returncode,0,r.stderr);return r.stdout
    def binary(self,machine):
        r=subprocess.run([str(BIN/'atlas-emit'),'--program-json','-'],input=machine,text=True,capture_output=True,check=True,timeout=30)
        return b''.join(row['word_u32'].to_bytes(4,'little') for row in json.loads(r.stdout)['instructions'])
    def test_all_operations_use_complete_standard_contracts(self):
        for op in OPS:
            for count in (1,35,MAX_LEAF_ELEMENTS[op]):
                source=request(op,count);machine=self.lower(source)
                r=subprocess.run([str(BIN/'atlas-opt'),'--verify-atlas-generated-schedule','--verify-atlas-machine-stream','--verify-atlas-timing','-'],input=machine,text=True,capture_output=True,timeout=30)
                self.assertEqual(r.returncode,0,r.stderr)
                for contract in ('virtual_dma_contract','virtual_tile_contract','virtual_cfg_contract','virtual_source_memory_contract','virtual_buffer_contract'):
                    self.assertIn(contract,machine)
                self.assertNotIn('exact_float_contract',machine)

    def test_rehashed_bound_rejected_by_source_replay(self):
        source=request('bf16_to_f32',35)
        with tempfile.TemporaryDirectory(prefix='atlas-float-bound-') as directory:
            compiled=Path(directory)/'compiled'
            manifest=compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
            proof=compiled/'float-instruction-bound.json'
            record=json.loads(proof.read_text());record['instruction_bound']-=1
            proof.write_text(json.dumps(record))
            manifest['stage_artifacts'][proof.name]=sha256(proof.read_bytes()).hexdigest()
            (compiled/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'instruction bound differs'):
                prepare_c_package(source,compiled,'changed_bound',target_source=ROOT)

    def test_oversized_physical_bound_refused_without_changing_limit(self):
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        source=request('bf16_to_f32',32768)
        with tempfile.TemporaryDirectory(prefix='atlas-float-limit-') as directory:
            # Bypass only the chunk policy to exercise the real bound gate.
            with patch.dict(MAX_LEAF_ELEMENTS,{'bf16_to_f32':32768}):
                with self.assertRaisesRegex(NativeCompilationError,'exceeds 1000000'):
                    compile_source(None,source,target_source=ROOT,destination=Path(directory)/'compiled',limits=None)
    def test_instruction_domain_status_and_module_mutations_rejected(self):
        machine=self.lower(request('bf16_add',35))
        mutations=[machine.replace('direction = "store"','direction = "load"',1),
                   machine.replace('kind = "sh"','kind = "sw"',1),
                   machine.replace('kind = "srl"','kind = "sll"',1),
                   re.sub(r'offset_bytes = (-?\d+) : i32',lambda m:f'offset_bytes = {int(m[1])+2} : i32',machine,count=1),
                   re.sub(r'atlas.virtual_scalar_result = (\d+) : i32',lambda m:f'atlas.virtual_scalar_result = {int(m[1])+1} : i32',machine,count=1)]
        for changed in mutations:
            self.assertNotEqual(changed,machine)
            r=subprocess.run([str(BIN/'atlas-opt'),'--verify-atlas-generated-schedule','-'],input=changed,text=True,capture_output=True,timeout=30)
            self.assertGreaterEqual(r.returncode,1,r.stderr)

if __name__=="__main__":unittest.main()
