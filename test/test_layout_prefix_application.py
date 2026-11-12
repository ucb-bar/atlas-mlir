"""Complete matrix-prefix slicing authenticates bytes and publishes atomically."""
from dataclasses import replace
from hashlib import sha256
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from atlas_mlir_native import layout_device as device
from atlas_mlir_native import layout_prefix_application as prefix
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.model import ConstantBinding, IndexMap
from test_layout_device import request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('ATLAS_OOT_BIN_DIR', ROOT / 'build/bin'))


def transpose(heads=32, rows=512, columns=64):
    return request('transpose', ((1, heads, rows, columns),), (1, heads, columns, rows), dtype='i8',
        mapping=IndexMap(4, ((1,0,0,0),(0,1,0,0),(0,0,0,1),(0,0,1,0)), (0,0,0,0)))


def concat(heads=32, rows=512, columns=32):
    return request('concat', ((1,heads,rows,columns),)*2, (1,heads,rows,columns*2), axis=3)


def arrays(source):
    rng = np.random.default_rng(512997)
    p = device.prepare_source(source)
    return [rng.integers(0, 256, size=size, dtype=np.uint8).tobytes() for size in p.input_bytes]


def expected(source, raw):
    p = device.prepare_source(source)
    operands = [np.frombuffer(value, dtype=np.dtype(f'V{p.width}')).reshape(node.type.shape)
                for node, value in zip(p.operands, raw, strict=True)]
    if p.output.op == 'transpose':
        return operands[0].swapaxes(-2,-1).copy().tobytes()
    return np.concatenate(operands, axis=dict(p.output.attrs)['axis']).tobytes()


class PrefixCase(unittest.TestCase):
    def setUp(self):
        context = codegen_options(CodegenOptions(io_memory_bytes=59768832))
        context.__enter__()
        self.addCleanup(context.__exit__, None, None, None)


class PrefixScheduleTest(PrefixCase):
    def test_real_geometries_reuse_one_complete_matrix_body(self):
        for source in (transpose(), concat()):
            plan = device.prepare_source(source)
            schedule = prefix.prepare_plan(plan)
            self.assertEqual(len(schedule.bodies), 1)
            self.assertEqual(len(schedule.launches), 4)
            body = schedule.bodies[0]
            self.assertEqual(body.nodes[-1].type.shape[:-2], (1,8))
            self.assertEqual(body.nodes[-1].type.shape[-2:], source.nodes[-1].type.shape[-2:])
            self.assertEqual(body.nodes[-1].attrs, source.nodes[-1].attrs)
            self.assertEqual(body.nodes[-1].index_maps, source.nodes[-1].index_maps)
            raw = arrays(source)
            result = bytearray(plan.output_bytes)
            coverage = []
            for launch in schedule.launches:
                parts = [raw[plan.output.inputs.index(span['node_id'])][span['offset_bytes']:span['offset_bytes']+span['bytes']]
                         for span in launch['inputs']]
                data = expected(body, parts)
                start = launch['output_offset_bytes']
                self.assertEqual(len(data), launch['output_bytes'])
                result[start:start+len(data)] = data
                coverage.append((start,start+len(data)))
            self.assertEqual(coverage, [(i*plan.output_bytes//4,(i+1)*plan.output_bytes//4) for i in range(4)])
            self.assertEqual(bytes(result), expected(source, raw))

    def test_flattened_prefix_remainder_and_immutable_byte_spans(self):
        source = concat(heads=5,rows=32,columns=16)
        # A nontrivial original multidimensional prefix remains dense.
        source = replace(source, nodes=tuple(replace(node,type=replace(node.type,
            shape=(2,*node.type.shape[1:]))) for node in source.nodes))
        raw = arrays(source)
        operand = replace(source.nodes[0],op='constant',effect='constant')
        source = replace(source,nodes=(operand,*source.nodes[1:]),input_storages=((source.nodes[1].id,'external'),),
            constants=(ConstantBinding(operand.id,'bf16-le-raw',raw[0].hex()),))
        with patch.object(prefix,'MAX_CONCAT_BYTES',6144):
            schedule = prefix.prepare_plan(device.prepare_source(source))
        self.assertEqual([call['matrix_count'] for call in schedule.launches], [3,3,3,1])
        result = bytearray(device.prepare_source(source).output_bytes)
        for call in schedule.launches:
            body = schedule.bodies[call['body_index']]
            constant = bytes.fromhex(body.constants[0].data_hex)
            span = call['constants'][0]
            self.assertEqual(constant,raw[0][span['offset_bytes']:span['offset_bytes']+span['bytes']])
            self.assertEqual(span['source_sha256'],sha256(raw[0]).hexdigest())
            self.assertEqual(span['child_sha256'],sha256(constant).hexdigest())
            variable = call['inputs'][0]
            data = expected(body,[constant,raw[1][variable['offset_bytes']:variable['offset_bytes']+variable['bytes']]])
            start=call['output_offset_bytes'];result[start:start+len(data)]=data
        self.assertEqual(bytes(result),expected(source,raw))


@unittest.skipUnless((BIN/'atlas-opt').is_file(),'select built Atlas tools')
class PrefixCompilationTest(PrefixCase):
    def test_only_mandatory_layout_uses_full_finalizer_coverage(self):
        from atlas_mlir_native.binding import NATIVE_CHECKS
        from merlin.semantic_compiler.target_binding import NativeCompilationError
        source=request(shapes=((3,32),),shape=(96,))
        original_run=device._run
        def guarded_run(command,**kwargs):
            if all(check in command for check in NATIVE_CHECKS):
                raise NativeCompilationError('compile_error','independent optimized postcheck control')
            return original_run(command,**kwargs)
        for pipeline in ('mandatory','optimized'):
            with self.subTest(pipeline=pipeline), tempfile.TemporaryDirectory(prefix='atlas-layout-finalizer-') as directory, \
                    patch.dict(os.environ,{'ATLAS_NATIVE_PIPELINE':pipeline}), \
                    patch.object(device,'_run',side_effect=guarded_run):
                destination=Path(directory)/'compiled'
                if pipeline == 'optimized':
                    with self.assertRaisesRegex(NativeCompilationError,'independent optimized postcheck control'):
                        device.compile_source(None,source,target_source=ROOT,destination=destination,limits=None)
                    self.assertFalse(destination.exists())
                else:
                    device.compile_source(None,source,target_source=ROOT,destination=destination,limits=None)
                    bound=json.loads((destination/'layout-instruction-bound.json').read_text())
                    self.assertTrue(bound['admitted'])
                    self.assertEqual(bound['max_instructions'],1000000)

    def test_export_and_rehashed_slice_mutations_refuse(self):
        source=concat()
        with tempfile.TemporaryDirectory(prefix='atlas-prefix-export-') as directory:
            root=Path(directory);compiled=root/'compiled'
            manifest=device.compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
            record,files=device.prepare_c_package(source,compiled,'prefix_test',target_source=ROOT)
            self.assertEqual(record['launches_per_invocation'],4)
            self.assertEqual(record['device_launch_schedule']['total_launches'],4)
            self.assertEqual(record['native_program_paths'],['segments/body_000'])
            bound=json.loads((compiled/'segments/body_000/layout-instruction-bound.json').read_text())
            self.assertTrue(bound['admitted']);self.assertEqual(bound['max_instructions'],1000000)
            for filename,data in files.items():
                path=root/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-fsyntax-only','prefix_test.c','runtime/atlas_host.c'],cwd=root,check=True,capture_output=True)
            path=compiled/'layout-prefix-schedule.json';original=path.read_bytes()
            for field,value in (('output_offset_bytes',0),('matrix_count',9)):
                changed=json.loads(original);changed['launches'][1][field]=value
                path.write_text(json.dumps(changed));manifest['stage_artifacts'][path.name]=sha256(path.read_bytes()).hexdigest()
                (compiled/'layout-prefix-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'slice schedule'):
                    device.prepare_c_package(source,compiled,'changed',target_source=ROOT)
            path.write_bytes(original);manifest['stage_artifacts'][path.name]=sha256(original).hexdigest()
            (compiled/'layout-prefix-manifest.json').write_text(json.dumps(manifest))
            changed=json.loads(original);changed['launches'][1]['inputs'][0]['offset_bytes']=0
            path.write_text(json.dumps(changed));manifest['stage_artifacts'][path.name]=sha256(path.read_bytes()).hexdigest()
            (compiled/'layout-prefix-manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'slice schedule'):
                device.prepare_c_package(source,compiled,'changed',target_source=ROOT)

    def test_indivisible_matrix_above_default_instruction_bound_refuses(self):
        source=transpose(heads=1,rows=1024,columns=1024)
        self.assertIsNone(prefix.prepare_plan(device.prepare_source(source)))
        with tempfile.TemporaryDirectory(prefix='atlas-prefix-refusal-') as directory:
            with self.assertRaisesRegex(Exception,'instruction bound exceeds 1000000'):
                device.compile_source(None,source,target_source=ROOT,destination=Path(directory)/'compiled',limits=None)


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select original npu_model')
class PrefixModelTest(PrefixCase):
    def run_case(self,source,raw,*,cap_patch=None):
        import torch
        model=Path(os.environ['ATLAS_NPU_MODEL_ROOT'])
        sys.path[:0]=[str(model),str(ROOT),str(ROOT/'tools'),str(ROOT.parent/'merlin/examples/atlas/whole-model/unchanged_fx')]
        from functional_runtime import FunctionalModelRuntime
        from model_runtime import READ,WRITE,MEMORY
        torch.set_num_threads(1)
        context=cap_patch or patch.object(prefix,'MAX_XLU_TILES',prefix.MAX_XLU_TILES)
        with context,tempfile.TemporaryDirectory(prefix='atlas-prefix-model-') as directory:
            root=Path(directory);compiled=root/'compiled'
            manifest=device.compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
            record,files=device.prepare_c_package(source,compiled,'prefix_test',target_source=ROOT)
            for filename,data in files.items():
                path=root/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            (root/'harness.c').write_text('''#include "prefix_test.h"
static prefix_test_context region; static prefix_test_source_context source;
static uint8_t workspace[PREFIX_TEST_WORKSPACE_BYTES]; static atlas_host host;
int run(int mode,atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get,const void *const *inputs,void *const *outputs){
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 if(prefix_test_bind(&region,&host,(atlas_tile_memory_io){put,get,0},workspace,mode==1?sizeof(workspace)-1:sizeof(workspace),8u))return 8;
 if(prefix_test_source_bind(&source,&region,0,0))return 7;
 return prefix_test_source_dispatch(&source,inputs,outputs);}
''')
            subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC','-O2','prefix_test.c','runtime/atlas_host.c','harness.c','-o','test.so'],cwd=root,check=True,capture_output=True)
            images=[];plans={}
            for row in manifest['segments']:
                leaf=compiled/row['path'];binary=(leaf/'program.bin').read_bytes();images.append(binary)
                plans[sha256(binary).hexdigest()]=json.loads((leaf/'execution_plan.json').read_text())
            runtime=FunctionalModelRuntime(ROOT,model,images,root/'run',memory_bytes=max(plan['workspace_bytes'] for plan in plans.values())+64,image_plans=plans,trace_payloads=False)
            function=ctypes.CDLL(str(root/'test.so')).run
            function.argtypes=[ctypes.c_int,READ,WRITE,MEMORY,MEMORY,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
            keys=[key for key,_ in source.input_storages];operands=device.prepare_source(source).operands
            buffers=[ctypes.create_string_buffer(raw[[node.id for node in operands].index(key)]) for key in keys]
            inputs=(ctypes.c_void_p*len(buffers))(*(ctypes.addressof(value) for value in buffers)) if buffers else None
            output=ctypes.create_string_buffer(bytes([0xa5])*device.prepare_source(source).output_bytes)
            outputs=(ctypes.c_void_p*1)(ctypes.addressof(output))
            self.assertEqual(function(0,*runtime.callbacks,inputs,outputs),0,runtime.errors)
            self.assertEqual(output.raw[:-1],expected(source,raw))
            before=len(runtime.launches)
            output.raw=bytes([0xa5])*len(output.raw)
            self.assertNotEqual(function(1,*runtime.callbacks,inputs,outputs),0)
            self.assertEqual(len(runtime.launches),before)
            self.assertEqual(output.raw,bytes([0xa5])*len(output.raw))
            self.assertNotEqual(function(0,*runtime.callbacks,inputs,(ctypes.c_void_p*1)(0)),0)
            self.assertEqual(len(runtime.launches),before)
            reads=0
            def late_failure(context,address,data,size):
                nonlocal reads
                reads+=1
                return -1 if reads==2 else runtime.callbacks[3](context,address,data,size)
            self.assertNotEqual(function(0,*runtime.callbacks[:3],MEMORY(late_failure),inputs,outputs),0)
            self.assertEqual(reads,2)
            self.assertEqual(output.raw,bytes([0xa5])*len(output.raw))
            result=runtime.finish();self.assertFalse(result['errors'])
            self.assertTrue(all(row['instructions']<=1000000 for row in result['launches']))
            print('PREFIX_MODEL',source.nodes[-1].op,record['launches_per_invocation'],[row['instructions'] for row in result['launches']],flush=True)

    def test_real_transpose_and_concat_byte_dispatch_and_late_failure(self):
        reversed_ports=concat()
        reversed_ports=replace(reversed_ports,input_storages=tuple(reversed(reversed_ports.input_storages)))
        for source in (transpose(),reversed_ports):
            with self.subTest(operation=source.nodes[-1].op):self.run_case(source,arrays(source))

    def test_zero_runtime_input_constant_dispatch_and_guard_failures(self):
        source=transpose(heads=4,rows=32,columns=32);raw=arrays(source)
        operand=replace(source.nodes[0],op='constant',effect='constant')
        source=replace(source,nodes=(operand,source.nodes[-1]),input_storages=(),
            constants=(ConstantBinding(operand.id,'i8',raw[0].hex()),))
        self.run_case(source,raw,cap_patch=patch.object(prefix,'MAX_XLU_TILES',1))


if __name__=='__main__':unittest.main()
