"""Full source float slicing, C publication, and unchanged-model integration."""
from dataclasses import replace
from contextlib import nullcontext
from hashlib import sha256
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from atlas_mlir_native import exact_float_application as app
from test_exact_float_device import request,host_reference,scalar_execute,OPS
ROOT=Path(__file__).resolve().parents[1]
BIN=Path(os.environ.get('ATLAS_OOT_BIN_DIR',ROOT/'build/bin'))

class ExactFloatScheduleTest(unittest.TestCase):
    def test_full_and_tail_source_mapping(self):
        for op in OPS:
            p=app.prepare_plan(request(op,2051))
            self.assertEqual(len(p.bodies),2)
            binary=op in ('bf16_add','bf16_sub');chunk=1024 if binary else 2048
            offsets=[0,1024,2048] if binary else [0,2048]
            self.assertEqual(len(p.launches),len(offsets))
            self.assertEqual([r['input_offset_bytes'] for r in p.launches],[i*p.plan.width for i in offsets])
            self.assertEqual([r['output_bytes'] for r in p.launches],[min(chunk,2051-i)*p.plan.output_width for i in offsets])

@unittest.skipUnless((BIN/'atlas-opt').is_file(),'select built Atlas tools')
class ExactFloatCompilationTest(unittest.TestCase):
    def test_every_op_two_chunks_c_export_and_rehashed_slice_rejection(self):
        for op in OPS:
            source=request(op,2051)
            with tempfile.TemporaryDirectory(prefix='atlas-float-export-') as directory:
                directory=Path(directory);compiled=directory/'compiled'
                manifest=app.compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
                exported=app.prepare_c_package(source,compiled,'float_test',target_source=ROOT)
                self.assertEqual(exported.description['device_launch_schedule']['total_launches'],3 if op in ('bf16_add','bf16_sub') else 2)
                package=directory/'package'
                for name,data in exported.files.items():
                    (package/name).parent.mkdir(parents=True,exist_ok=True);(package/name).write_bytes(data)
                subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-fsyntax-only','float_test.c','runtime/atlas_host.c'],cwd=package,check=True,capture_output=True)
                # Rehashing attacker-controlled schedule metadata cannot change slices.
                schedule=compiled/'float-schedule.json';record=json.loads(schedule.read_text());record['launches'][0]['output_offset_bytes']=2
                schedule.write_text(json.dumps(record));manifest['stage_artifacts']['float-schedule.json']=sha256(schedule.read_bytes()).hexdigest()
                (compiled/'float-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError,'slice schedule'):
                    app.prepare_c_package(source,compiled,'changed',target_source=ROOT)

@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model')
class ExactFloatModelTest(unittest.TestCase):
    def test_all_operations_actual_c_dispatch_and_late_invalid_second_chunk(self):
        import torch
        model_root=Path(os.environ['ATLAS_NPU_MODEL_ROOT'])
        sys.path[:0]=[str(model_root),str(ROOT),str(ROOT/'tools'),str(ROOT.parent/'merlin/examples/atlas/whole-model/unchanged_fx')]
        from functional_runtime import FunctionalModelRuntime
        from model_runtime import READ,WRITE,MEMORY
        torch.set_num_threads(1)
        retained=os.environ.get('ATLAS_FLOAT_MODEL_TEST_OUT')
        if retained:Path(retained).mkdir(parents=True,exist_ok=False)
        workspace=nullcontext(retained) if retained else tempfile.TemporaryDirectory(prefix='atlas-float-model-')
        with workspace as directory:
            directory=Path(directory)
            for op in OPS:
                # Two genuine chunks prove atomic publication on late failure.
                count=2051 if op in ('bf16_to_f32','f32_to_bf16','bf16_add','bf16_sub') else 35
                source=request(op,count);p=app.prepare_plan(source);compiled=directory/op
                manifest=app.compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
                exported=app.prepare_c_package(source,compiled,'float_test',target_source=ROOT)
                package=directory/(op+'-c');package.mkdir()
                for name,data in exported.files.items():
                    (package/name).parent.mkdir(parents=True,exist_ok=True);(package/name).write_bytes(data)
                harness='''#include "float_test.h"
static float_test_context region;static float_test_source_context source;
static uint8_t workspace[FLOAT_TEST_WORKSPACE_BYTES];static atlas_host host;
int run(atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get,const void *const *inputs,void *const *outputs) {
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 if(float_test_bind(&region,&host,(atlas_tile_memory_io){put,get,0},workspace,sizeof(workspace),8u))return 8;
 if(float_test_source_bind(&source,&region,0,0))return 7;
 return float_test_source_dispatch(&source,inputs,outputs);
}
'''
                (package/'harness.c').write_text(harness)
                subprocess.run(['cc','-std=c11','-Wall','-Wextra','-Werror','-shared','-fPIC','-O2','float_test.c','runtime/atlas_host.c','harness.c','-o','test.so'],cwd=package,check=True,capture_output=True)
                library=ctypes.CDLL(str(package/'test.so'));function=library.run;function.argtypes=[READ,WRITE,MEMORY,MEMORY,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
                images=[];plans={}
                for segment in manifest['segments']:
                    leaf=compiled/segment['path'];binary=(leaf/'program.bin').read_bytes();images.append(binary)
                    plans[sha256(binary).hexdigest()]=json.loads((leaf/'execution_plan.json').read_text())
                runtime=FunctionalModelRuntime(ROOT,model_root,images,directory/(op+'-run'),memory_bytes=max(pl['workspace_bytes'] for pl in plans.values())+64,image_plans=plans,trace_payloads=False)
                # Directed ties, subnormals, signed zero, infinities, and NaN payload sign flips.
                width=p.plan.width
                codes=[0,1,127,128,0x8000,0x3f80,0x3bc0,0x7f7f,0x7f80,0xff80]
                if op.endswith('_neg'):codes += [0x7fc1,0xffff]
                left=[codes[i%len(codes)] for i in range(count)]
                raw0=b''.join((x if width==2 else x<<16).to_bytes(width,'little') for x in left)
                arrays=[raw0]+([bytes(count*2)] if len(p.plan.operands)>1 else [])
                if op=='f32_to_bf16':
                    directed=(0,0x80000000,1,0x007fffff,0x00800000,0x3f887fff,
                              0x3f888000,0x3f888001,0x3f898000,0x7f7fffff,0x7f800000)
                    arrays=[b''.join(directed[i%len(directed)].to_bytes(4,'little') for i in range(count))]
                if op in ('bf16_add','bf16_sub'):
                    # Keep complete tiles inside each vector domain, and
                    # separate exceptional tiles, so all three routes execute.
                    domains=(
                        ((0x3f80,0x3bc0),(0x3f80,0x3b80),(0x3f81,0x3b80),
                         (0x3f80,0xbf80),(0x8000,0x8000),(0,0x8000)),
                        ((0x2000,0x1c00),(0x2001,0x1c00),(0x6000,0x5c00),
                         (0x6001,0x5c00),(0x6001,0),(0xa000,0x9c00),
                         (0x0100,0x0100),(0x0100,0x0080),
                         (0x7f00,0xfe80),(0x7f00,0x7e00)),
                        ((1,127),(128,0x8001),(0x7f7f,0x7f7f),
                         (0x7f80,0),(0xff80,0),(0x8000,0x8000)),
                    )
                    pairs=[domains[(i//32)%3][i%len(domains[(i//32)%3])] for i in range(count)]
                    arrays=[b''.join(pair[k].to_bytes(2,'little') for pair in pairs) for k in (0,1)]
                inputs=[ctypes.create_string_buffer(raw) for raw in arrays];output=ctypes.create_string_buffer(bytes([0xa5])*(count*p.plan.output_width))
                pointers=(ctypes.c_void_p*len(inputs))(*(ctypes.addressof(v) for v in inputs));outputs=(ctypes.c_void_p*1)(ctypes.addressof(output))
                self.assertEqual(function(*runtime.callbacks,pointers,outputs),0,(op,runtime.errors))
                self.assertEqual(output.raw[:-1],host_reference(source,arrays)[0],op)
                if not op.endswith('_neg'):
                    nan=(0x7fc1 if width==2 else 0x7f800001).to_bytes(width,'little')
                    ctypes.memmove(ctypes.addressof(inputs[0])+(count-1)*width,nan,width)
                    output.raw=bytes([0xa5])*len(output.raw)
                    self.assertEqual(function(*runtime.callbacks,pointers,outputs),2,(op,runtime.errors))
                    self.assertEqual(output.raw,bytes([0xa5])*len(output.raw),op)
                record=runtime.finish();self.assertFalse(record['errors'])
