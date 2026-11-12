"""Counted row-block C transport, atomic publication and original-model check."""
from dataclasses import replace
from hashlib import sha256
import ctypes
import json
from math import prod
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from atlas_mlir_native import quantized_loop_c, quantized_row_blocks as blocks
from atlas_mlir_native import quantized_row_blocks_c as native
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.quantized_loop import execution_layout
from merlin.compile.launch_schedule import expand_launch_schedule
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.search import SearchLimits
from test_already_quantized import source_request
from test_looped_mnk import build_snapshot


ROOT=Path(__file__).resolve().parents[1]


def row_source():
    source=source_request(65,5,33,batch=(2,))
    return replace(source,nodes=tuple(replace(node,type=replace(node.type,shape=(2,65,1)))
        if node.id=='sx' else node for node in source.nodes))


class RowBlockExportTest(unittest.TestCase):
    def test_export_rejects_tile_children_before_C_publication(self):
        from types import SimpleNamespace
        source=row_source();plan=prepare_source(source)
        schedule=SimpleNamespace(plan=plan,bodies=(source,),launches=({'body_index':0},))
        manifest={'segments':[dict(path='segments/body_000',binary_sha256='0'*64)]}
        with patch.object(blocks,'verify_publication',return_value=(schedule,manifest)), \
             patch.object(quantized_loop_c,'prepare_c_package',return_value=(
                 dict(execution_route='bounded_k32_tile_dispatch',launches_per_invocation=6,restoration_placement='host'),{})):
            with self.assertRaisesRegex(ValueError,'complete counted children'):native.prepare_c_package(source,Path('unused'),'rows',target_source=ROOT)

    def test_export_carries_only_admitted_default_child_bounds(self):
        from types import SimpleNamespace
        source=row_source();schedule=SimpleNamespace(plan=prepare_source(source),bodies=(source,),launches=({'body_index':0},))
        manifest={'segments':[dict(path='segments/body_000',binary_sha256='0'*64)]}
        child=dict(execution_route='checked_counted_image',launches_per_invocation=1,restoration_placement='device')
        with tempfile.TemporaryDirectory(prefix='atlas-row-block-bound-') as directory:
            root=Path(directory);leaf=root/'segments/body_000';leaf.mkdir(parents=True)
            with patch.object(blocks,'verify_publication',return_value=(schedule,manifest)), \
                 patch.object(quantized_loop_c,'prepare_c_package',return_value=(child,{})):
                for proof in (dict(admitted=False,instruction_bound=100),dict(admitted=True,instruction_bound=1000001)):
                    with self.subTest(proof=proof):
                        (leaf/'restoration-instruction-bound.json').write_text(json.dumps(proof))
                        with self.assertRaisesRegex(ValueError,'admitted default instruction bound'):
                            native.prepare_c_package(source,root,'rows',target_source=ROOT)


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1' and os.environ.get('MERLIN_NATIVE_ROOT')
                     and os.environ.get('ATLAS_OOT_BIN_DIR'),'select native tools and original npu_model')
class RowBlockModelTest(unittest.TestCase):
    def test_forced_window_split_batched_row_exponents_tail_and_late_rejection(self):
        import torch
        model=Path(os.environ['ATLAS_NPU_MODEL_ROOT'])
        sys.path[:0]=[str(model),str(ROOT),str(ROOT/'tools'),str(ROOT.parent/'merlin/examples/atlas/whole-model/unchanged_fx')]
        from functional_runtime import FunctionalModelRuntime
        from model_runtime import READ,WRITE,MEMORY
        torch.set_num_threads(1)
        source=row_source();original=source.record()
        # This selected I/O window fits one complete32-row counted contraction,
        # not the full65 rows. No proof, instruction, allocation or wall override.
        child=replace(source,nodes=tuple(replace(node,type=replace(node.type,
            shape=(1,32,33) if node.id=='a' else (1,32,1) if node.id=='sx' else (1,32,5)))
            if node.id in ('a','sx','y') else node for node in source.nodes))
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            layout=execution_layout(prepare_source(child))
            window=layout['status_address']+layout['status_bytes']-0x90000000
        with codegen_options(CodegenOptions(io_memory_bytes=window)),tempfile.TemporaryDirectory(prefix='atlas-row-block-C-') as temporary:
            root=Path(temporary);snapshot=build_snapshot(root/'native')
            schedule=blocks.prepare_schedule(source,target_source=ROOT,limits=SearchLimits())
            self.assertIsNotNone(schedule)
            self.assertEqual([row['rows'] for row in schedule.launches],[32,32,1,32,32,1])
            self.assertEqual(len(schedule.bodies),2)
            compiled=root/'compiled'
            manifest=blocks.compile_source(snapshot,source,schedule,target_source=ROOT,destination=compiled,limits=SearchLimits())
            description,files=native.prepare_c_package(source,compiled,'rows',target_source=ROOT)
            self.assertEqual(description['restoration_placement'],'device')
            self.assertEqual(description['launches_per_invocation'],6)
            images=[];plans={}
            for row in manifest['segments']:
                leaf=compiled/row['path'];binary=(leaf/'program.bin').read_bytes();images.append(binary)
                plans[sha256(binary).hexdigest()]=json.loads((leaf/'execution_plan.json').read_text())
            hashes=[sha256(image).hexdigest() for image in images]
            expected_order=tuple(hashes[row['body_index']] for row in schedule.launches)
            self.assertEqual(expand_launch_schedule(description['device_launch_schedule'],image_sha256s=hashes),expected_order)
            for filename,data in files.items():
                path=root/filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            (root/'harness.c').write_text('''#include "rows.h"
static rows_context region; static rows_source_context source;
static uint8_t workspace[ROWS_WORKSPACE_BYTES]; static atlas_host host;
int invoke(int short_workspace,atlas_host_read32 read,atlas_host_write32 write,atlas_tile_write put,atlas_tile_read get,
           const void *const *inputs,void *const *outputs) {
 if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){read,write,0},(atlas_host_memory){1,0,0}))return 9;
 if(rows_bind(&region,&host,(atlas_tile_memory_io){put,get,0},workspace,sizeof(workspace)-(short_workspace?1u:0u),8u))return 8;
 if(rows_source_bind(&source,&region,0,0))return 7;
 return rows_source_dispatch(&source,inputs,outputs);}
''')
            subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',
                '-shared','-fPIC','rows.c','runtime/atlas_host.c','harness.c','-o','rows.so'],cwd=root,
                check=True,capture_output=True,timeout=30)
            function=ctypes.CDLL(str(root/'rows.so')).invoke
            function.argtypes=[ctypes.c_int,READ,WRITE,MEMORY,MEMORY,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_void_p)]
            a=np.empty((2,65,33),np.uint8)
            for batch in range(2):
                for row in range(65):a[batch,row,:]=0x38 if (batch+row)%2==0 else 0x40
            sx=(np.arange(130,dtype=np.int32)%3-1).reshape(2,65,1)
            w=np.repeat(np.array([0x38,0x40,0x38,0x40,0x38],np.uint8)[:,None],33,axis=1)
            sw=np.array([-1,0,1,0,-1],np.int32).reshape(5,1)
            buffers=(a,sx,w,sw);before=[array.tobytes() for array in buffers]
            inputs=(ctypes.c_void_p*4)(*(array.ctypes.data for array in buffers))
            output=np.full((2,65,5),0xa5a5,np.uint16);outputs=(ctypes.c_void_p*1)(output.ctypes.data)
            runtime=FunctionalModelRuntime(ROOT,model,images,root/'run',memory_bytes=window,image_plans=plans,trace_payloads=False)
            self.assertEqual(function(0,*runtime.callbacks,inputs,outputs),0,runtime.errors)
            av=np.where(a[...,0]==0x38,1.0,2.0);wv=np.where(w[:,0]==0x38,1.0,2.0)
            expected=np.ldexp((33*av[...,None]*wv).astype(np.float32),sx+sw.reshape(1,1,5))
            np.testing.assert_array_equal(output,(expected.view(np.uint32)>>16).astype(np.uint16))
            self.assertEqual([array.tobytes() for array in buffers],before)
            initial_launches=len(runtime.launches);output.fill(0xa5a5)
            self.assertNotEqual(function(1,*runtime.callbacks,inputs,outputs),0)
            self.assertEqual(len(runtime.launches),initial_launches)
            self.assertTrue(np.all(output==0xa5a5))
            # Only the final block's signed scale is invalid: earlier child
            # results remain private, and host restoration cannot mask status.
            sx[-1,-1,0]=200
            self.assertNotEqual(function(0,*runtime.callbacks,inputs,outputs),0)
            self.assertTrue(np.all(output==0xa5a5))
            result=runtime.finish();self.assertFalse(result['errors'])
            self.assertTrue(all(row['instructions']<=1000000 for row in result['launches']))
            self.assertEqual(source.record(),original)
            print('ROW_BLOCK_MODEL',window,len(schedule.bodies),len(schedule.launches),
                  [row['instructions'] for row in result['launches']],flush=True)


if __name__=='__main__':unittest.main()
