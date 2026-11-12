"""Generic pointwise-chain enlargement with preserved internal admission."""
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from math import prod
from types import SimpleNamespace
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_mlir_native import adjacent_pointwise_chain as native
from atlas_mlir_native.binding import TARGET_IDENTITY
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, IndexMap, pointwise_identity_maps
from merlin.xdsl_dialects.lowering.global_plan import RegionAlternative, BufferRepresentation, ValueRepresentation, CycleInterval
from test_exponent_restoration import BIN, ROOT, compile_fixture


def case(rows=3,columns=64,divide=False):
    scalar=TensorType((rows,1),'f32','raw-bit-pattern');matrix=TensorType((rows,columns),'f32','raw-bit-pattern')
    def inp(name,type):return SemanticNode(name,'input',(),type,effect='input')
    boundaries=[inp('x',matrix),inp('average',scalar),inp('epsilon',scalar)]
    add=SemanticNode('add','f32_add',('average','epsilon'),scalar,(('numeric_semantics','ieee754-f32-add-rne-v1'),),index_maps=pointwise_identity_maps(2,2))
    if divide:
        boundaries.append(inp('numerator',scalar))
        middle=SemanticNode('middle','f32_div',('numerator','add'),scalar,(('numeric_semantics','ieee754-f32-div-rne-v1'),),index_maps=pointwise_identity_maps(2,2))
    else:
        middle=SemanticNode('middle','source_f32_rsqrt_unqualified',('add',),scalar,(('numeric_semantics','source-f32-rsqrt-unqualified-v1'),),index_maps=pointwise_identity_maps(2,1))
    mul=SemanticNode('multiply','f32_mul',('x','middle'),matrix,(('numeric_semantics','ieee754-f32-mul-rne-v1'),),
        index_maps=(pointwise_identity_maps(2,1)[0],IndexMap(2,((1,0),(0,0)),(0,0)),pointwise_identity_maps(2,1)[0]))
    source=KernelRequest((*boundaries,add,middle,mul),('multiply',),('external',),tuple((node.id,'external') for node in boundaries),TARGET_IDENTITY,'diagnostic')
    def ports(keys):return tuple(BufferRepresentation(key,ValueRepresentation('host_memory','dense','f32')) for key in keys)
    choices=tuple(RegionAlternative('r'+str(i),(i,),'source','accelerator',CycleInterval.unknown('test'),ports(node.inputs),ports((node.id,)),
        metadata=(('kind','pointwise_math' if i==1 else 'pointwise'),)) for i,node in enumerate((add,middle,mul)))
    adapter=SimpleNamespace(bf16_pointwise=None,qualified_rne_add=None,source_exp_policy=None)
    return source,adapter,choices


def descriptors(group):
    middle=['atlas.vpu_recip_bf16_bounded_tile','atlas.vpu_mul_bf16_zero_tile'] if group['precisions'][1].request.nodes[-1].op=='divide_bf16_lut' else ['atlas.vpu_sqrt_bf16_bounded_tile','atlas.vpu_recip_bf16_bounded_tile']
    return ('atlas.vpu_add_bf16_tile',middle,'atlas.vpu_mul_bf16_zero_tile')


def shared_source_case():
    source,adapter,choices=case()
    nodes=tuple(replace(node,inputs=('x','epsilon')) if node.id=='add' else node for node in source.nodes if node.id!='average')
    return replace(source,nodes=nodes,input_storages=(('x','external'),('epsilon','external'))),adapter,choices


class PointwiseChainTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        context=codegen_options(CodegenOptions(io_memory_bytes=67108864))
        context.__enter__()
        self.addCleanup(context.__exit__,None,None,None)

    def test_specialized_refusal_cannot_reenter_as_an_oversized_generic_chain(self):
        from atlas_mlir_native.composition_adapter import AdjacentDeviceAdapter, _ADJACENT_PATTERNS
        from merlin.compile.semantic_regions import extract_semantic_region
        from test_adjacent_pointwise import case as exp_case
        source,base=exp_case(shape=(1,64,5632))
        original=source.record()
        choices=base.plan(source).selected
        adapter=AdjacentDeviceAdapter(base)
        selected=adapter.application_region_alternatives(source)
        self.assertEqual(selected,choices)
        for _,state in _ADJACENT_PATTERNS:
            self.assertEqual(getattr(adapter,state),{})
        with self.assertRaisesRegex(ValueError,'1056 > 512'):
            native.prepare(extract_semantic_region(source,('exp','add','divide')),adapter)
        self.assertEqual(source.record(),original)

    def test_shared_cutoff_retains_384_dense_and_compact_rms_chains(self):
        from merlin.compile.semantic_regions import extract_semantic_region
        from test_adjacent_pointwise import case as exp_case
        source,adapter=exp_case(shape=(1,64,2048))
        group=native.prepare(extract_semantic_region(source,('exp','add','divide')),adapter)
        self.assertEqual(sum(prod(plan.tile_counts) for plan in group['plans']),384)
        for divide in (False,True):
            source,adapter,choices=case(rows=64,columns=2048,divide=divide)
            selected=native.alternatives(adapter,source,choices)
            self.assertEqual(len(selected),len(choices)+1)
            group,=adapter._pointwise_chain_groups.values()
            self.assertEqual(sum(prod(plan.tile_counts) for plan in group['plans']),132)

    @unittest.skipUnless((BIN/'atlas-opt').is_file(),'Atlas compiler binaries unavailable')
    def test_retained_fixed64_rms_chain_passes_default_proof_and_instruction_bound(self):
        from atlas_mlir_native.restoration_budget import instruction_bound
        for divide in (False,True):
            source,adapter,_=case(rows=64,columns=2048,divide=divide)
            original=source.record();group=native.prepare(source,adapter)
            virtual=native.virtual_group(group,descriptors(group),tools=BIN)
            machine,program=compile_fixture(virtual)
            bound=instruction_bound(virtual,machine,program)
            self.assertTrue(bound['admitted'],bound)
            self.assertEqual(source.record(),original)

    def test_hook_and_layout_ownership(self):
        source,adapter,choices=case()
        result=native.alternatives(adapter,source,choices)
        groups=[row for row in result if dict(row.metadata).get('kind')=='adjacent_pointwise_chain']
        self.assertEqual(len(groups),1);self.assertEqual(groups[0].nodes,(0,1,2))
        group=next(iter(adapter._pointwise_chain_groups.values()))
        self.assertEqual([edge['kind'] for edge in group['edges']],['identity','broadcast_column'])
        self.assertTrue(group['plans'][2].compact_broadcasts)
        self.assertEqual(group['plans'][2].input_shape(1),(3,32))
        fast=replace(groups[0],metadata=(('kind','adjacent_pointwise'),))
        self.assertFalse(any(dict(row.metadata).get('kind')=='adjacent_pointwise_chain' for row in native.alternatives(adapter,source,(*choices,fast))))
        barrier=(*choices[:1],replace(choices[1],placement='host'),choices[2])
        self.assertEqual(native.alternatives(adapter,source,barrier),barrier)
        escaped=replace(source,outputs=('add','multiply'),output_storages=('external','external'))
        with self.assertRaises(ValueError):native.prepare(escaped,adapter)

    @unittest.skipUnless((BIN/'atlas-opt').is_file(),'Atlas compiler binaries unavailable')
    def test_actual_linked_virtual_passes_default_checks(self):
        for divide in (False,True):
            source,adapter,_=case(divide=divide);group=native.prepare(source,adapter)
            virtual=native.virtual_group(group,descriptors(group),tools=BIN)
            _,program=compile_fixture(virtual)
            self.assertEqual(sum(row['operation']=='atlas.trap' for row in program['instructions']),1)

    @unittest.skipUnless((BIN/'atlas-opt').is_file(),'Atlas compiler binaries unavailable')
    def test_registered_selected_pipeline_lowers_linked_image_once(self):
        from atlas_mlir_native import binding
        from atlas_mlir_native.region_composition import compile_adjacent_virtual

        source, adapter, _ = case()
        group = native.prepare(source, adapter)
        virtual = native.virtual_group(group, descriptors(group), tools=BIN)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {'ATLAS_NATIVE_PIPELINE': 'optimized', 'ATLAS_CAPTURE_POSTSPILL': '1'}):
                with patch.object(binding, '_run', wraps=binding._run) as run:
                    compile_adjacent_virtual(virtual, destination=temporary, target_source=ROOT)
            compiles = [call.args[0] for call in run.call_args_list
                        if any(option in {'--atlas-compile', '--atlas-compile-unscheduled'} for option in call.args[0])]
            self.assertEqual(len(compiles), 1)
            self.assertEqual(compiles[0][-2:], [*binding.NATIVE_PIPELINES['optimized'], '-'])
            self.assertIn('--mlir-print-ir-before=lower-atlas-virtual-to-machine', compiles[0])
            self.assertTrue((Path(temporary)/'mandatory-spilled-virtual.mlir').is_file())
            program = json.loads((Path(temporary)/'physical-program.json').read_text())
            self.assertEqual(sum(row['operation']=='atlas.trap' for row in program['instructions']),1)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_original_model_c_abi_guards_atomic_rejection_and_shared_public_input(self):
        groups=[]
        with tempfile.TemporaryDirectory(prefix='atlas-pointwise-chain-') as temporary:
            directory=Path(temporary)
            for index,(label,fixture) in enumerate((('rsqrt',case()),('divide',case(divide=True)),('shared',shared_source_case()))):
                source,adapter,_=fixture;group=native.prepare(source,adapter)
                virtual=native.virtual_group(group,descriptors(group),tools=BIN)
                _,program=compile_fixture(virtual)
                binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
                folder=directory/label;folder.mkdir()
                records={'manifest.json':{},'execution_plan.json':native.execution_plan(group),
                    'enlargement.json':dict(proof=group['proof'],members=[],precision_plan=[p.record() for p in group['precisions']],
                        plans=[p.record() for p in group['plans']],source_request_digest=source.digest())}
                for name,value in records.items():(folder/name).write_text(json.dumps(value))
                (folder/'program.bin').write_bytes(binary)
                with patch('merlin.semantic_compiler.target_binding.verify_native_publication'):
                    exported=native.export_group(group,folder,'chain',target_source=ROOT)
                for name,data in exported.files.items():
                    path=folder/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
                (folder/'client.c').write_text('''#include "chain.h"
int call(atlas_host_read32 rd,atlas_host_write32 wr,atlas_tile_write put,atlas_tile_read get,
         const float *const *inputs,float *output) {
  atlas_host host;chain_context raw;chain_source_context source;
  uint8_t scratch[CHAIN_WORKSPACE_BYTES],conversion[CHAIN_SOURCE_WORKSPACE_BYTES];
  if(atlas_host_init(&host,(atlas_host_map){0x20000u,0x40000u,131072u},(atlas_host_mmio){rd,wr,0},(atlas_host_memory){1,0,0}) ||
     chain_bind(&raw,&host,(atlas_tile_memory_io){put,get,0},scratch,sizeof(scratch),8u) ||
     chain_source_bind(&source,&raw,conversion,sizeof(conversion))) return -1;
  return chain_source_run(&source,inputs,output,0);
}''')
                result=subprocess.run(['cc','-std=c11','-O2','-Wall','-Wextra','-Werror','-shared','-fPIC',
                    'chain.c','runtime/atlas_host.c','client.c','-o','chain.so'],cwd=folder,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                values=np.resize(np.array([4,1,2,4],np.float32),(3,64));epsilon=np.zeros((3,1),np.float32)
                average=np.array([[4],[16],[1]],np.float32)
                cases=[]
                for failure in (False,True):
                    data={'x':values.copy(),'epsilon':epsilon.copy(),'average':average.copy(),'numerator':np.ones((3,1),np.float32)}
                    if label=='shared':data['x'][:,0]=4
                    if failure:
                        if label=='shared':data['x'][0,0]=0
                        else:data['average'][0,0]=0
                    expected=values*(np.array([[.5],[.25],[1]],np.float32) if label=='rsqrt' else
                                     np.array([[.25],[.0625],[1]],np.float32) if label=='divide' else .5)
                    cases.append(dict(name='zero_internal' if failure else 'valid',status=2 if failure else 0,
                        inputs=[data[key].tolist() for key in group['inputs']],expected=expected.tobytes().hex()))
                if label=='divide':
                    data={'x':np.ones((3,64),np.float32),'epsilon':epsilon.copy(),
                          'average':np.full((3,1),2.0**-63,np.float32),'numerator':np.full((3,1),2.0**63,np.float32)}
                    cases.append(dict(name='late_boundary_out_of_domain',status=2,
                        inputs=[data[key].tolist() for key in group['inputs']],expected=''))
                if label=='shared':
                    self.assertEqual(exported.description['source_input_indices'],[0,1,0])
                groups.append(dict(name=label,binary=binary.hex(),library=str(folder/'chain.so'),execution=native.execution_plan(group),
                                   arena_bytes=group['arena_bytes'],output_shape=[3,64],cases=cases))
            spec=directory/'spec.json';receipt=directory/'receipt.json';spec.write_text(json.dumps(groups))
            result=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('pointwise_chain_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=os.environ.get('ATLAS_POINTWISE_CHAIN_RECEIPT')
            if output:Path(output).write_bytes(receipt.read_bytes())


if __name__=='__main__':unittest.main()
