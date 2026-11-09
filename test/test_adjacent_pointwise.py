"""Enlargement keeps source order and proves every removed numeric guard."""
from dataclasses import replace
from types import SimpleNamespace
from math import prod
import json
import os
from pathlib import Path
import tempfile
import unittest

from atlas_mlir_native.adjacent_pointwise import alternatives, intermediate_domain_proof, prepare
from atlas_mlir_native.binding import TARGET_IDENTITY
from atlas_mlir_native.source_exp import POLICY
from merlin.semantic_compiler.model import (ConstantBinding, IndexMap, KernelRequest,
    SemanticNode, TensorType, pointwise_identity_maps)
from merlin.xdsl_dialects.lowering.global_plan import (RegionAlternative,
    BufferRepresentation, ValueRepresentation, CycleInterval)


def case(one='0000803f', barrier=False, shape=(32,32)):
    rank=len(shape)
    t=TensorType(shape,'f32','raw-bit-pattern')
    x=SemanticNode('x','input',(),t,effect='input')
    neg=SemanticNode('neg','input',(),t,effect='input')
    literal=SemanticNode('literal','constant',(),TensorType((),'f32','raw-bit-pattern'),effect='constant')
    identity=pointwise_identity_maps(rank,1)[-1]
    ones=SemanticNode('ones','gather',('literal',),t,index_maps=(IndexMap(rank,(),()),identity))
    exp=SemanticNode('exp','source_f32_exp_unqualified',('neg',),t,
        (('numeric_semantics','source-f32-exp-unqualified-v1'),),index_maps=pointwise_identity_maps(rank,1))
    add=SemanticNode('add','f32_add',('ones','exp'),t,
        (('numeric_semantics','ieee754-f32-add-rne-v1'),),index_maps=pointwise_identity_maps(rank,2))
    divide=SemanticNode('divide','f32_div',('x','add'),t,
        (('numeric_semantics','ieee754-f32-div-rne-v1'),),index_maps=pointwise_identity_maps(rank,2))
    source=KernelRequest((x,neg,literal,ones,exp,add,divide),('divide',),('external',),
        (('x','external'),('neg','external')),TARGET_IDENTITY,'diagnostic',
        constants=(ConstantBinding('literal','f32-le-raw',one),))
    def ports(keys):
        return tuple(BufferRepresentation(key,ValueRepresentation('host_memory','dense','f32')) for key in keys)
    choices=tuple(RegionAlternative('r'+str(i),(i,),'source',
        'host' if i==0 or barrier and i==2 else 'accelerator',CycleInterval.unknown('test'),
        ports(node.inputs),ports((node.id,)),metadata=(('kind','pointwise_math'),))
        for i,node in enumerate((ones,exp,add,divide)))
    adapter=SimpleNamespace(plan=lambda _:SimpleNamespace(selected=choices),bf16_pointwise=None,
        qualified_rne_add=None,source_exp_policy=POLICY)
    return source,adapter


class AdjacentPointwiseTest(unittest.TestCase):
    def setUp(self):
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        options=codegen_options(CodegenOptions(io_memory_bytes=67108864))
        options.__enter__()
        self.addCleanup(options.__exit__,None,None,None)

    def test_every_admitted_exp_code_proves_both_internal_boundaries(self):
        proof=intermediate_domain_proof()
        self.assertEqual(proof['checked_codes'],33540)
        self.assertGreaterEqual(proof['exp_range'][0],0x2000)
        self.assertLessEqual(proof['exp_range'][1],0x5f00)
        self.assertGreaterEqual(proof['sum_range'][0],0x3f80)
        self.assertLessEqual(proof['sum_range'][1],0x5f00)

    def test_hook_admits_ordered_connected_region(self):
        source,adapter=case()
        selected=[row for row in alternatives(adapter,source) if dict(row.metadata)['kind']=='adjacent_pointwise']
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0].nodes,(1,2,3))
        self.assertEqual([p.buffer for p in selected[0].outputs],['divide'])
        self.assertEqual(len(adapter._adjacent_groups),1)

    def test_actual_fixed64_mlp_chain_retains_three_complete_device_regions(self):
        from merlin.compile.semantic_regions import extract_semantic_region
        source,adapter=case(shape=(1,64,5632))
        original=source.record()
        choices=adapter.plan(source).selected
        self.assertEqual(alternatives(adapter,source),choices)
        self.assertEqual(adapter._adjacent_groups,{})
        self.assertEqual([row.placement for row in choices],['host','accelerator','accelerator','accelerator'])
        region=extract_semantic_region(source,('exp','add','divide'))
        with self.assertRaisesRegex(ValueError,'1056 > 512'):
            prepare(region,adapter,'ones')
        self.assertEqual(source.record(),original)

    def test_complete_tile_cutoff_keeps_smaller_chains_and_padding(self):
        for shape,admitted in (((1,64,2048),True),((32,5440),True),
                               ((32,5441),False),((1,512,5632),False)):
            with self.subTest(shape=shape):
                source,adapter=case(shape=shape)
                choices=adapter.plan(source).selected
                selected=alternatives(adapter,source)
                self.assertEqual(len(selected),len(choices)+int(admitted))
                self.assertEqual(len(adapter._adjacent_groups),int(admitted))

    @unittest.skipUnless(os.environ.get('MERLIN_NATIVE_ROOT') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                         'select native compiler tools')
    def test_retained_fixed64_chain_compiles_as_one_bounded_image(self):
        from atlas_mlir_native.adjacent_pointwise import compile_group, export_group
        from atlas_mlir_native.restoration_budget import instruction_bound
        from merlin.semantic_compiler.search import SearchLimits
        from test_looped_mnk import build_snapshot
        source,adapter=case(shape=(1,64,2048))
        original=source.record()
        alternatives(adapter,source)
        group,=adapter._adjacent_groups.values()
        self.assertEqual(sum(prod(plan.tile_counts) for plan in group['plans']),384)
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix='atlas-adjacent-bound-') as directory:
            folder=Path(directory)
            snapshot=build_snapshot(folder/'native')
            compiled=folder/'compiled'
            compile_group(group,snapshot,target_source=root,destination=compiled,limits=SearchLimits())
            bound=instruction_bound((compiled/'atlas-virtual.mlir').read_text(),
                                    (compiled/'atlas-machine.mlir').read_text(),
                                    json.loads((compiled/'physical-program.json').read_text()))
            self.assertTrue(bound['admitted'],bound)
            exported=export_group(group,compiled,'bounded',target_source=root)
            self.assertTrue(exported.files)
            self.assertEqual(exported.description['launches_per_invocation'],1)
            self.assertEqual(exported.description['source_request_digest'],group['region'].digest())
            receipt=os.environ.get('ATLAS_ADJACENT_POINTWISE_BOUND_RECEIPT')
            if receipt:
                Path(receipt).write_text(json.dumps({'shape':[1,64,2048],'member_tile_visits':384,
                    'instruction_bound':bound,'source_record_unchanged':source.record()==original},indent=2)+'\n')
        self.assertEqual(source.record(),original)

    def test_wrong_constant_or_host_barrier_cannot_enlarge(self):
        for kwargs in ({'one':'00000040'},{'barrier':True}):
            source,adapter=case(**kwargs)
            self.assertFalse(any(dict(row.metadata)['kind']=='adjacent_pointwise' for row in alternatives(adapter,source)))


if __name__=='__main__':unittest.main()
