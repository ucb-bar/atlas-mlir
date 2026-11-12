"""Device readout against the retained independent host-domain reference."""
from dataclasses import replace
import json
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from atlas_mlir_native.already_quantized import prepare_source, restore_bf16
from atlas_mlir_native.binding import DRAM_BASE, NATIVE_PIPELINES, NATIVE_CHECKS
from atlas_mlir_native.exponent_restoration import Restoration
from atlas_mlir_native.quantized_loop import execution_layout
from atlas_mlir_native.region_composition import rematerialize_scalar_constants
from merlin.semantic_compiler.model import TensorType
from test_already_quantized import source_request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get('ATLAS_OOT_BIN_DIR', ROOT/'build/bin'))
S = '!atlas.virtual_state'
T = '!atlas.virtual_bf16'


def raw_fixture(plan, layout):
    """Feed completed readouts through the typed hook without an MXU."""
    batches = int(np.prod(plan.batch_shape))
    out = layout['output_addresses'][0]
    restoration = Restoration(plan, layout, vector=False, logical_rows=False)
    def c(name, value):
        return f'%{name} = arith.constant {value-(1<<32) if value >= 1<<31 else value} : i32'
    pages = restoration.pages()
    carried = [i for i, page in enumerate(pages) if page[2]]
    page_names = ['restore_lpage', 'restore_rpage']
    args = ''.join(', %'+page_names[i] for i in carried)
    types = ', i32'*len(carried)
    params = ''.join(', %'+page_names[i]+': i32' for i in carried)
    initial = ''.join(', %page_base'+str(i) for i in carried)
    lines = ['module {',f'func.func @looped_matmul() -> {S} attributes {{atlas.input_dram_base = {DRAM_BASE} : i64, atlas.output_dram_base = {out} : i64, atlas.output_dram_bytes = {layout["output_bytes"]+layout["status_bytes"]} : i64}} {{',
             f'%start = "atlas.virtual_start"() : () -> {S}',*restoration.entry(),c('zero_m',0),c('input_base',DRAM_BASE),c('output_base',out)]
    for i, page in enumerate(pages):
        lines.append(c('page_base'+str(i), page[0]))
    lines += [f'cf.br ^batch_header(%restore_start, %zero_m, %input_base, %output_base{initial} : {S}, i32, i32, i32{types})',
             f'^batch_header(%bs: {S}, %batch: i32, %raw_address: i32, %batch_y: i32{params}):',c('batch_count',batches),
             '%more = arith.cmpi ult, %batch, %batch_count : i32',
             f'cf.cond_br %more, ^batch_body(%bs : {S}), ^all_done(%bs : {S})',
             f'^batch_body(%batch_start: {S}):',f'cf.br ^m_body(%batch_start : {S})',
             f'^m_body(%mb: {S}):','%ytile = arith.addi %batch_y, %zero_m : i32',c('input_bytes',2048),
             f'%read, %raw = "atlas.virtual_dma_load_bf16"(%mb, %raw_address, %input_bytes) : ({S}, i32, i32) -> ({S}, {T})',
             f'cf.br ^store(%read, %raw : {S}, {T})',f'^store(%ss: {S}, %result: {T}):',c('out_bytes',2048)]
    lines += restoration.tile(exponent_addresses=[page_names[i] if i in carried else 'page_base'+str(i) for i in range(2)])
    lines += [c('one',1),c('stride',2048),'%batch_next = arith.addi %batch, %one : i32',
             '%raw_next = arith.addi %raw_address, %stride : i32','%out_next = arith.addi %batch_y, %stride : i32']
    for i in carried:
        lines += [c('page_stride'+str(i),pages[i][2]),f'%next_page{i} = arith.addi %{page_names[i]}, %page_stride{i} : i32']
    next_pages = ''.join(', %next_page'+str(i) for i in carried)
    lines += [f'cf.br ^batch_header(%stored, %batch_next, %raw_next, %out_next{next_pages} : {S}, i32, i32, i32{types})',
             f'^all_done(%final: {S}):',*restoration.finish('final'),'}}']
    return rematerialize_scalar_constants('\n'.join(lines)+'\n')


def compile_fixture(source, *, pipeline='mandatory'):
    result = subprocess.run([str(BIN/'atlas-opt'),*NATIVE_PIPELINES[pipeline],'-'],input=source,
                            capture_output=True,text=True,timeout=30)
    if result.returncode:
        raise AssertionError(result.stderr)
    checked = subprocess.run([str(BIN/'atlas-opt'),*NATIVE_CHECKS,'--verify-atlas-timing','-'],input=result.stdout,
                             capture_output=True,text=True,timeout=30)
    if checked.returncode:
        raise AssertionError(checked.stderr)
    emitted = subprocess.run([str(BIN/'atlas-emit'),'--program-json','-'],input=result.stdout,
                             capture_output=True,text=True,timeout=30)
    if emitted.returncode:
        raise AssertionError(emitted.stderr)
    return result.stdout,json.loads(emitted.stdout)


@unittest.skipUnless((BIN/'atlas-opt').is_file(), 'Atlas compiler binaries unavailable')
class ExponentRestorationTest(unittest.TestCase):
    def test_counter_freshness_distinguishes_dynamic_iterations_with_equal_writer_sets(self):
        from atlas_mlir_native.restoration_budget import _fresh_counter_read, _natural_loops
        # Equal global {seed,update} writer IDs do not distinguish a load made
        # in an enclosing iteration from the current inner-header read.
        graph={0:[1],1:[2,4],2:[3],3:[1],4:[]}
        restores={10:(100,1,7),20:(200,0,7),30:(300,2,7)}
        effects={0:[('restore',200,7,20)],1:[('restore',100,7,10)],
                 2:[('restore',300,7,30)],3:[('save',400,7,50)],4:[]}
        members={1,2,3}
        self.assertFalse(_fresh_counter_read(1,members,10,20,restores=restores,effects=effects,successors=graph))
        self.assertTrue(_fresh_counter_read(1,members,10,30,restores=restores,effects=effects,successors=graph))
        effects[1].insert(0,('save',600,7,70))
        self.assertFalse(_fresh_counter_read(1,members,10,30,restores=restores,effects=effects,successors=graph))
        effects[1].pop(0)
        effects[2].insert(0,('save',500,7,60))
        self.assertFalse(_fresh_counter_read(1,members,10,30,restores=restores,effects=effects,successors=graph))
        # A nested body can execute the same update writer twice before
        # returning to the outer header. With step2^31, those two writes wrap
        # its counter back to zero despite identical global writer ID sets.
        nested={0:[1],1:[2,6],2:[3],3:[4,5],4:[3],5:[1],6:[]}
        nested_effects={block:[] for block in nested}
        nested_effects[1]=[('restore',100,7,10)]
        nested_effects[4]=[('restore',300,7,30),('save',400,7,50)]
        nested_restores={10:(100,1,7),30:(300,4,7)}
        self.assertFalse(_fresh_counter_read(1,{1,2,3,4,5},10,30,restores=nested_restores,
                                            effects=nested_effects,successors=nested))
        # Two entries into a cycle have no natural header. Dropping certified
        # backedges must expose it as an uncounted cycle and fail closed.
        with self.assertRaisesRegex(ValueError,'uncounted CFG cycle'):
            _natural_loops({0:[1,2],1:[2],2:[1,3],3:[]},0)

    def test_large_images_explicitly_refuse_unbudgeted_scalar_fallback(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        from atlas_mlir_native.restoration_budget import instruction_bound
        plan=prepare_source(source_request(m=64,n=96,k=65,batch=()))
        layout=execution_layout(plan)
        self.assertEqual(layout['restoration_admission']['scalar_fallback_tile_budget'],0)
        self.assertEqual(layout['restoration_admission']['successful_exponent_domain'],'vector_domain_only')
        source=virtual_source(plan,layout)
        self.assertNotIn('^restore_word_header',source)
        self.assertIn('^restore_scalar',source)
        machine,physical=compile_fixture(source)
        proof=instruction_bound(source,machine,physical)
        self.assertTrue(proof['admitted'])
        self.assertLessEqual(proof['instruction_bound'],1_000_000)
        # Every encoded word, including both numeric arms and control delay
        # slots, belongs to exactly one proved block in the bound receipt.
        self.assertEqual(sum(block['encoded_words'] for block in proof['blocks']),len(physical['instructions']))
        broken=json.loads(json.dumps(physical))
        broken['instructions'][0]['fields'].pop('atlas.virtual_cfg_block')
        with self.assertRaisesRegex(ValueError,'ownership'):
            instruction_bound(source,machine,broken)
        with self.assertRaisesRegex(ValueError,'final loop condition'):
            instruction_bound(source,machine.replace('predicate = 6 : i32','predicate = 7 : i32'),physical)
        helper=json.loads(json.dumps(physical))
        helper['instructions'][0]['fields']['atlas.virtual_cfg_helper']='pack'
        with self.assertRaisesRegex(ValueError,'helper control'):
            instruction_bound(source,machine,helper)
        # Keep the same unsigned comparison bounds, but change all captured
        # unit increments to zero. A bound-only inventory would miss this.
        zero_step=re.sub(r'constant = 1 : i32', 'constant = 0 : i32',machine)
        with self.assertRaisesRegex(ValueError,'zero final induction step'):
            instruction_bound(source,zero_step,physical)
        with self.assertRaisesRegex(ValueError,'instruction bound:'):
            instruction_bound(source,re.sub(r'constant = 1 : i32','constant = -1 : i32',machine),physical)

    def test_small_images_retain_scalar_domain_and_overbudget_images_are_refused(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        from atlas_mlir_native.restoration_budget import instruction_bound
        plan=prepare_source(source_request(m=32,n=32,k=65,batch=(2,)))
        layout=execution_layout(plan)
        self.assertEqual(layout['restoration_admission']['scalar_fallback_tile_budget'],2)
        self.assertEqual(layout['restoration_admission']['successful_exponent_domain'],'full_scalar_admission')
        source=virtual_source(plan,layout)
        self.assertIn('^restore_word_header',source)
        machine,physical=compile_fixture(source)
        proof=instruction_bound(source,machine,physical)
        self.assertTrue(proof['admitted'])
        # An image whose charged physical body is too large cannot pass the
        # admission predicate, even if its observed executions take a fast arm.
        expanded=json.loads(json.dumps(physical))
        expanded['instructions'] *= 10
        for index,word in enumerate(expanded['instructions']):
            # Repeated dict references need independent records for indices.
            expanded['instructions'][index]={**word,'word_index':index}
        self.assertFalse(instruction_bound(source,machine,expanded)['admitted'])

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_zero_fallback_budget_refuses_old_admitted_exponents_and_status_is_sticky(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        from atlas_mlir_native.restoration_budget import instruction_bound
        plan=prepare_source(source_request(m=32,n=96,k=32,batch=()))
        layout=execution_layout(plan)
        source=virtual_source(plan,layout)
        machine,physical=compile_fixture(source)
        proof=instruction_bound(source,machine,physical)
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in physical['instructions'])
        memory_bytes=layout['status_address']-DRAM_BASE+layout['status_bytes']
        cases=[]
        for label,lhs,rhs in [('all_fallback_old_admitted',48,[0]*96),
                              ('first_fallback_then_fast',0,[48]*32+[0]*64)]:
            memory=bytearray(memory_bytes)
            for address,size in zip(layout['input_addresses'],layout['input_buffer_bytes'],strict=True):
                at=address-DRAM_BASE
                memory[at:at+size]=bytes([0x38])*size
            at=layout['exponent_addresses'][0]-DRAM_BASE
            memory[at:at+4]=np.array(lhs,dtype='<i4').tobytes()
            for page in range(3):
                at=layout['exponent_addresses'][1]-DRAM_BASE+page*2048
                memory[at:at+128]=np.array(rhs[page*32:page*32+32],dtype='<i4').tobytes()
            # 32 unit products have raw BF16 0x4200; exponent48 remains a
            # finite normal in the old full scalar domain (0x5a00).
            self.assertEqual(int(restore_bf16(np.array(0x4200,dtype=np.uint16),
                                             np.array(48,np.int32),np.array(0,np.int32))),0x5a00)
            expected=bytes(6144) if lhs else bytes(2048)+np.full(2048,0x4200,dtype='<u2').tobytes()
            cases.append(dict(label=label,rejected=True,memory=memory.hex(),expected=expected.hex()))
        with tempfile.TemporaryDirectory(prefix='atlas-bounded-restoration-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json'
            spec.write_text(json.dumps(dict(binary=binary.hex(),layout=layout,cases=cases)))
            process=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('exponent_restoration_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            results=json.loads(receipt.read_text())
            self.assertTrue(all(row['instructions']<=proof['instruction_bound'] for row in results['cases']))
            results['instruction_bound']=proof
            destination=os.environ.get('ATLAS_BOUNDED_RESTORATION_RECEIPT')
            if destination:Path(destination).write_text(json.dumps(results,indent=2)+'\n')

    def test_counted_batch_has_one_body_and_one_default_compile(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        plan=prepare_source(source_request(m=32,n=32,k=65,batch=(32,)))
        virtual=virtual_source(plan,execution_layout(plan))
        self.assertEqual(virtual.count('^batch_header('), 3)
        self.assertEqual(virtual.count('"atlas.virtual_mxu_reset"'), 1)
        self.assertEqual(virtual.count('atlas.exponent_scratch = "status"'), 1)
        _, program=compile_fixture(virtual)
        self.assertLess(len(program['instructions']), 32768)

    def test_raw_completed_readout_fixture_compiles_with_default_checks(self):
        source=source_request(m=32,n=32,k=32,batch=())
        plan=prepare_source(source);layout=execution_layout(plan)
        virtual=raw_fixture(plan,layout)
        machine,program=compile_fixture(virtual)
        self.assertTrue(program['instructions'])
        self.assertIn('slot_index_load',machine)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_nonlogical_readout_padding_still_rejects_invalid_bits(self):
        source=source_request(m=1,n=1,k=32,batch=(2,))
        source=replace(source,nodes=tuple(replace(node,type=TensorType((2,1,1),'i32','raw-bit-pattern')) if node.id=='sx' else node for node in source.nodes))
        plan=prepare_source(source);layout=execution_layout(plan)
        _,physical=compile_fixture(raw_fixture(plan,layout))
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in physical['instructions'])
        size=layout['status_address']-DRAM_BASE+layout['status_bytes'];cases=[]
        for label,word in [('padding_negative_zero',0x8000),('padding_subnormal',1),('padding_infinity',0x7f80),('padding_nan',0xffc1)]:
            memory=bytearray(size)
            # The logical (0,0) result and the entire later tile remain zero.
            memory[2046:2048]=word.to_bytes(2,'little')
            cases.append(dict(label=label,rejected=True,memory=memory.hex(),expected=None))
        with tempfile.TemporaryDirectory(prefix='atlas-padding-restoration-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json'
            spec.write_text(json.dumps(dict(binary=binary.hex(),layout=layout,cases=cases)))
            process=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('exponent_restoration_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            destination=os.environ.get('ATLAS_PAD_RESTORATION_RECEIPT')
            if destination:Path(destination).write_bytes(receipt.read_bytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_ordered_k65_mxu_readout_restores_row_scales_on_original_model(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        from atlas_mlir_native.already_quantized import pack_operand
        from atlas_mlir_native.layout import pack_bf16_pair_halves
        source=source_request(m=32,n=32,k=65,batch=(2,))
        source=replace(source,nodes=tuple(replace(node,type=TensorType((2,32,1),'i32','raw-bit-pattern')) if node.id=='sx' else node for node in source.nodes))
        plan=prepare_source(source);layout=execution_layout(plan)
        machine,physical=compile_fixture(virtual_source(plan,layout))
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in physical['instructions'])
        size=layout['status_address']-DRAM_BASE+layout['status_bytes']
        payloads=[]
        for label,lhs,rhs,zeros,rejected in [('row_scales',None,None,False,False),
                ('vector_scale_min',-50,-50,False,False),('vector_scale_max',47,47,False,False),
                ('zero_scale_min',-127,-6,True,False),('bad_lhs',128,0,True,True),
                ('bad_paired_scale',-127,-7,True,True),('underflow',-127,-6,False,True),
                ('overflow',127,0,False,True)]:
            sx=np.zeros((2,32,1),np.int32)
            sx[0]=(np.arange(32,dtype=np.int32)%5-2).reshape(32,1) if lhs is None else np.full((32,1),lhs,np.int32)
            sx[1]=(np.arange(32,dtype=np.int32)%3-1).reshape(32,1) if lhs is None else 0
            sw=(np.arange(32,dtype=np.int32)%3-1).reshape(32,1) if rhs is None else np.full((32,1),rhs,np.int32)
            values=np.full((32,65),0 if zeros else 0x38,np.uint8)
            codes=b''.join(pack_operand(values,k=k) for k in range(0,65,32))
            memory=bytearray([0xa5])*size
            for index,address in enumerate(layout['input_addresses']):
                payload=codes*2 if index==0 else codes
                at=address-DRAM_BASE;memory[at:at+len(payload)]=payload
            for exponents,address in zip((sx,sw),layout['exponent_addresses'],strict=True):
                at=address-DRAM_BASE
                pages=exponents.reshape(-1,32)
                for page,values in enumerate(pages):
                    first=at+page*2048;memory[first:first+2048]=bytes(2048);memory[first:first+128]=values.tobytes()
            expected=None
            if not rejected:
                core=np.full((32,32),0 if zeros else 0x4282,np.uint16)
                expected=b''.join(pack_bf16_pair_halves(restore_bf16(core,sx[b],sw.reshape(1,32)).tobytes()) for b in range(2)).hex()
            payloads.append(dict(label=label,rejected=rejected,memory=memory.hex(),expected=expected))
        with tempfile.TemporaryDirectory(prefix='atlas-mxu-restoration-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json'
            spec.write_text(json.dumps(dict(binary=binary.hex(),layout=layout,cases=payloads)))
            process=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('exponent_restoration_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            destination=os.environ.get('ATLAS_MXU_RESTORATION_RECEIPT')
            if destination:Path(destination).write_bytes(receipt.read_bytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_unroll8_odd_k_tail_preserves_every_bf16_reseed(self):
        from atlas_mlir_native.quantized_loop import virtual_source
        from atlas_mlir_native.already_quantized import pack_operand
        source=source_request(m=32,n=32,k=321,batch=())
        plan=prepare_source(source);layout=execution_layout(plan)
        _,physical=compile_fixture(virtual_source(plan,layout))
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in physical['instructions'])
        # Eight unit-product tiles produce256. The following2^-7 tile is
        # discarded at its BF16 readout; a -256 tile then cancels to+0.
        # Keeping extra accumulator precision would incorrectly retain2^-7.
        lhs=np.full((32,321),0x38,np.uint8);rhs=lhs.copy()
        lhs[:,256:288]=8;rhs[:,256:288]=8
        rhs[:,288:320]=0xd0
        lhs[:,320]=0;rhs[:,320]=0
        payloads=[b''.join(pack_operand(value,k=k) for k in range(0,321,32)) for value in (lhs,rhs)]
        cases=[]
        size=layout['status_address']-DRAM_BASE+layout['status_bytes']
        for label,sx,sw,rejected in [('fast',0,0,False),('fallback',-127,-6,False),('rejected',128,0,True)]:
            memory=bytearray(size)
            for address,payload in zip(layout['input_addresses'],payloads,strict=True):
                at=address-DRAM_BASE;memory[at:at+len(payload)]=payload
            for address,scale,count in zip(layout['exponent_addresses'],(sx,sw),(1,32),strict=True):
                at=address-DRAM_BASE;payload=np.full(count,scale,dtype='<i4').tobytes();memory[at:at+len(payload)]=payload
            cases.append(dict(label=label,rejected=rejected,memory=memory.hex(),expected=bytes(layout['output_bytes']).hex()))
        with tempfile.TemporaryDirectory(prefix='atlas-unroll8-restoration-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json'
            spec.write_text(json.dumps(dict(binary=binary.hex(),layout=layout,cases=cases)))
            result=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('exponent_restoration_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            output=os.environ.get('ATLAS_UNROLL8_RESTORATION_RECEIPT')
            if output:Path(output).write_bytes(receipt.read_bytes())

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_bit_exact_domain_edges_and_early_rejection_survives_later_tiles(self):
        source=source_request(m=32,n=32,k=32,batch=(2,))
        source=replace(source,nodes=tuple(replace(node,type=TensorType((2,1,1),'i32','raw-bit-pattern')) if node.id=='sx' else node for node in source.nodes))
        plan=prepare_source(source);layout=execution_layout(plan)
        machine,physical=compile_fixture(raw_fixture(plan,layout))
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in physical['instructions'])
        memory_bytes=layout['status_address']-DRAM_BASE+layout['status_bytes']
        cases=[('mixed_normal',0,0,0x3f80,False),('cancel_min',-127,127,0x3f80,False),
               ('cancel_max',127,-127,0x3f80,False),('scale_min',-127,-6,0x7f00,False),
               ('scale_max',0,127,0x0080,False),('zero_scale_min',-127,-6,0,False),
               ('lhs_low',-128,0,0,True),('lhs_high',128,0,0,True),
               ('rhs_low',0,-128,0,True),('rhs_high',0,128,0,True),
               ('lhs_i32_min',-2147483648,0,0,True),('rhs_i32_max',0,2147483647,0,True),
               ('paired_low',-127,-7,0,True),('paired_high',1,127,0,True),
               ('underflow',-1,0,0x0080,True),('overflow',1,0,0x7f00,True),
               ('negative_zero',0,0,0x8000,True),('subnormal',0,0,0x0001,True),
               ('infinity',0,0,0x7f80,True),('nan',0,0,0xffc1,True)]
        payloads=[]
        for label,lhs,rhs,word,rejected in cases:
            # The first tile alone contains the invalid value; the second tile
            # remains admitted, exposing accidental status reset at a tile edge.
            raw=np.full((2,1024),0x3f80,dtype='<u2')
            raw[0].fill(word)
            raw[1].fill(max(1,min(254,127-rhs))<<7)
            sx=np.array([lhs,0],dtype='<i4')
            sw=np.full((32,1),rhs,dtype='<i4')
            if label in ('mixed_normal','cancel_min','cancel_max'):
                raw[0]=np.resize(np.array([0,0x0080,0x7f00,0xbf81,0x3f80,0x8001],'<u2'),1024)
                raw[0,5::6]=0x3f81
            memory=bytearray([0xa5])*memory_bytes
            memory[:raw.nbytes]=raw.tobytes()
            ea,ew=layout['exponent_addresses']
            for b in range(2):memory[ea-DRAM_BASE+b*2048:ea-DRAM_BASE+b*2048+4]=sx[b].tobytes()
            memory[ew-DRAM_BASE:ew-DRAM_BASE+2048]=bytes(2048)
            memory[ew-DRAM_BASE:ew-DRAM_BASE+128]=sw.tobytes()
            expected_bytes=None
            if not rejected:
                # Physical half layout is independent of exponent restoration:
                # scalar lhs/rhs apply identically to every lane of this tile.
                expected=raw.copy()
                for b in range(2):
                    expected[b]=restore_bf16(raw[b],np.array(sx[b],np.int32),np.array(rhs,np.int32))
                expected_bytes=expected.tobytes().hex()
            payloads.append(dict(label=label,rejected=rejected,memory=memory.hex(),expected=expected_bytes))
        with tempfile.TemporaryDirectory(prefix='atlas-restoration-model-') as directory:
            directory=Path(directory)
            spec=directory/'spec.json'
            spec.write_text(json.dumps(dict(binary=binary.hex(),layout=layout,cases=payloads)))
            receipt=directory/'receipt.json'
            runner=Path(__file__).with_name('exponent_restoration_model.py')
            process=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(runner),str(spec),str(receipt),str(ROOT)],
                                   capture_output=True,text=True,timeout=180)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            destination=os.environ.get('ATLAS_RESTORATION_RECEIPT')
            if destination:
                Path(destination).write_bytes(receipt.read_bytes())



if __name__=='__main__':
    unittest.main()
