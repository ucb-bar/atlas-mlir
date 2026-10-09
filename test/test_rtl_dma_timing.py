"""Explicit selected conditional DMA timing policy; no private evidence defaults.

Run with ATLAS_OOT_BIN_DIR, ATLAS_RTL_EVIDENCE_REPORT and
ATLAS_RTL_DMA_EVIDENCE_REPORT pointing to the selected compiler/evidence.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from test_delay_insertion import OPT, EMIT, addi, program, run
from test_rtl_timing import HALT, MARKER, VLOAD, word_base

REPORT=os.environ.get('ATLAS_RTL_EVIDENCE_REPORT')
DMA_REPORT=os.environ.get('ATLAS_RTL_DMA_EVIDENCE_REPORT')
CONSUMERS=('--insert-atlas-delays','--schedule-atlas-stream')

def config(reg=5,channel=0):
    return ('dma_config',f'channel = {channel} : i32, base_reg = {reg} : i32')
def transfer(direction='load',channel=0,reg=6,dram=1,size=2):
    return ('dma',f'direction = "{direction}", channel = {channel} : i32, reg = {reg} : i32, dram = {dram} : i32, size = {size} : i32')
def wait(channel=0): return ('dma_wait',f'channel = {channel} : i32')
def setup(vmem=0,offset=0x90000000,size=128,base=0):
    return [*word_base(5,base),config(),*word_base(6,vmem),*word_base(1,offset),*word_base(2,size)]

@unittest.skipUnless(REPORT and DMA_REPORT and OPT.is_file() and EMIT.is_file(),
                     'requires explicitly selected VLS/DMA receipts and built tools')
class SelectedDMATimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(REPORT).resolve();value=json.loads(path.read_text())
        dma=Path(DMA_REPORT).resolve()
        cls.options={'evidence':str(path),'evidence-sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'manifest-sha256':value['inputs']['manifest']['sha256'],
                     'hardware-ir-sha256':value['inputs']['hardware_ir']['sha256'],
                     'allow-conditional':'true','dma-evidence':str(dma),
                     'dma-evidence-sha256':hashlib.sha256(dma.read_bytes()).hexdigest()}
        cls.receipt=json.loads(dma.read_text())
    def selected(self,ops,*passes,**overrides):
        options={**self.options,**overrides}
        for key in [key for key,value in options.items() if value is None]: del options[key]
        text=ops if isinstance(ops,str) else program(ops)
        selection='--select-atlas-rtl-evidence='+' '.join(f'{k}={v}' for k,v in options.items())
        return run(OPT,text,selection,*passes)
    def check_consumers(self,ops,accepted):
        for consumer in CONSUMERS:
            with self.subTest(consumer=consumer):
                checked=self.selected(ops,consumer,'--verify-atlas-rtl-timing')
                self.assertEqual(checked.returncode==0,accepted,checked.stderr)
                if accepted:
                    self.assertEqual(run(EMIT,checked.stdout).returncode,0,checked.stderr)
        return checked
    def test_both_consumers_and_final_export(self):
        ops=[*setup(),transfer(),wait(),HALT]
        for consumer in CONSUMERS:
            checked=self.selected(ops,consumer,'--verify-atlas-rtl-timing')
            self.assertEqual(checked.returncode,0,checked.stderr)
            self.assertIn('atlas.vls_dma.serialized.v1',checked.stdout)
            export=run(EMIT,checked.stdout,'--rtl-timing-json')
            self.assertEqual(export.returncode,0,export.stderr)
            data=json.loads(export.stdout)
            self.assertEqual(data['schema'],'atlas.resolved_rtl_timing.v1')
            self.assertFalse(data['scheduling_qualified'])
            self.assertEqual(data['qualification'],'conditional')
            self.assertEqual(data['resolver']['id'],'atlas.vls_dma.serialized.v1')
            dma=[i for i in data['instructions'] if i['mnemonic'].startswith('dma.')]
            launch=next(i for i in dma if i['mnemonic']=='dma.load.ch0')
            retired=next(i for i in dma if i['mnemonic']=='dma.wait.ch0')
            self.assertIsNone(launch['footprint']['done_age'])
            self.assertIsNone(launch['footprint']['dma_cycles'])
            memory=[a for a in launch['footprint']['accesses'] if a['resource']=='vmem']
            self.assertTrue(memory)
            for access in memory:
                self.assertTrue(access['at_completion'])
                self.assertIsNone(access['age']);self.assertIsNone(access['step'])
            self.assertGreater(retired['issue_epoch'],launch['issue_epoch'])
            self.assertEqual(retired['epoch_offset'],0)
            self.assertNotIn('logical_issue_cycle',launch)
            self.assertIn('minimum_issue_cycle',launch)
    def test_config_is_synchronous_not_transfer(self):
        self.check_consumers([addi(5,0,31),config(),HALT],True)
    def test_pending_scalar_and_global_base_clobber_allowed(self):
        self.check_consumers([*setup(),transfer(),*word_base(1,0x90001000),addi(6,0,32),
                              addi(2,0,32),addi(5,0,1),config(),wait(),
                              addi(5,0,0),config(),HALT],True)
    def test_channels_and_waited_reuse(self):
        for channel in range(8):
            with self.subTest(channel=channel):
                self.check_consumers([*setup(),transfer(channel=channel),wait(channel),
                                      transfer('store',channel),wait(channel),HALT],True)
    def test_pending_is_not_released_by_config_scalar_or_empty_wrong_wait(self):
        prefix=[*setup(),transfer()]
        for suffix in ([wait(1),HALT],[transfer(channel=1),wait(1),HALT],
                       [addi(5,0,1),config(),VLOAD,wait(),HALT],
                       [MARKER,wait(),HALT],[HALT],
                       [('delay','cycles = 100 : i32'),HALT]):
            with self.subTest(suffix=suffix): self.check_consumers([*prefix,*suffix],False)
        for ops in ([wait(),HALT],[*setup(),transfer(),wait(),wait(),HALT]):
            self.check_consumers(ops,False)
    def test_operand_domain_boundaries_and_unknowns(self):
        for vmem,offset,size,base in [(0,0,32,0),(392192,0xffffffe0,32,31),
                                    (392192,0x90000000,4096,31)]:
            self.check_consumers([*setup(vmem,offset,size,base),transfer(),wait(),HALT],True)
        for vmem,offset,size,base in [(1,0,32,0),(393216,0,32,0),(65528,0,64,0),
                                    (0,1,32,0),(0,0xffffffe0,64,0),(0,0,0,0),
                                    (0,0,31,0),(0,0,33,0),(0,0,4128,0),(0,0,32,32)]:
            with self.subTest(vmem=vmem,offset=offset,size=size,base=base):
                self.check_consumers([*setup(vmem,offset,size,base),transfer(),wait(),HALT],False)
        self.check_consumers([addi(6,0,0),transfer(),wait(),HALT],False)
    def test_completion_barrier_cannot_be_stripped_before_final_emission(self):
        selected=self.selected([*setup(),transfer(),wait(),HALT],'--verify-atlas-rtl-timing')
        self.assertEqual(selected.returncode,0,selected.stderr)
        source=selected.stdout
        lines=source.splitlines();alias={};kept=[]
        import re
        for line in lines:
            matched=re.search(r'(%\w+) = "atlas.dma_wait"\((%\w+)\)',line)
            if matched: alias[matched[1]]=matched[2];continue
            for old,new in alias.items(): line=line.replace('('+old+')','('+new+')')
            kept.append(line)
        unsafe='\n'.join(kept)
        self.assertNotEqual(run(OPT,unsafe,'--verify-atlas-rtl-timing').returncode,0)
        self.assertNotEqual(run(EMIT,unsafe).returncode,0)
    def test_llvm_entrypoints_and_structured_finalization_recheck_completion(self):
        import re
        selected=self.selected([*setup(),transfer(),wait(),HALT],'--verify-atlas-rtl-timing')
        self.assertEqual(selected.returncode,0,selected.stderr)
        # Remove the terminal wait while reconnecting the physical state chain.
        unsafe=re.sub(r'(%\w+) = "atlas.dma_wait"\((%\w+)\)([^\n]*)',
                      lambda m: m.group(1)+' = "atlas.alu_imm"('+m.group(2)+') '
                      '<{dst = 0 : i32, immediate = 0 : i32, kind = "addi", src = 0 : i32}> '
                      ': (!atlas.state) -> !atlas.state',selected.stdout)
        self.assertNotEqual(unsafe,selected.stdout)
        for entrypoint in ('--convert-atlas-to-llvm','--convert-atlas-to-llvm-calls'):
            with self.subTest(entrypoint=entrypoint):
                self.assertNotEqual(run(OPT,unsafe,entrypoint).returncode,0)
        staged=run(OPT,selected.stdout,'--convert-atlas-to-llvm-calls')
        self.assertEqual(staged.returncode,0,staged.stderr)
        positive=run(OPT,staged.stdout,'--finalize-atlas-llvm-calls')
        self.assertEqual(positive.returncode,0,positive.stderr)
        removed=0;lines=[];index=0
        for line in staged.stdout.splitlines():
            if 'atlas.source_op = "atlas.dma_wait"' in line:
                removed+=1;continue
            if 'atlas.word_index =' in line:
                line=re.sub(r'atlas.word_index = [0-9]+ : i32',
                            'atlas.word_index = '+str(index)+' : i32',line)
                index+=1
            lines.append(line)
        self.assertEqual(removed,1)
        mutated='\n'.join(lines)
        finalized=run(OPT,mutated,'--finalize-atlas-llvm-calls')
        self.assertNotEqual(finalized.returncode,0)
        self.assertIn('DMA',finalized.stderr)

    def test_vls_only_never_silently_enables_dma(self):
        selected=self.selected([*setup(),transfer(),wait(),HALT],'--insert-atlas-delays',
                               **{'dma-evidence':None,'dma-evidence-sha256':None})
        self.assertNotEqual(selected.returncode,0)
    def test_hash_and_copied_success_flags_rejected(self):
        source=[*setup(),transfer(),wait(),HALT]
        self.assertNotEqual(self.selected(source,'--verify-atlas-rtl-timing',
                            **{'dma-evidence-sha256':'0'*64}).returncode,0)
        with tempfile.TemporaryDirectory(prefix='atlas-dma-test-') as directory:
            receipt=Path(directory)/'receipt.json'
            for field,value in [('state','prepared'),('scheduling_qualified',True),
                                ('qualification','qualified')]:
                changed=copy.deepcopy(self.receipt);changed[field]=value
                receipt.write_text(json.dumps(changed))
                selected=self.selected(source,'--verify-atlas-rtl-timing',**{
                    'dma-evidence':str(receipt),'dma-evidence-sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()})
                self.assertNotEqual(selected.returncode,0,field)
    def test_rehashed_event_mutations_rejected(self):
        with tempfile.TemporaryDirectory(prefix='atlas-dma-event-test-') as directory:
            directory=Path(directory);receipt=directory/'receipt.json';events=directory/'events.json'
            case=next(c for c in self.receipt['cases'] if c['name']=='capture_after_launch')
            original=json.loads(Path(case['events']['path']).read_text())
            for kind,field,value in [('a','address',0),('d','source',63),('wait','channel',7),
                                     ('launch','size',32),('launch','dram_address',0x90001000),
                                     ('launch','vmem_word',8),('config','base',1),('a','source',63),
                                     ('d','edge',1)]:
                changed=copy.deepcopy(original)
                target=next(e for e in changed['events'] if e['kind']==kind)
                if kind=='a' and field=='source':
                    requests=[e for e in changed['events'] if e['kind']=='a']
                    target=requests[1];value=requests[0]['source']
                target[field]=value
                events.write_text(json.dumps(changed));report=copy.deepcopy(self.receipt)
                selected_case=next(c for c in report['cases'] if c['name']==case['name'])
                selected_case['events']={'path':str(events),'sha256':hashlib.sha256(events.read_bytes()).hexdigest(),'bytes':events.stat().st_size}
                receipt.write_text(json.dumps(report))
                result=self.selected([HALT],'--verify-atlas-rtl-timing',**{'dma-evidence':str(receipt),
                    'dma-evidence-sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()})
                self.assertNotEqual(result.returncode,0,(kind,field))

if __name__=='__main__': unittest.main()
