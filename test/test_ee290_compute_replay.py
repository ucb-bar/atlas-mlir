"""Independent finite compute trace checks and mutations, without an RTL build.

The captured checks require ATLAS_EE290_COMPUTE_REPLAY_REPORT; no workspace
capture is selected implicitly. Portable arithmetic and guard tests always run.
"""
import copy
import importlib.util
import os
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('ee290_compute_replay',ROOT/'tools/replay-ee290-compute.py')
COMPUTE=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPUTE)
REPORT=os.environ.get('ATLAS_EE290_COMPUTE_REPLAY_REPORT')

class PortableComputeReplayTest(unittest.TestCase):
    def test_bf16_powers_exact_sign_exponent_reference(self):
        lhs=b''.join(x.to_bytes(2,'little') for x in [0x3f80,0x4000,0xbf80,0xc000]*4)
        rhs=b''.join(x.to_bytes(2,'little') for x in [0x4000,0x3f00,0xbf80,0x4000]*4)
        expected=b''.join(x.to_bytes(2,'little') for x in [0x4000,0x3f80,0x3f80,0xc080]*4)
        self.assertEqual(COMPUTE.bf16_power_product(lhs,rhs),expected)
    def test_bf16_reference_rejects_nonpowers_and_exceptions(self):
        normal=(0x3f80).to_bytes(2,'little')*16
        for code in (0,0x7f80,0x7fc0,0x3f81):
            with self.subTest(code=code),self.assertRaises(COMPUTE.OBS.ObservationError):
                COMPUTE.bf16_power_product(code.to_bytes(2,'little')*16,normal)
    def test_reference_covers_paired_halves(self):
        memory=COMPUTE.initial_memory('vmul')
        self.assertEqual(len(memory),8192)
        self.assertNotEqual(memory[:1024],memory[1024:2048])
        self.assertNotEqual(memory[:2048],memory[2048:4096])
    def test_restricted_path_rejected_before_access(self):
        for path in ('/tmp/VLSI/nonexistent','/tmp/HaMmEr/nonexistent'):
            with self.assertRaises(ValueError): COMPUTE.bootstrap(path)

@unittest.skipUnless(REPORT,'requires explicitly selected captured compute replay')
class CapturedComputeReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checker=COMPUTE.CHECK.Checker();path=COMPUTE.CHECK.allowed(Path(REPORT))
        checker.identity(path)
        report=COMPUTE.CHECK.strict_json(path.read_bytes())
        if report['schema']!='atlas.ee290_compute_replay.v0': raise ValueError('wrong selected capture schema')
        cls.captures={}
        for case in report['cases']:
            trace=checker.verify(case['trace'],path.parent);words=checker.verify(case['words'],path.parent)
            values=[int(w,16) for w in COMPUTE.CHECK.allowed(words['path']).read_text().split()]
            with COMPUTE.CHECK.allowed(trace['path']).open() as stream:
                samples=list(COMPUTE.COMPUTE_OBS.vcd_edges(stream,'TOP.AtlasCore'))
            cls.captures[case['name']]=(samples,values)
        checker.recheck()
    def validate(self,mode,samples=None,words=None):
        original,original_words=self.captures[mode]
        return COMPUTE.validate_compute_edges(original if samples is None else samples,
                                              original_words if words is None else words,mode)
    def mutate(self,mode,key,predicate,value):
        samples,words=self.captures[mode];samples=copy.deepcopy(samples)
        sample=next(s for s in samples if predicate(s['values']))
        sample['values'][key]=value(sample['values'][key]) if callable(value) else value
        with self.assertRaises(COMPUTE.OBS.ObservationError): self.validate(mode,samples,words)
    def test_all_three_captured_programs_and_full_drain(self):
        for mode in ('xlu','vmul','dma_xlu'):
            with self.subTest(mode=mode):
                result=self.validate(mode)
                self.assertEqual(result['full_memory_checked_bytes'],8192)
                self.assertGreaterEqual(result['terminal_drain_edges'],2)
                halt=next(e for e in result['endpoints'] if e['kind']=='halt')
                self.assertTrue(all(v==0 for v in halt['engine_state'].values()))
    def test_xlu_row_identity_response_and_transpose_mutations(self):
        for key,valid,value in [('xrid','xr',2),('xrrow','xr',lambda x:x^1),
                              ('xdata','xresp',lambda x:x^1),('xwdata','xw',lambda x:x^1),
                              ('xwid','xw',35),('xarid','xar',3)]:
            with self.subTest(key=key): self.mutate('xlu',key,lambda v:v[valid],value)
    def test_vmul_both_ports_halves_and_product_mutations(self):
        for key,predicate,value in [('prid0',lambda v:v['pr0'] and v['prid0']==1,0),
                                   ('prid1',lambda v:v['pr1'] and v['prid1']==3,2),
                                   ('prrow0',lambda v:v['pr0'],lambda x:x^1),
                                   ('pdata0',lambda v:v['presp0'],lambda x:x^1),
                                   ('pdata1',lambda v:v['presp1'],lambda x:x^1),
                                   ('pwid0',lambda v:v['pw0'] and v['pwid0']==5,4),
                                   ('pwdata0',lambda v:v['pw0'],lambda x:x^1),
                                   ('pw1',lambda v:v['pw0'],1),('pw0',lambda v:v['pw0'],None)]:
            with self.subTest(key=key): self.mutate('vmul',key,predicate,value)
    def test_selected_words_and_spurious_decoded_command_rejected(self):
        for mode in self.captures:
            samples,words=self.captures[mode];changed=words.copy();changed[0]^=1
            with self.subTest(mode=mode),self.assertRaises(COMPUTE.OBS.ObservationError): self.validate(mode,samples,changed)
        self.mutate('xlu','xcmd',lambda v:v['fire'] and (v['word']&127)==0x13,1)
    def test_dma_composition_response_wait_and_store_mutations(self):
        key=lambda name: COMPUTE.DMA_KEYS[name]
        for name,predicate,value in [('aa',lambda v:v[key('av')] and v[key('ar')],0x90001000),
                                     ('dd',lambda v:v[key('dv')] and v[key('dr')],lambda x:x^1),
                                     ('ds',lambda v:v[key('dv')] and v[key('dr')],63),
                                     ('ad',lambda v:v[key('av')] and v[key('ar')] and v[key('ao')]==0,lambda x:x^1),
                                     ('vwd',lambda v:v[key('vw')],lambda x:x^1),
                                     ('vdata',lambda v:v[key('vresp')],lambda x:x^1),
                                     ('busy0',lambda v:v[key('busy0')],0)]:
            with self.subTest(name=name): self.mutate('dma_xlu',key(name),predicate,value)
        self.mutate('dma_xlu','fire',lambda v:v['valid'] and not v['fire'] and (v['word']&127)==0x7f,1)
    def test_unexpected_late_writes_and_truncated_drain_rejected(self):
        for mode in self.captures:
            samples,words=self.captures[mode]
            result=self.validate(mode);halt=next(e['edge'] for e in result['endpoints'] if e['kind']=='halt')
            tail=[copy.deepcopy(s) for s in samples if s['edge']>halt]
            self.assertTrue(tail)
            changed=copy.deepcopy(samples);next(s for s in changed if s['edge']>halt)['values']['xw']=1
            with self.subTest(mode=mode,kind='latewrite'),self.assertRaises(COMPUTE.OBS.ObservationError): self.validate(mode,changed,words)
            with self.subTest(mode=mode,kind='truncate'),self.assertRaises(COMPUTE.OBS.ObservationError): self.validate(mode,[s for s in samples if s['edge']<=halt],words)
    def test_missing_vmul_last_row_and_early_release_rejected(self):
        samples,words=self.captures['vmul'];samples=copy.deepcopy(samples)
        last=max((s for s in samples if s['values']['pw0']),key=lambda s:s['edge'])
        last['values']['pw0']=0
        with self.assertRaises(COMPUTE.OBS.ObservationError): self.validate('vmul',samples,words)
    def test_unknown_active_control_rejected(self):
        self.mutate('vmul','presp1',lambda v:v['presp1'],None)

if __name__=='__main__': unittest.main()
