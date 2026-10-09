"""Portable mutations of the fixed DMA boundary decoder; no RTL build required."""
import copy
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('ee290_dma_replay',ROOT/'tools/replay-ee290-dma.py')
DMA=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DMA)
WORDS=[0x00000293,0x0002807f,0x00000313,0x900000b7,0x08000113,
       0x0020837b,0x900010b7,0x0200007f,0x900001b7,0x40018193,
       0x022311fb,0x0200107f,0x00100093,0xc1009073,0x00000073]

def fixture():
    samples=[]
    for edge in range(1,122):
        v={k:0 for k in DMA.DMA_SIGNALS};v.update(ar=1,dr=1)
        samples.append({'edge':edge,'time':3*edge,'values':v})
    def put(edge,**values): samples[edge-1]['values'].update(values)
    def instruction(edge,index,fire=1):
        word=WORDS[index];opcode=word&127
        cmd=(2 if word>>25 else 1) if opcode==0x7b else (4 if word>>25 else 3) if opcode==0x7f else 0
        put(edge,valid=1,fire=fire,pc=index,word=word,decoded=cmd,channel=(word>>12)&7)
    for index in range(7): instruction(index+2,index)
    put(3,rs1=0)
    put(7,launch=1,op=1,vmem=0,address=0x90000000,size=128,launch_channel=0)
    for edge in range(8,61): put(edge,busy0=1,engine0=1)
    for edge in range(9,61): instruction(edge,7,0);put(edge,stall=1)
    instruction(61,7)
    for beat in range(4):
        put(10+beat,av=1,ao=4,aa=0x90000000+32*beat,**{'as':beat})
        put(53+beat,dv=1,ds=beat,dd=DMA.pattern(0,32*beat),vw=1,vwg=1,vwb=0,vwa=beat,vwd=DMA.pattern(0,32*beat))
    for edge,index in [(62,8),(63,9),(64,10)]: instruction(edge,index)
    put(64,launch=1,op=2,vmem=0,address=0x90000400,size=128,launch_channel=1)
    for edge in range(65,117): instruction(edge,11,0);put(edge,stall=1,busy1=1,engine1=1)
    instruction(117,11)
    for beat in range(4):
        put(65+beat,vr=1,vrg=1,vrb=0,vra=beat)
        put(66+beat,vresp=1,vdata=DMA.pattern(0,32*beat))
        put(70+beat,av=1,ao=0,aa=0x90000400+32*beat,ad=DMA.pattern(0,32*beat),**{'as':4+beat})
        put(113+beat,dv=1,ds=4+beat)
    instruction(118,12);instruction(119,13)
    put(119,csr=1,csr_addr=0xc10,csr_op=1,csr_data=1)
    instruction(120,14,0);put(120,ecall=1,halt=1,marker=1)
    put(121,halt=1,marker=1)
    return samples

class DMAReplayDecoderTest(unittest.TestCase):
    def validate(self,samples): return DMA.validate_dma_edges(samples,WORDS,0,1)
    def test_complete_delayed_transfer_pair(self):
        r=self.validate(fixture())
        self.assertEqual([e['channel'] for e in r['events'] if e['kind']=='wait'],[0,1])
        self.assertEqual(len(r['memory_events']),12)
        self.assertEqual(len(r['wait_stalls']),2)
    def test_reject_mutations(self):
        mutations=[(2,'launch',1),(7,'address',0x90001000),(7,'size',32),(7,'vmem',8),
                   (10,'aa',0x190000000),(10,'ao',0),(11,'as',0),
                   (53,'ds',7),(53,'dd',1),(53,'vwd',1),(53,'vwa',1),
                   (66,'vdata',1),(70,'ad',1),(9,'busy0',0),(9,'engine0',0),
                   (9,'fire',1),(9,'channel',1),(119,'csr_data',2),
                   (120,'marker',0),(8,'word',0),(7,'size',None)]
        for edge,key,value in mutations:
            with self.subTest(edge=edge,key=key):
                samples=fixture();samples[edge-1]['values'][key]=value
                # engine0=0 alone preserves scalar bridge and is permitted.
                if key=='engine0': samples[edge-1]['values']['busy2']=1
                with self.assertRaises(DMA.OBS.ObservationError): self.validate(samples)
    def test_missing_terminal_rejected(self):
        with self.assertRaises(DMA.OBS.ObservationError): self.validate(fixture()[:119])
    def test_response_before_fixture_delay_rejected(self):
        samples=fixture();samples[20]['values'].update(dv=1,ds=0,dd=DMA.pattern(0,0))
        with self.assertRaises(DMA.OBS.ObservationError): self.validate(samples)
    def test_unknown_active_control_rejected(self):
        samples=fixture();samples[90]['values']['dr']=None
        with self.assertRaises(DMA.OBS.ObservationError): self.validate(samples)
    def test_phantom_request_after_terminal_rejected(self):
        samples=fixture();samples[-1]['values'].update(av=1,aa=0x90000000,ao=4)
        with self.assertRaises(DMA.OBS.ObservationError): self.validate(samples)
    def test_before_launch_control_changes_typed_program(self):
        cases=DMA.cases()
        self.assertNotEqual(cases['before_launch_control'][0],cases['capture_after_launch'][0])
        self.assertEqual(cases['before_launch_control'][1],1)
        self.assertIn('base_reg = 5',cases['capture_scalar_and_base_pending'][0])
    def test_path_guard_rejects_spelling_before_access(self):
        for value in ('/tmp/VLSI/no-file','/tmp/not-hammer-safe/no-file'):
            with self.assertRaises(ValueError): DMA.bootstrap(value)

if __name__=='__main__': unittest.main()
