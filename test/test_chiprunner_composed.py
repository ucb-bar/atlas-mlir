"""Runner protocol tests, distinct from model/ELF integration qualification."""
import hashlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch
from atlas_mlir.chiprunner import composed


def inventory(root):
    names=['manifest.json','device.fs','host.elf','input.bin','weight.bin','a.bin','b.bin']
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256((root/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))


def fixture(root):
    for name,data in {'device.fs':b'protocol stub','host.elf':b'protocol stub','input.bin':b'\x02','weight.bin':b'\x03','a.bin':b'\x06','b.bin':b'\x07'}.items():
        (root/name).write_bytes(data)
    address=0x200c0000
    regions=[dict(id='a',kind='device',program='device.fs',inputs=[dict(id='x',address=address,bytes=1,file='input.bin')],weights=[dict(id='w',address=address+1,bytes=1,file='weight.bin')],outputs=[dict(id='y',address=address+2,bytes=1,reference='a.bin')]),dict(id='b',kind='host',elf='host.elf',inputs=[dict(id='x',address=address,bytes=1,source=dict(region='a',output='y'))],outputs=[dict(id='y',address=address+2,bytes=1,reference='b.bin')])]
    (root/'manifest.json').write_text(json.dumps(dict(composed=dict(regions=regions))))
    raw=bytearray(184); raw[:6]=b'\x7fELF\x02\x01'; struct.pack_into('<H',raw,18,243); struct.pack_into('<QQ',raw,24,0x08000000,64); struct.pack_into('<HH',raw,54,56,1); struct.pack_into('<IIQQQQQQ',raw,64,1,5,120,0x08000000,0x08000000,64,64,8)
    (root/'host.elf').write_bytes(raw)
    manifest=json.loads((root/'manifest.json').read_text()); manifest['composed']['regions'][1].update(entry=0x08000000,status=dict(address=0x0800f000,done_value=1))
    (root/'manifest.json').write_text(json.dumps(manifest)); inventory(root)


def elf(entry, segments):
    """ELF64 RISC-V image with one PT_LOAD per (address, flags, data) segment."""
    table=64+56*len(segments); raw=bytearray(table); raw[:6]=b'\x7fELF\x02\x01'
    struct.pack_into('<H',raw,18,243); struct.pack_into('<QQ',raw,24,entry,64); struct.pack_into('<HH',raw,54,56,len(segments))
    for i,(address,flags,data) in enumerate(segments):
        struct.pack_into('<IIQQQQQQ',raw,64+56*i,1,flags,len(raw),address,address,len(data),len(data),8); raw+=data
    return bytes(raw)


class MemoryTransport:
    instances=[]
    def __init__(self,*args): self.mem={}; self.closed=False; self.seen=[]; self.instances.append(self)
    def write(self,address,data):
        for i,b in enumerate(data): self.mem[address+i]=b
    def read(self,address,size): return bytes(self.mem[address+i] for i in range(size))
    def launch(self,region):
        self.seen.append(self.read(region['inputs'][0]['address'],1))
        output=self.seen[-1][0]*(self.mem[0x200c0001] if region['kind']=='device' else 1)+(region['kind']=='host')
        for row in region['outputs']: self.write(row['address'],bytes([output&0xff]))
    def wait(self,*args): pass
    def close(self): self.closed=True


class ComposedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'bundle'; self.root.mkdir(); fixture(self.root)
        self.logs=Path(self.tmp.name)/'logs'; MemoryTransport.instances=[]
    def run_plan(self,**kw): return composed.execute(self.root,self.logs,backend='protocol-test',transport_factory=MemoryTransport,**kw)
    def edit(self,func):
        p=self.root/'manifest.json'; m=json.loads(p.read_text()); func(m['composed']['regions']); p.write_text(json.dumps(m)); inventory(self.root)
    def test_actual_producer_bytes_poison_and_hashes(self):
        journal=self.run_plan(); instance=MemoryTransport.instances[0]
        self.assertEqual(instance.seen,[b'\x02',b'\x06']); self.assertTrue(instance.closed)
        self.assertEqual(len(journal['completed']),2)
        self.assertEqual(journal['completed'][0]['outputs']['y']['before'],composed.sha(b'\xa5'))
        self.assertEqual(journal['completed'][1]['inputs']['inputs:x'],composed.sha(b'\x06'))
        with patch.object(composed,'load_plan',wraps=composed.load_plan):
            self.run_plan(resume=True)
        self.assertEqual(len(MemoryTransport.instances),1)
    def test_resume_verified_prefix_reloads_actual_outputs(self):
        original=composed.atomic_json
        def stop(path,value):
            original(path,value)
            if len(value['completed'])==1 and value['active'] is None: raise RuntimeError('interrupted between regions')
        with patch.object(composed,'atomic_json',side_effect=stop):
            with self.assertRaises(RuntimeError): self.run_plan()
        self.run_plan(resume=True)
        self.assertEqual(MemoryTransport.instances[1].seen,[b'\x06'])
    def test_mutation_fails_boundary_and_incomplete_resume(self):
        (self.root/'input.bin').write_bytes(b'\x04'); inventory(self.root)
        with self.assertRaisesRegex(ValueError,'boundary mismatch'): self.run_plan()
        with self.assertRaisesRegex(ValueError,'incomplete checkpoint'): self.run_plan(resume=True)
    def test_stale_and_corrupted_checkpoints_fail(self):
        self.run_plan(); (self.logs/'a-output-0.bin').write_bytes(b'\xff')
        with self.assertRaisesRegex(ValueError,'checkpoint byte drift'): self.run_plan(resume=True)
        (self.logs/'a-output-0.bin').write_bytes(b'\x06')
        (self.root/'weight.bin').write_bytes(b'\x04'); inventory(self.root)
        with self.assertRaisesRegex(ValueError,'stale'): self.run_plan(resume=True)
    def test_missing_producer_overlaps_and_inventory_drift_fail_before_transport(self):
        self.edit(lambda rs:rs.reverse())
        with self.assertRaisesRegex(ValueError,'producer'): self.run_plan()
        fixture(self.root); self.edit(lambda rs:rs[0]['outputs'][0].update(address=0x200c0000))
        with self.assertRaisesRegex(ValueError,'overlapping'): self.run_plan()
        fixture(self.root); (self.root/'weight.bin').write_bytes(b'\x04')
        with self.assertRaisesRegex(ValueError,'inventory drift'): self.run_plan()
        self.assertEqual(MemoryTransport.instances,[])
    def test_host_buffers_cannot_overlap_executable_code(self):
        for role in ('inputs','outputs'):
            with self.subTest(role=role):
                fixture(self.root); self.edit(lambda rs:rs[1][role][0].update(address=0x08000020))
                with self.assertRaisesRegex(ValueError,'buffer overlaps executable code'): self.run_plan()
        self.assertEqual(MemoryTransport.instances,[])
    def test_failing_boundary_keeps_every_output_and_promotes_nothing(self):
        self.edit(lambda rs:rs[0]['outputs'].append(dict(id='z',address=0x200c0003,bytes=1,reference='a.bin')))
        (self.root/'input.bin').write_bytes(b'\x04'); inventory(self.root)
        with self.assertRaisesRegex(composed.BoundaryError,'a/y; boundary mismatch: a/z'): self.run_plan()
        self.assertEqual([(self.logs/f'a-output-{i}.bin').read_bytes() for i in (0,1)],[b'\x0c',b'\x0c'])
        journal=json.loads((self.logs/'journal.json').read_text())
        self.assertEqual((journal['completed'],journal['active']),([],'a'))
        self.assertEqual(set(journal['rejected']['outputs']),{'y','z'})
        self.assertEqual(len(MemoryTransport.instances[0].seen),1)
    def partial_poison(self, short):
        # Region a: y mismatches the reference after mutation; z stays poisoned or reads short.
        self.edit(lambda rs:rs[0]['outputs'].append(dict(id='z',address=0x200c0003,bytes=1,reference='a.bin')))
        class Partial(MemoryTransport):
            def launch(self,region):
                if region['id']!='a' or self.read(0x200c0000,1)==b'\x02': return super().launch(region)
                self.write(0x200c0002,b'\x09')
            def read(self,address,size):
                return b'' if short and address==0x200c0003 and 0x200c0002 in self.mem and self.mem[0x200c0002]==9 else super().read(address,size)
        return Partial
    def test_poisoned_or_short_output_is_never_a_numerical_mismatch(self):
        for short,message in ((False,'output remained poisoned: a/z'),(True,'short output read: a/z')):
            with self.subTest(short=short):
                fixture(self.root); transport=self.partial_poison(short)
                (self.root/'input.bin').write_bytes(b'\x04'); inventory(self.root)
                logs=Path(self.tmp.name)/f'mixed-{short}'
                with self.assertRaisesRegex(ValueError,'boundary mismatch: a/y; '+message) as caught:
                    composed.execute(self.root,logs,backend='protocol-test',transport_factory=transport)
                self.assertNotIsInstance(caught.exception,composed.BoundaryError)
    def test_qualification_rejects_poisoned_or_short_negative(self):
        from atlas_mlir.chiprunner import step_inputs
        for short,message in ((False,'output remained poisoned: a/z'),(True,'short output read: a/z')):
            with self.subTest(short=short):
                fixture(self.root); transport=self.partial_poison(short)
                logs=Path(self.tmp.name)/f'qualify-{short}'
                with self.assertRaisesRegex(ValueError,message) as caught:
                    step_inputs.qualify(self.root,logs,backend='protocol-test',transport_factory=transport)
                self.assertNotIsInstance(caught.exception,composed.BoundaryError)
                self.assertTrue((logs/'positive/journal.json').exists())
    def test_elf_with_one_overwritten_segment_is_reloaded_every_call(self):
        # h1's data segment shares an address with h2's output; its code segment shares nothing.
        (self.root/'h1.elf').write_bytes(elf(0x08000000,[(0x08000000,5,b'\x13'*64),(0x08002000,6,b'\x05'*16)]))
        (self.root/'h2.elf').write_bytes(elf(0x08004000,[(0x08004000,5,b'\x13'*64)]))
        for name,data in {'y.bin':b'\x07','z.bin':b'\x08'}.items(): (self.root/name).write_bytes(data)
        regions=[dict(id='h1',kind='host',elf='h1.elf',entry=0x08000000,status=dict(address=0x0800f000,done_value=1),
                      inputs=[dict(id='x',address=0x200c0000,bytes=1,file='input.bin')],
                      outputs=[dict(id='y',address=0x200c0002,bytes=1,reference='y.bin')]),
                 dict(id='h2',kind='host',elf='h2.elf',entry=0x08004000,status=dict(address=0x0800f008,done_value=1),
                      inputs=[dict(id='x',address=0x200c0004,bytes=1,source=dict(region='h1',output='y'))],
                      outputs=[dict(id='z',address=0x08002000,bytes=1,reference='z.bin')])]
        (self.root/'manifest.json').write_text(json.dumps(dict(composed=dict(regions=regions))))
        names=['manifest.json','h1.elf','h2.elf','input.bin','y.bin','z.bin']
        (self.root/'SHA256SUMS').write_text(''.join(hashlib.sha256((self.root/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in names))
        class Loader(MemoryTransport):
            root=self.root
            def prepare(self,region):
                from atlas_mlir.chiprunner.composed_sim import _elf_loads
                self.prepared.append(region['id'])
                for address,data,n,flags in _elf_loads((self.root/region['elf']).read_bytes())[1]: self.write(address,data)
            def launch(self,region):
                x=self.read(region['inputs'][0]['address'],1)[0]
                y=x+self.mem[0x08002000] if region['id']=='h1' else x+1
                self.write(region['outputs'][0]['address'],bytes([y&0xff]))
        Loader.prepared=[]
        session=composed.Session(self.root,self.logs,backend='protocol-test',transport_factory=Loader)
        self.assertNotIn(('h1','code',''),session.resident)
        self.assertIn(('h2','code',''),session.resident)
        try:
            for call in range(3): session.call(Path(self.tmp.name)/f'call-{call}')
        finally: session.close()
        self.assertEqual(Loader.prepared,['h2']+['h1']*3)
    def test_timeout_propagates_and_no_checkpoint_is_promoted(self):
        with patch.object(MemoryTransport,'wait',side_effect=TimeoutError('completion')):
            with self.assertRaises(TimeoutError): self.run_plan()
        journal=json.loads((self.logs/'journal.json').read_text())
        self.assertEqual(journal['completed'],[]); self.assertEqual(journal['active'],'a')
        self.assertTrue(MemoryTransport.instances[0].closed)

class PersistentTclTests(unittest.TestCase):
    def test_one_session_loads_and_verifies_before_staging(self):
        from atlas_mlir.chiprunner.composed_tcl import TclTransport
        class RPC:
            timeout=1
            broken=False
            def __init__(self): self.commands=[]; self.memory={}
            def command(self,text):
                import re
                self.commands.append(text)
                match=re.fullmatch(r'(load_image|verify_image|dump_image) \{([^}]+)\}(?: (0x[0-9a-f]+) (bin|[0-9]+))?',text)
                if match and match[3]:
                    op,path,address,tail=match.groups(); address=int(address,16)
                    if op=='load_image':
                        for i,b in enumerate(Path(path).read_bytes()): self.memory[address+i]=b
                    elif op=='dump_image': Path(path).write_bytes(bytes(self.memory[address+i] for i in range(int(tail))))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'app.elf').write_bytes(b'packaging stub')
            rpc=RPC(); transport=TclTransport(root,root,{},rpc=rpc)
            region=dict(kind='host',elf='app.elf',entry=0x08000000,status=dict(address=0x0800f000,done_value=1))
            for value in (b'first',b'second'):
                transport.prepare(region); transport.write(0x200c0000,value)
                self.assertEqual(transport.read(0x200c0000,len(value)),value)
                transport.launch(region); transport.write(0x0800f000,(1).to_bytes(8,'little'))
                transport.wait(region,1)
            transport.close()
            self.assertEqual(rpc.commands.count('reset halt'),1)
            self.assertEqual(rpc.commands.count('resume 0x8000000'),2)
            self.assertEqual(rpc.commands.count('load_image {'+str(root/'app.elf')+'}'),2)


if __name__=='__main__': unittest.main()
