"""Packed-tensor device boundary layouts and admission on the original model."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np

from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.device_boundary import page_count, virtual_transition
from atlas_mlir_native.resident_graph import compose_fragments
from test_exponent_restoration import BIN, ROOT, compile_fixture

RANGES=((0,0),(0x3f00,0x4080),(0x8000,0x8000),(0xbf00,0xc080))


def packed(matrix, padding):
    rows,columns=matrix.shape;mc,nc=(rows+31)//32,(columns+31)//32
    result=np.full(mc*nc*1024,padding,dtype='<u2')
    for row in range(rows):
        for column in range(columns):
            at=((row//32)*nc+column//32)*1024+(column%32//16)*512+(row%32)*16+column%16
            result[at]=matrix[row,column]
    return result.tobytes()


def reference(matrix, kind, ranges, pad, canonical):
    values=matrix.copy()
    if canonical:values[values==0x8000]=0
    admitted=np.full(values.shape,not ranges,np.bool_)
    for low,high in ranges:admitted|=(values>=low)&(values<=high)
    rejected=not bool(admitted.all())
    values=np.where(admitted,values,pad).astype('<u2')
    if kind=='column_to_row':values=values.reshape(1,-1)
    elif kind=='broadcast_column':values=np.repeat(values,32,axis=1)
    return packed(values,pad),rejected


class DeviceBoundaryTest(unittest.TestCase):
    def test_invalid_layouts_ranges_and_aliases_fail_closed(self):
        args=dict(source_address=DRAM_BASE,destination_address=DRAM_BASE+8192,status_address=DRAM_BASE+12288,
                  rows=35,columns=1,admitted_code_ranges=RANGES,pad=0)
        for change in (dict(kind='unknown'),dict(kind='column_to_row',columns=2),dict(rows=0),
                       dict(source_address=DRAM_BASE+1),dict(destination_address=DRAM_BASE),
                       dict(status_address=DRAM_BASE+8192),dict(admitted_code_ranges=((0,65536),)),
                       dict(admitted_code_ranges=((0x3f80,0x4000),),pad=0)):
            with self.subTest(change=change),self.assertRaises(ValueError):virtual_transition(**{**args,**change})
        self.assertEqual(page_count(35,37),4)
        self.assertEqual(page_count(35,1,'column_to_row'),2)

    @unittest.skipUnless((BIN/'atlas-opt').is_file(),'Atlas compiler binaries unavailable')
    def test_each_layout_lowers_with_default_checks(self):
        for kind in ('identity','column_to_row','broadcast_column'):
            with self.subTest(kind=kind):
                source=virtual_transition(source_address=DRAM_BASE,destination_address=DRAM_BASE+0x10000,
                    status_address=DRAM_BASE+0x18000,rows=35,columns=37 if kind=='identity' else 1,
                    kind=kind,admitted_code_ranges=RANGES,canonicalize_negative_zero=True)
                _,program=compile_fixture(source)
                self.assertTrue(program['instructions'])
                self.assertIn('atlas.exponent_scratch = "status"',source)

    @unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','select pinned npu_model explicitly')
    def test_layout_admission_zero_canonicalization_padding_and_guard_bytes(self):
        groups=[]
        source_address=DRAM_BASE+2048;destination_address=DRAM_BASE+0x10000;status=DRAM_BASE+0x18000
        for kind in ('identity','column_to_row','broadcast_column'):
            rows,columns=35,37 if kind=='identity' else 1
            for canonical,ranges,pad in ((False,RANGES,0),(True,tuple(pair for pair in RANGES if pair!=(0x8000,0x8000)),0),(True,((0x3f80,0x4000),),0x3f80)):
                virtual=virtual_transition(source_address=source_address,destination_address=destination_address,status_address=status,
                    rows=rows,columns=columns,kind=kind,admitted_code_ranges=ranges,pad=pad,
                    canonicalize_negative_zero=canonical)
                _,program=compile_fixture(virtual)
                binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
                cases=[]
                for bad in (False,True):
                    values=np.resize(np.array([0x3f80,0x8000,0x4000,0xbf80,0],'<u2'),(rows,columns))
                    if pad:values.fill(0x3f80);values[0,0]=0x4000
                    if bad:values[0,0]=0x7f80
                    # Nonlogical producer lanes are deliberately nonfinite.
                    payload=packed(values,0x7fc1)
                    wanted,rejected=reference(values,kind,ranges,pad,canonical)
                    memory=bytearray([0xa5])*(status-DRAM_BASE+4096)
                    at=source_address-DRAM_BASE;memory[at:at+len(payload)]=payload
                    expected=bytearray(memory);at=destination_address-DRAM_BASE
                    expected[at:at+len(wanted)]=wanted
                    at=status-DRAM_BASE;expected[at:at+2048]=int(rejected).to_bytes(4,'little')+bytes(2044)
                    cases.append(dict(label=f'canonical={canonical},pad={pad},bad={bad}',memory=memory.hex(),expected_memory=expected.hex()))
                groups.append(dict(kind=kind,binary=binary.hex(),cases=cases))
        # A later admitted transition must retain an earlier rejection status.
        rows=3;values=np.array([[0x7f80],[0x8000],[0x3f80]],'<u2')
        middle=DRAM_BASE+0x10000;last=DRAM_BASE+0x12000
        fragments=[virtual_transition(source_address=source_address,destination_address=middle,status_address=status,
                        rows=rows,columns=1,kind='broadcast_column',admitted_code_ranges=RANGES,canonicalize_negative_zero=True),
                   virtual_transition(source_address=middle,destination_address=last,status_address=status,
                        rows=rows,columns=32,admitted_code_ranges=RANGES,initialize_status=False)]
        joined=compose_fragments(fragments,tools=BIN,boundaries_lowered=True)
        _,program=compile_fixture(joined)
        first,rejected=reference(values,'broadcast_column',RANGES,0,True)
        memory=bytearray([0xa5])*(status-DRAM_BASE+4096)
        memory[source_address-DRAM_BASE:source_address-DRAM_BASE+2048]=packed(values,0x7fc1)
        expected=bytearray(memory)
        for address in (middle,last):expected[address-DRAM_BASE:address-DRAM_BASE+2048]=first
        at=status-DRAM_BASE;expected[at:at+2048]=int(rejected).to_bytes(4,'little')+bytes(2044)
        binary=b''.join(row['word_u32'].to_bytes(4,'little') for row in program['instructions'])
        groups.append(dict(kind='composed_status_carry',binary=binary.hex(),cases=[dict(label='first_bad_then_good',memory=memory.hex(),expected_memory=expected.hex())]))
        with tempfile.TemporaryDirectory(prefix='atlas-device-boundary-') as directory:
            directory=Path(directory);spec=directory/'spec.json';receipt=directory/'receipt.json'
            spec.write_text(json.dumps(groups))
            process=subprocess.run([os.environ['IMPORT_MODEL_PYTHON'],str(Path(__file__).with_name('device_boundary_model.py')),
                str(spec),str(receipt),str(ROOT)],capture_output=True,text=True,timeout=180)
            self.assertEqual(process.returncode,0,process.stdout+process.stderr)
            output=os.environ.get('ATLAS_BOUNDARY_RECEIPT')
            if output:Path(output).write_bytes(receipt.read_bytes())


if __name__=='__main__':
    unittest.main()
