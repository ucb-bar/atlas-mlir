"""Pressure bank boundaries, optionally compared with the real allocator."""
from pathlib import Path
import json,os,subprocess,tempfile,unittest
from atlas_egglog_ir import parse_program
from atlas_egglog_cost import TimingCosts


def pressure_source(splats, mixed=False):
    lines=['module {','  func.func @pressure() -> !atlas.virtual_state attributes {atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64} {',
           '    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state']
    state='%s'
    if mixed:
        lines.append('    %io, %fp8 = "atlas.virtual_input_fp8"(%s) {index = 0 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)');state='%io'
    for i in range(splats):lines.append(f'    %v{i} = "atlas.virtual_vpu_splat"() {{bits = {16256+i} : i32}} : () -> !atlas.virtual_bf16')
    value='%v0'
    for i in range(1,splats):
        lines.append(f'    %r{i} = "atlas.virtual_vpu_binary"({value}, %v{i}) {{kind = "max"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16');value=f'%r{i}'
    lines += [f'    %end = "atlas.virtual_output_bf16"({state}, {value}) {{index = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state',
              '    return %end : !atlas.virtual_state','  }','}']
    return '\n'.join(lines)+'\n'


class PressureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        path=self.root/'timing.json'
        path.write_text(json.dumps(dict(op_timing=[dict(name=name,first_free_age=age,resolved=True) for name,age in {
            'vpu.vliAll':65,'vpu.pairmax':66,'vlsu.vload':35,'vlsu.vstore':35,'scalar_lsu.load':4}.items()])))
        self.cost=TimingCosts(path)

    def tearDown(self):self.temp.cleanup()

    def test_bf16_only_and_mixed_use_distinct_bank_capacities(self):
        for n,mixed,fits in ((17,False,True),(30,False,True),(31,False,False),(14,True,True),(15,True,False)):
            with self.subTest(n=n,mixed=mixed):
                cost=self.cost.estimate(parse_program(pressure_source(n,mixed)))
                self.assertEqual(dict(cost.peak_pressure)['bf16-pairs'],n+1)
                self.assertEqual('bf16-pairs' not in cost.exceeds_capacity,fits)
                self.assertNotIn('matrix-registers',dict(cost.peak_pressure))

    def test_fp8_reservation_does_not_consume_bf16_pairs(self):
        plain=self.cost.estimate(parse_program(pressure_source(14)))
        mixed=self.cost.estimate(parse_program(pressure_source(14,True)))
        self.assertEqual(dict(plain.peak_pressure)['bf16-pairs'],dict(mixed.peak_pressure)['bf16-pairs'])
        self.assertEqual(dict(mixed.peak_pressure)['fp8-registers'],1)
        self.assertFalse(mixed.exceeds_capacity)

    @unittest.skipUnless(os.environ.get('ATLAS_OPT'),'set ATLAS_OPT for the real allocation boundary gate')
    def test_real_allocator_agrees_at_bank_boundaries_without_rescheduling(self):
        for n,mixed,fits in ((17,False,True),(30,False,True),(31,False,False),(14,True,True),(15,True,False)):
            with self.subTest(n=n,mixed=mixed):
                source=pressure_source(n,mixed);report=self.root/f'alloc-{n}-{mixed}.json'
                result=subprocess.run([os.environ['ATLAS_OPT'],'-','--verify-atlas-virtual-stream',f'--lower-atlas-virtual-to-machine=allocation-report={report}'],
                                      input=source,text=True,capture_output=True,timeout=30)
                self.assertEqual(result.returncode==0,fits,result.stderr)
                allocation=json.loads(report.read_text())
                self.assertEqual(allocation['registers']['bf16']['capacity'],15 if mixed else 31)
                self.assertEqual(allocation['registers']['bf16']['peak_live'],n+1)
                self.assertEqual(allocation['fits'],fits)


if __name__=='__main__':unittest.main()
