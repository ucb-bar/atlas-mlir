"""Real extraction diversity and shared-term/seed compatibility checks."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from atlas_egglog_ir import parse_program
from atlas_egglog_facts import node_digest
from atlas_egglog_cost import TimingCosts
from explore_atlas_egglog import search


def program(scales):
    lines=['module {','  func.func @recipes() -> !atlas.virtual_state {',
           '    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state',
           '    %x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16']
    state='%s'
    for index,scale in enumerate(scales):
        lines += [f'    %p{index} = "atlas.virtual_pack_fp8"(%x) {{scale_code = {scale} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8',
                  f'    %l{index}, %a{index} = "atlas.virtual_mxu_load_acc_fp8"({state}, %p{index}) {{unit = 1 : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)',
                  f'    %r{index}, %b{index} = "atlas.virtual_mxu_readout_bf16"(%l{index}, %a{index}) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)',
                  f'    %o{index} = "atlas.virtual_output_bf16"(%r{index}, %b{index}) {{index = {index} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state']
        state=f'%o{index}'
    return parse_program('\n'.join(lines+[f'    return {state} : !atlas.virtual_state','  }','}'])+'\n')


class RecipeTieTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        path=Path(self.temp.name)/'timing.json'
        costs={'vpu.vliAll':65,'vpu.pairmax':66,'vpu.pairmin':66,'vpu.fp8pack':70,
               'vlsu.vstore':35,'vlsu.vload':35,'scalar_lsu.load':4,'scalar_lsu.store':4,
               'scalar_lsu.scale_load':2}
        for unit in (0,1):
            for op in ('push_acc_bf16','push_acc_fp8','pop_acc_fp8','pop_acc_bf16'):
                costs[f'mxu{unit}.{op}']=33
        path.write_text(json.dumps({'op_timing':[dict(name=name,first_free_age=value,resolved=True) for name,value in costs.items()]}))
        self.timing=TimingCosts(path)

    def tearDown(self):self.temp.cleanup()

    def generate(self,scales):
        p=program(scales)
        chosen,report=search(p,self.timing,attempts=6,limit=9,seed=194)
        rows={name:row for name,row in report['details'].items() if name.startswith('egg-')}
        self.assertEqual(len(rows),6)
        self.assertFalse(report['rejected'])
        self.assertEqual(report['stats']['stop_reason'],'fixed_point')
        for candidate in chosen:parse_program(candidate.source.decode())
        return p,rows,report

    def test_same_producer_different_scales_extract_mixed_implementations(self):
        _,rows,report=self.generate((122,123,124,125))
        assignments={tuple(c['unit'] for c in row['choices']) for row in rows.values()}
        self.assertGreater(len(assignments),2)
        self.assertTrue(any(len(set(assignment))>1 for assignment in assignments))
        self.assertEqual(list(report['tie_breaking']['distinct_recipes_per_producer'].values()),[4])

    def test_identical_pack_expressions_still_share_the_same_choice(self):
        _,rows,_=self.generate((122,123,122,123))
        for row in rows.values():
            choices={c['result']:c['unit'] for c in row['choices']}
            self.assertEqual(choices['%p0'],choices['%p2'])
            self.assertEqual(choices['%p1'],choices['%p3'])

    def test_single_recipe_producer_retains_existing_seeded_preferences(self):
        p,rows,report=self.generate((127,127))
        source_key=node_digest(p.nodes['%x'])
        self.assertEqual(report['tie_breaking']['distinct_recipes_per_producer'][source_key],1)
        for index in range(6):
            expected=index if index<2 else hashlib.sha256((source_key+':'+str(194+index)).encode()).digest()[0]&1
            self.assertEqual([c['unit'] for c in rows[f'egg-{index:02}']['choices']],[expected,expected])


if __name__=='__main__':unittest.main()
