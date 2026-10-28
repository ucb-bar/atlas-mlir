"""Complete captured phase coverage and immutable live-inventory controls."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).absolute().parents[1]/'tools'),
              str(Path(__file__).absolute().parents[1]/'tools/host_iss')]
import smol_full_phases as phases
from smol_full_plan import calls, panels
from elf_image import Image


class PhaseTests(unittest.TestCase):
    def test_complete_plan_and_folded_boundary_geometry(self):
        ranges=phases.segments()
        self.assertEqual(len(ranges),200)
        self.assertEqual([sum(x['kind']==k for x in ranges) for k in ('vision','prefix','expert')],[24,16,160])
        self.assertEqual([i for r in ranges for i in range(r['launch_start'],r['launch_end'])],list(range(2715)))
        self.assertEqual([i for r in ranges for i in range(r['call_start'],r['call_end'])],list(range(1421)))
        self.assertEqual((ranges[0]['launch_start'],ranges[0]['launch_end']),(0,56))
        self.assertEqual((ranges[23]['launch_end'],ranges[24]['launch_start']),(1170,1170))
        self.assertEqual((ranges[-1]['launch_start'],ranges[-1]['launch_end']),(2705,2715))
        for r in ranges:
            actual=[p['index'] for p in panels() if r['call_start']<=p['call']<r['call_end']]
            self.assertEqual(actual,list(range(r['launch_start'],r['launch_end'])))

    def test_missing_tail_and_reordered_call_fail(self):
        changed=calls()[:-1]
        with patch.object(phases,'calls',return_value=changed),self.assertRaises(ValueError): phases.segments()
        changed=calls();changed[7],changed[13]=changed[13],changed[7]
        with patch.object(phases,'calls',return_value=changed),self.assertRaises(ValueError): phases.segments()

    def test_current_observations_and_canonical_history(self):
        values=[0]*2715
        values[56:104]=[7]*48
        self.assertEqual(phases.check_observations(values,56,104),[7]*48)
        for index,value in ((0,1),(104,1),(56,0),(56,True),(56,-1)):
            changed=values.copy();changed[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):
                phases.check_observations(changed,56,104)

    @unittest.skipUnless(os.environ.get('SMOL_PHASE_BUNDLE'),'actual baseline ELF fixture required')
    def test_actual_full_elf_live_buffers_and_omission_controls(self):
        root=Path(os.environ['SMOL_PHASE_BUNDLE'])
        data=json.loads((root/'host/smol-full-contract.json').read_text())
        image=Image((root/'host/host.elf').read_bytes())
        image.symbols['smol_full_phase_state']=dict(size=40,address=0,section=0)
        live=phases.live_regions(data,image)
        self.assertEqual(len(live),56)
        self.assertEqual(live['smol_full_actual_prefix_key_cache'],1812480)
        self.assertIn('smol_b_output',live)
        scratch=phases.scratch_regions(image,live)
        self.assertNotIn('smol_b_output',scratch)
        self.assertIn('smol_g_positions',scratch)
        self.assertIn('smol_full_input_copy',scratch)
        for mutate in ('reference','size','symbol'):
            wrong=copy.deepcopy(data);symbols=copy.deepcopy(image.symbols)
            if mutate=='reference': wrong['reference_buffers'].pop()
            elif mutate=='size': wrong['reference_buffers'][0]['bytes']+=1
            else: symbols.pop('smol_b_output')
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                phases.live_regions(wrong,SimpleNamespace(symbols=symbols))


if __name__=='__main__': unittest.main()
