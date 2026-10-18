#!/usr/bin/env python3
"""Meaningful fail-closed and causality tests, no held-out execution IO."""
import copy
import importlib.util
import json
import sys
import unittest
from predict import HERE, GEOMETRY, predict

PROFILE={'kind':'fixed','params':{'latency':43,'cycles_per_beat':1}}

class PredictorTests(unittest.TestCase):
    def setUp(self):
        self.program=json.loads((HERE/'testdata/mlp-physical-program.json').read_text())

    def edit(self, operation):
        return next(i for i in self.program['instructions'] if i['operation']=='atlas.'+operation)

    def reject(self, message):
        with self.assertRaisesRegex(ValueError,message):predict(self.program,PROFILE)

    def test_no_torch_or_model_arch_import(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('npu_model.hardware.core',sys.modules)

    def test_ebreak_is_not_drained_ecall(self):
        self.edit('trap')['fields']['kind']='ebreak';self.reject('trap kind')

    def test_unknown_control_value_is_rejected(self):
        self.edit('branch')['fields']['lhs']=31;self.reject('unknown scalar')

    def test_unknown_dma_size_is_rejected(self):
        self.edit('dma')['fields']['size']=31;self.reject('unknown scalar')

    def test_unknown_dma_address_is_rejected(self):
        self.edit('dma')['fields']['dram']=31;self.reject('unknown scalar')

    def test_unsupported_geometry_is_rejected(self):
        self.program['geometry']={**GEOMETRY,'rows':64};self.reject('unsupported geometry')

    def test_unsupported_revision_is_rejected(self):
        self.program['selected_rtl_revision']='other';self.reject('revision')

    def test_unsupported_unit_is_rejected(self):
        self.edit('mxu_matmul')['fields']['unit']=2;self.reject('MXU unit')

    def test_unsupported_fields_are_rejected(self):
        self.edit('mxu_matmul')['fields']['surprise']=1;self.reject('unsupported fields')

    def test_unsupported_scalar_kind_is_rejected(self):
        self.edit('scalar_load')['fields']['kind']='lb';self.reject('scalar_load kind')

    def test_dma_direction_is_validated(self):
        self.edit('dma')['fields']['direction']='other';self.reject('DMA direction')

    def test_dma_byte_count_alignment(self):
        dma=self.edit('dma');size=dma['fields']['size']
        for i in self.program['instructions'][:dma['word_index']]:
            if i['operation']=='atlas.alu_imm' and i['fields']['dst']==size:
                i['fields']['immediate']=1023
        self.reject('DMA transfer size')

    def test_dma_dram_alignment(self):
        dma=self.edit('dma');reg=dma['fields']['dram']
        for i in self.program['instructions'][:dma['word_index']]:
            if i['operation']=='atlas.upper' and i['fields']['dst']==reg:
                i['operation']='atlas.alu_imm'
                i['fields']={'dst':reg,'src':0,'kind':'addi','immediate':33}
        self.reject('unaligned DRAM')

    def test_vls_alignment(self):
        self.edit('vload')['fields']['offset']=1;self.reject('aligned 1KiB')

    def test_unknown_operation_is_rejected(self):
        self.edit('upper')['operation']='atlas.magic';self.reject('unsupported operation')

    def test_explicit_delays_change_compute_schedule(self):
        original=predict(self.program,PROFILE)
        # Final program delay is before output DMA. Its larger gap must move the
        # output command and final drain; a constant trace replay would miss it.
        final_delay=[i for i in self.program['instructions'] if i['operation']=='atlas.delay'][-1]
        final_delay['fields']['cycles']+=10
        changed=predict(self.program,PROFILE)
        self.assertEqual(changed['cycles']-original['cycles'],10)

    def test_memory_feedback_moves_later_dma_launch(self):
        fast=predict(self.program,{'kind':'fixed','params':{'latency':1,'cycles_per_beat':1}})
        slow=predict(self.program,{'kind':'fixed','params':{'latency':688,'cycles_per_beat':48}})
        fast_last=[i['absolute_cycle'] for i in fast['issued'] if i['operation']=='atlas.dma'][-1]
        slow_last=[i['absolute_cycle'] for i in slow['issued'] if i['operation']=='atlas.dma'][-1]
        self.assertGreater(slow_last,fast_last)

    def test_bank_omission_changes_traffic_timing(self):
        fast={'kind':'fixed','params':{'latency':1,'cycles_per_beat':1}}
        aware=predict(self.program,fast);omitted=predict(self.program,fast,use_banks=False)
        self.assertGreater(aware['bank_denials']['load_write'],0)
        self.assertNotEqual(aware['dma_beats'],omitted['dma_beats'])

if __name__=='__main__':unittest.main(verbosity=2)
