"""Gate controls plus opt-in native regressions for numerical observability."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

TOOLS=Path(__file__).absolute().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
spec=importlib.util.spec_from_file_location('overlap_validation_under_test',TOOLS/'atlas_validate_overlap.py')
v=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=v
spec.loader.exec_module(v)


def fixture(minimum=1):
    facts=dict(model={'source_sha256':'a'*64},op_timing=[{'name':'vpu.relu','first_free_age':66}])
    raw=json.dumps(facts).encode()
    candidate=dict(schema='atlas.model_overlap_derived.v1',age_convention='issue_edge_zero',model=facts['model'],per_op_sha256=v.digest(raw),derivation={'contract':'source-resource-occupancy-v1'},
                   pairs=[dict(first='vpu.relu',second='vpu.relu',relation={'mreg_banks':'disjoint'},minimum=minimum,high=66)])
    sample=dict(id='case0',mechanism='issue-slot',engine='vpu',operations=['vpu.relu']*2,assembly=['one','two'],gaps=[minimum],relation={'mreg_banks':'disjoint'},expectation='positive',category='pair',numerical_equivalent=True,minimum=minimum,high=66)
    return facts,raw,candidate,sample


class NativeValidationTests(unittest.TestCase):
    def test_invalid_one_short_zero_is_not_native_rejection(self):
        _,_,_,sample=fixture()
        def native(*args,**kwargs):
            return dict(initial_sha256='a',final_sha256='b')
        with patch.object(v,'check_sample_assembly'),patch.object(v,'native_run',side_effect=native):
            result=v.run_sample(sample)
        zeros=[r for r in result['runs'] if r['variant']=='one_short']
        self.assertEqual([r['outcome'] for r in zeros],['zero_unencodable']*2)
        self.assertEqual(v.summarize_runs(result['runs'])['vpu']['negative_rejections'],0)

    def test_passing_one_short_is_reported_conservative(self):
        _,_,_,sample=fixture(3)
        with patch.object(v,'check_sample_assembly'),patch.object(v,'native_run',return_value=dict(initial_sha256='a',final_sha256='b')):
            result=v.run_sample(sample)
        self.assertTrue(result['passed'])
        self.assertEqual(sum(r['outcome']=='conservative_one_short' for r in result['runs']),2)
        self.assertEqual(v.summarize_runs(result['runs'])['vpu']['negative_rejections'],0)

    def test_native_positive_failure_cannot_pass(self):
        _,_,_,sample=fixture(3)
        def native(*args,**kwargs):
            if kwargs.get('serial'):return dict(initial_sha256='a',final_sha256='b')
            raise v.NativeFault('RuntimeError','VPU command relu issued while its issue-busy bit is set',2,[0])
        with patch.object(v,'check_sample_assembly'),patch.object(v,'native_run',side_effect=native):
            result=v.run_sample(sample)
        self.assertFalse(result['passed'])

    def test_malformed_failure_cannot_count_as_resource_rejection(self):
        self.assertFalse(v.resource_fault(v.NativeFault('ValueError','bad assembly',0,[])))
        self.assertFalse(v.resource_fault(v.NativeFault('RuntimeError','CUDA unavailable',0,[])))
        self.assertFalse(v.resource_fault(v.NativeFault('BankConflictError','MRF register outside m0..m63',0,[])))
        self.assertTrue(v.resource_fault(v.NativeFault('BankConflictError','MRF bank conflict: multiple read ports',3,[0])))

    def test_malformed_gap_and_duplicate_domain_rejected(self):
        facts,raw,candidate,sample=fixture()
        v.validate_candidate(candidate,facts,raw)
        for change in ({'minimum':0},{'high':65},{'minimum':True}):
            bad=copy.deepcopy(candidate);bad['pairs'][0].update(change)
            with self.assertRaises(ValueError):v.validate_candidate(bad,facts,raw)
        candidate['pairs'].append(copy.deepcopy(candidate['pairs'][0]))
        with self.assertRaises(ValueError):v.validate_candidate(candidate,facts,raw)

    def test_sample_gap_must_bind_derived_minimum(self):
        _,_,candidate,sample=fixture(3)
        sample['gaps']=[2]
        with self.assertRaises(ValueError):v.sample_plan(dict(schema='atlas.overlap_samples.v1',samples=[sample]),candidate)

    def test_sample_bound_does_not_silently_truncate(self):
        _,_,candidate,sample=fixture()
        other=dict(sample,id='case1',mechanism='another-source-predicate')
        with self.assertRaises(ValueError):v.sample_plan(dict(schema='atlas.overlap_samples.v1',samples=[sample,other]),candidate,max_samples=1)

    def test_each_chain_mechanism_retains_its_positive_and_reject_control(self):
        _,_,candidate,pair=fixture()
        samples=[]
        for mechanism in ('read-port-history','retired-prefix-history'):
            for expectation in ('positive','reject'):
                samples.append(dict(pair,id=mechanism+'-'+expectation,
                                    mechanism=mechanism,category='port_history',
                                    operations=['vpu.relu']*3,assembly=['one','two','three'],
                                    gaps=[1,32] if expectation=='positive' else [1,31],
                                    expectation=expectation))
        planned=v.sample_plan(dict(schema='atlas.overlap_samples.v1',samples=samples),candidate)
        self.assertEqual({row['id'] for row in planned},{row['id'] for row in samples})
        self.assertEqual(len(planned),4)

    def test_actual_alias_decoding_does_not_trust_supplied_relation(self):
        a={'registers':[0,1],'accumulator':0};b={'registers':[32,1],'accumulator':1}
        self.assertEqual(v.actual_relation(a,b),{'mreg_banks':'0~0,1=1','accumulator':'other'})

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):json.loads('{"a":1,"a":2}',object_pairs_hook=v._object)

    def test_cross_unit_scope_is_explicit_and_direction_independent(self):
        for first, second in [('vpu.relu', 'xlu.transpose'), ('xlu.transpose', 'vpu.sub')]:
            self.assertEqual(v.pair_engine(first, second), 'vpu_xlu')
        with self.assertRaisesRegex(ValueError, 'unsupported cross-engine'):
            v.pair_engine('vpu.relu', 'mxu0.matmul')

    def test_chain_counters_only_count_actual_rejections(self):
        runs=[dict(engine='vpu',category='chain',expected='positive',passed=True,outcome='accepted_equal',numerical_checked=True,numerical_equal=True),
              dict(engine='vpu',category='capacity',expected='reject',passed=True,outcome='native_rejected',numerical_checked=False),
              dict(engine='vpu',category='pair',expected='reject_or_conservative',passed=True,outcome='zero_unencodable',numerical_checked=False)]
        counts=v.summarize_runs(runs)['vpu']
        self.assertEqual(counts['negative_runs'],2)
        self.assertEqual(counts['negative_rejections'],1)
        self.assertEqual(counts['chain_positive_runs'],1)
        self.assertEqual(counts['chain_negative_runs'],1)

    def test_nonzero_patterns_distinguish_every_register_row_and_column(self):
        patterns=[v.nonzero_mreg_bytes(bank) for bank in range(64)]
        self.assertEqual(len(set(patterns)),64)
        self.assertNotEqual(patterns[0],patterns[32])
        self.assertNotEqual(patterns[1],patterns[33])
        self.assertNotEqual(patterns[0],patterns[8])
        for raw in patterns:
            rows=[raw[i:i+32] for i in range(0,len(raw),32)]
            self.assertEqual(len(set(rows)),32)
            for row in rows:
                self.assertEqual(len({row[i:i+2] for i in range(0,32,2)}),16)

    def test_nonzero_patterns_are_positive_small_finite_in_both_formats(self):
        for bank in range(64):
            raw=v.nonzero_mreg_bytes(bank)
            # Positive E4M3FN encodings 0x01..0x3e have no NaN/infinity,
            # and span only positive subnormal values through 1.75.
            self.assertTrue(all(1<=byte<=0x3e for byte in raw))
            for position in range(0,len(raw),2):
                bits=int.from_bytes(raw[position:position+2],'little')<<16
                value=struct.unpack('<f',struct.pack('<I',bits))[0]
                self.assertTrue(math.isfinite(value) and 0<value<0.2)


@unittest.skipUnless(os.environ.get('ATLAS_TEST_NATIVE_OVERLAP_MODEL'),
                     'set ATLAS_TEST_NATIVE_OVERLAP_MODEL for native initializer regressions')
class NativeInitializerObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v.worker_init(os.environ['ATLAS_TEST_NATIVE_OVERLAP_MODEL'])

    def test_streamed_accumulator_push_pop_changes_aliased_destination(self):
        assembly=['vmatpush.acc.bf16.mxu0 acc0, m0',
                  'vmatpop.bf16.acc.mxu0 m32, acc0']
        actual=v.native_run('mxu0',assembly,[2],True)
        serial=v.native_run('mxu0',assembly,[2],True,serial=True)
        self.assertEqual(actual['final_sha256'],serial['final_sha256'])
        tensors=actual['final_state']['tensors']
        for source,destination in ((0,32),(1,33)):
            self.assertEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(source)))
            self.assertNotEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(destination)))

    def test_vmov_changes_both_destination_registers(self):
        actual=v.native_run('vpu',['vmov m8, m0'],[],True)
        tensors=actual['final_state']['tensors']
        for source,destination in ((0,8),(1,9)):
            self.assertEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(source)))
            self.assertNotEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(destination)))

    def test_cross_unit_prototype_points_and_source_minima(self):
        for assembly, gap in [
                (['vredmax.row.bf16 m4, m0', 'vtrpose.xlu m12, m10'], 1),
                (['vtrpose.xlu m12, m10', 'vsub.bf16 m16, m20, m24'], 1),
                (['vtrpose.xlu m12, m10', 'vsub.bf16 m16, m20, m24'], 33)]:
            with self.subTest(assembly=assembly, gap=gap):
                actual = v.native_run('vpu_xlu', assembly, [gap], True)
                serial = v.native_run('vpu_xlu', assembly, [gap], True, serial=True)
                self.assertEqual(actual['final_sha256'], serial['final_sha256'])
                self.assertEqual(actual['issued_ages'], [0, gap])

    def test_cross_unit_read_alias_has_no_vpu_cache_waiver(self):
        assembly = ['vtrpose.xlu m12, m0', 'vmov m8, m0']
        actual = v.native_run('vpu_xlu', assembly, [33], True)
        serial = v.native_run('vpu_xlu', assembly, [33], True, serial=True)
        self.assertEqual(actual['final_sha256'], serial['final_sha256'])
        with self.assertRaises(v.NativeFault) as caught:
            v.native_run('vpu_xlu', assembly, [32], True)
        self.assertTrue(v.resource_fault(caught.exception))

    def test_initialized_matrices_remain_finite_without_fp8_saturation(self):
        import torch
        from npu_model.hardware.arch_state import ArchState
        from atlas_model_timing import configuration
        cfg,_=configuration()
        state=ArchState(cfg.arch_state_config)
        try:
            v.initialize_state(state,True)
            for bank in range(64):
                for dtype in (torch.bfloat16,torch.float8_e4m3fn):
                    values=state.mrf[bank].view(dtype).float()
                    self.assertTrue(torch.isfinite(values).all().item())
                    self.assertTrue(((values>0)&(values<2)).all().item())
            acc_hashes=[];weight_hashes=[]
            for engine in ('mxu0','mxu1'):
                for slot in (0,1):
                    acc=state.acc[engine][slot]
                    self.assertTrue(torch.isfinite(acc).all().item())
                    acc_hashes.append(v.digest(acc.contiguous().view(torch.uint8).numpy().tobytes()))
                    weight=state.wb[engine][slot]
                    weight_hashes.append(v.digest(weight.contiguous().view(torch.uint8).numpy().tobytes()))
            self.assertEqual(len(set(acc_hashes)),4)
            self.assertEqual(len(set(weight_hashes)),4)
        finally:
            state.close()


if __name__=='__main__':unittest.main()
