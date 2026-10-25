"""Gate controls plus opt-in native regressions for numerical observability."""
import copy
import importlib.util
import json
import os
from pathlib import Path
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
    def test_removed_pair_domains_refuse_before_native_execution(self):
        for first, second, minimum, high in (("scalar_lsu.load", "scalar_lsu.load", 3, 4),
                                             ("vpu.relu", "xlu.transpose", 1, 66)):
            with self.subTest(first=first, second=second):
                facts, _, candidate, _ = fixture()
                facts['op_timing'].extend(dict(name=name, first_free_age=high)
                                          for name in {first, second} - {'vpu.relu'})
                candidate['pairs'] = [dict(first=first, second=second,
                                          relation={'mreg_banks': 'disjoint'}, minimum=minimum, high=high)]
                raw = json.dumps(facts).encode()
                candidate['per_op_sha256'] = v.digest(raw)
                with self.assertRaisesRegex(ValueError, 'unsupported cross-engine pair'):
                    v.validate_candidate(candidate, facts, raw)

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

@unittest.skipUnless(os.environ.get('ATLAS_TEST_NATIVE_OVERLAP_MODEL'),
                     'set ATLAS_TEST_NATIVE_OVERLAP_MODEL for native initializer regressions')
class NativeInitializerObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v.worker_init(os.environ['ATLAS_TEST_NATIVE_OVERLAP_MODEL'])

    def test_native_copy_changes_both_destination_registers(self):
        for engine,assembly,gaps,dest in (
                ('mxu0',['vmatpush.acc.bf16.mxu0 acc0, m0','vmatpop.bf16.acc.mxu0 m32, acc0'],[2],32),
                ('vpu',['vmov m8, m0'],[],8)):
            with self.subTest(engine=engine):
                actual=v.native_run(engine,assembly,gaps,True)
                serial=v.native_run(engine,assembly,gaps,True,serial=True)
                self.assertEqual(actual['final_sha256'],serial['final_sha256'])
                tensors=actual['final_state']['tensors']
                for source,destination in ((0,dest),(1,dest+1)):
                    self.assertEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(source)))
                    self.assertNotEqual(tensors[destination],v.digest(v.nonzero_mreg_bytes(destination)))

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
