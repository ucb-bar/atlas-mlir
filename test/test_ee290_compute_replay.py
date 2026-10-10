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

SCHEDULED_REPORT=os.environ.get('ATLAS_EE290_SCHEDULED_COMPUTE_REPORT')

class PortableExportBindingTest(unittest.TestCase):
    def test_selected_word_decoding(self):
        mnemonic,operands=COMPUTE.BIND.decoded_operands(0x06100257)
        self.assertEqual(mnemonic,'vmul.bf16')
        self.assertEqual((operands['rd'],operands['rs1'],operands['rs2']),(4,0,2))
        with self.assertRaises(COMPUTE.BIND.BindingError): COMPUTE.BIND.decoded_operands(0)
    def test_stream_expands_contiguous_pair_without_timing_table(self):
        stream=dict(resource='mreg',write=True,first=128,count=64,age=2,step=1,anywhere=False,at_completion=False)
        points=COMPUTE.BIND.finite_stream(stream,100)
        self.assertEqual(points[0],('mreg',True,128,102))
        self.assertEqual(points[-1],('mreg',True,191,165))
        for field,value in [('anywhere',True),('age',None),('count',0),('step',0)]:
            with self.subTest(field=field),self.assertRaises((COMPUTE.BIND.BindingError,TypeError)):
                COMPUTE.BIND.finite_stream({**stream,field:value},100)

@unittest.skipUnless(SCHEDULED_REPORT,'requires explicitly selected scheduled compute capture')
class ScheduledExportBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        checker=COMPUTE.CHECK.Checker();path=COMPUTE.CHECK.allowed(Path(SCHEDULED_REPORT));checker.identity(path)
        report=COMPUTE.CHECK.strict_json(path.read_bytes())
        if report['schema']!='atlas.ee290_scheduled_compute_replay.v0' or report['state']!='finite_scheduled_compute_cases_passed': raise ValueError('successful scheduled packet required')
        cls.captures={}
        ids={item['kind']:checker.verify(item['identity'],path.parent) for item in report['selected_evidence']}
        expected={'evidence_sha256':ids['vls']['sha256'],**{key+'_evidence_sha256':ids[key]['sha256'] for key in ('dma','xlu','vmul')},'manifest_sha256':checker.verify(report['manifest'],path.parent)['sha256'],'hardware_ir_sha256':checker.verify(report['hardware_ir'],path.parent)['sha256']}
        model=checker.verify(report['model'],path.parent)
        for dependency in report['producer_snapshots']: checker.verify(dependency['snapshot'],path.parent)
        for case in report['cases']:
            for key in ('program','authored_program'): checker.verify(case[key],path.parent)
            for key in ('optimizer_phase','emission_phase','export_phase','execution_phase'):
                phase_id=checker.verify(case[key],path.parent)
                phase=COMPUTE.CHECK.strict_json(COMPUTE.CHECK.allowed(phase_id['path']).read_bytes())
                if phase['state']!='phase_completed' or phase['inputs_before']!=phase['inputs_after'] or phase['command']['returncode']!=0 or phase['command']['timed_out']: raise ValueError('invalid captured phase')
                inputs={member['role']:checker.verify(member['identity'],Path(phase_id['path']).parent) for member in phase['inputs_before']}
                outputs=[checker.verify(member['identity'],Path(phase_id['path']).parent) for output in phase['outputs'] for member in output['files']]
                if key in ('emission_phase','export_phase') and inputs['program']!=case['program']: raise ValueError('compiler phases selected different final programs')
                if key=='optimizer_phase' and (inputs['program']!=case['authored_program'] or any(inputs[k+'_evidence']!=ids[k] for k in ids)): raise ValueError('consumer input selection mismatch')
                if key=='execution_phase' and (inputs['tool']!=model or inputs['words']!=case['words'] or phase['command']['argv'][:2]!=[model['path'],case['words']['path']] or case['trace'] not in outputs): raise ValueError('executed model/words/trace not captured')
                if key=='export_phase' and case['resolved_timing'] not in outputs: raise ValueError('resolved export not captured output')
                if key=='emission_phase' and not any((o['sha256'],o['bytes'])==(case['words']['sha256'],case['words']['bytes']) for o in outputs): raise ValueError('selected words differ from compiler output')
            checked={key:checker.verify(case[key],path.parent) for key in ('resolved_timing','events','words','trace','export_binding')}
            export=COMPUTE.CHECK.strict_json(COMPUTE.CHECK.allowed(checked['resolved_timing']['path']).read_bytes())
            saved=COMPUTE.CHECK.strict_json(COMPUTE.CHECK.allowed(checked['events']['path']).read_bytes())
            words=[int(word,16) for word in COMPUTE.CHECK.allowed(checked['words']['path']).read_text().split()]
            mode=case['name'].rsplit('_',1)[0]
            actual=COMPUTE.analyze_compute(Path(checked['trace']['path']),words,mode)
            for key,value in actual.items():
                if saved[key]!=value: raise ValueError('saved boundary transcript differs from decoded VCD')
            cls.captures[case['name']]=(export,saved,expected,words)
        checker.recheck()
    def bind(self,case,export=None,events=None,evidence=None,words=None):
        e,o,ids,w=self.captures[case]
        return COMPUTE.BIND.bind_export(e if export is None else export,o if events is None else events,ids if evidence is None else evidence,w if words is None else words,'atlas.vls_dma_xlu_vmul.serialized.v1')
    def test_actual_both_consumers_exact_access_release_and_epochs(self):
        self.assertEqual(set(self.captures),{'vmul_delay','vmul_schedule','dma_xlu_delay','dma_xlu_schedule'})
        for case in self.captures:
            with self.subTest(case=case):
                result=self.bind(case)
                self.assertGreater(result['memory_access_elements_bound'],0)
                self.assertFalse(result['scheduling_qualified'])
                self.assertEqual(len(result['dynamic_dma_intervals']),2 if case.startswith('dma_xlu') else 0)
    def test_export_word_operand_stream_release_and_evidence_mutations(self):
        case='vmul_schedule';original,events,ids,words=self.captures[case]
        compute=next(i for i in original['instructions'] if i['mnemonic']=='vmul.bf16')
        index=compute['word_index']
        mutations=[lambda e:e['instructions'][index]['operands'].__setitem__('rd',6),
                   lambda e:e['instructions'][index]['footprint']['accesses'][0].__setitem__('first',32),
                   lambda e:e['instructions'][index]['footprint']['accesses'][0].__setitem__('age',1),
                   lambda e:e['instructions'][index]['footprint'].__setitem__('done_age',66),
                   lambda e:e['instructions'][index].__setitem__('epoch_offset',compute['epoch_offset']+1),
                   lambda e:e['evidence'].__setitem__('vmul_evidence_sha256','0'*64),
                   lambda e:e.__setitem__('scheduling_qualified',True),
                   lambda e:e['instructions'][index]['footprint']['accesses'].pop(),
                   lambda e:e['instructions'][index]['footprint']['accesses'].append(copy.deepcopy(e['instructions'][index]['footprint']['accesses'][0])),
                   lambda e:e['instructions'][index]['footprint']['holds'][0].__setitem__('to',66)]
        # Lifetimes and serialized holds hidden by the combined release maximum.
        footprint=lambda e,i=index:e['instructions'][i]['footprint']
        vload=next(i['word_index'] for i in original['instructions'] if i['mnemonic']=='vload')
        mutations+=[lambda e:footprint(e).__setitem__('read_release',63),
                    lambda e:footprint(e).__setitem__('write_release',64),
                    lambda e:footprint(e).__setitem__('done_age',64),
                    lambda e:footprint(e)['mreg_reads'].pop(),
                    lambda e:footprint(e)['mreg_writes'].append(6),
                    lambda e:footprint(e)['holds'].pop(0),
                    lambda e:footprint(e)['holds'][1].__setitem__('to',64),
                    lambda e:footprint(e,vload)['holds'].pop(1),
                    lambda e:footprint(e,vload).__setitem__('mreg_writes',[1])]
        for position,mutate in enumerate(mutations):
            changed=copy.deepcopy(original);mutate(changed)
            with self.subTest(position=position),self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,export=changed)
        changed=words.copy();changed[index]^=1
        with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,words=changed)
    def test_observed_response_operand_and_dynamic_completion_mutations(self):
        case='vmul_delay';_,original,_,_=self.captures[case]
        position=next(i for i,c in enumerate(original['commands']) if c['engine']=='vmul')
        for metadata in ('target_config','data_encoding'):
            changed=copy.deepcopy(original);changed.pop(metadata)
            with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
        for field in ('responses','writes'):
            changed=copy.deepcopy(original);changed['commands'][position][field][0]['edge']+=1
            with self.subTest(field=field),self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
        changed=copy.deepcopy(original);changed['commands'][position]['writes'][0]['row']^=1
        with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
        changed=copy.deepcopy(original);responses=changed['commands'][position]['responses'];responses[-1]=copy.deepcopy(responses[0])
        with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
        for field,mutate in [('reads',lambda e:e.pop('port')),('responses',lambda e:e.pop('port')),
                             ('writes',lambda e:e.pop('port')),('writes',lambda e:e.__setitem__('port',1))]:
            changed=copy.deepcopy(original);mutate(changed['commands'][position][field][0])
            with self.subTest(field=field),self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
        case='dma_xlu_schedule';export,original,_,_=self.captures[case]
        dma=next(c for c in original['commands'] if c['engine']=='dma');index=dma['word_index']
        for mutate in [lambda e:e['instructions'][index]['footprint'].__setitem__('done_age',50),
                       lambda e:e['instructions'][index]['footprint']['accesses'][-1].__setitem__('at_completion',False)]:
            changed=copy.deepcopy(export);mutate(changed)
            with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,export=changed)
        wait=next(i for i in export['instructions'] if i['event_kind']=='matching_wait_acceptance')
        changed=copy.deepcopy(export);changed['instructions'][wait['word_index']]['issue_epoch']-=1
        with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,export=changed)
        changed=copy.deepcopy(original);c=next(c for c in changed['commands'] if c['engine']=='dma');c['memory'][0]['edge']=c['release_edge']
        with self.assertRaises(COMPUTE.BIND.BindingError): self.bind(case,events=changed)
