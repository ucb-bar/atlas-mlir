"""Full Tiny policy controls preserve baseline admission and reject dishonest DMA."""
import copy
import sys
from pathlib import Path
import unittest

TOOLS = Path(__file__).absolute().parents[1]/'tools'
sys.path[:0] = [str(TOOLS),str(TOOLS/'host_iss')]
from atlas_program import dma_load_bytes
from tiny_full_contract import (CYCLE_CAP,parameter_geometry,plan,reference_geometry,validate_plan)


def fixture(reuse):
    kernels = []
    for index,shape in enumerate(plan(1)):
        m,k,n = shape['physical_mkn']
        load = (m*k*(n//32)+k*n if k*n <= 262144 else
                m*k+k*n*(m//32) if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        if reuse:
            load = dma_load_bytes(m,k,n,traffic_reuse=True)
        kernels.append(dict(shape,index=index,A_scale=0,B_scale=0,rescale=0,dram_base=0x100000000,
                            output_offset=m*k+k*n+256,total_dram_bytes=m*k+k*n+512+2*m*n,
                            requested_dma=dict(load_bytes=load,store_bytes=2*m*n)))
    references = []
    for name,(shape,dtype) in reference_geometry(1).items():
        count = 1
        for extent in shape:
            count *= extent
        references.append(dict(actual_symbol=name,expected_symbol=name+'_expected',shape=shape,dtype=dtype,
                               bytes=count*(2 if dtype == 'uint16' else 4)))
    parameters = {name:dict(shape=shape,dtype=dtype,symbol=f'tiny_full_param_{index:03d}')
                  for index,(name,(shape,dtype)) in enumerate(sorted(parameter_geometry().items()))}
    data = dict(schema='atlas.tiny_full_contract.v1',decode_count=1,kernel_count=len(kernels),projection_calls=310,
                cycle_cap=CYCLE_CAP,kernels=kernels,reference_buffers=references,host_parameters=parameters,
                greedy_tokens=[0,0],mailbox_symbol='tiny_full_mailbox',panel_symbol='tiny_full_panels',
                state_symbol='tiny_full_state_global',context_symbol='tiny_full_projection_state')
    if reuse:
        data.update(program_policy='traffic-reuse-v1',reuse=dict(captured_data_unchanged=True,instruction_execution=False))
    return data,copy.deepcopy(data)


class FullReuseTest(unittest.TestCase):
    def test_original_and_public_reuse_have_distinct_valid_dma(self):
        original,old_provenance = fixture(False)
        reuse,new_provenance = fixture(True)
        validate_plan(original,old_provenance)
        validate_plan(reuse,new_provenance)
        self.assertTrue(any(a['requested_dma'] != b['requested_dma']
                            for a,b in zip(original['kernels'],reuse['kernels'])))
        self.assertEqual([x['physical_mkn'] for x in original['kernels']],
                         [x['physical_mkn'] for x in reuse['kernels']])

    def test_reuse_cannot_retain_original_dma_or_change_policy(self):
        original,_ = fixture(False)
        reuse,provenance = fixture(True)
        index = next(i for i,(a,b) in enumerate(zip(original['kernels'],reuse['kernels']))
                     if a['requested_dma'] != b['requested_dma'])
        reuse['kernels'][index]['requested_dma'] = original['kernels'][index]['requested_dma']
        provenance['kernels'] = copy.deepcopy(reuse['kernels'])
        with self.assertRaisesRegex(ValueError,'physical layout or DMA'):
            validate_plan(reuse,provenance)
        reuse,provenance = fixture(True)
        reuse['program_policy'] = 'original'
        with self.assertRaisesRegex(ValueError,'policy binding'):
            validate_plan(reuse,provenance)

    def test_reuse_cannot_claim_instruction_execution(self):
        reuse,provenance = fixture(True)
        reuse['reuse']['instruction_execution'] = True
        provenance['reuse'] = copy.deepcopy(reuse['reuse'])
        with self.assertRaisesRegex(ValueError,'preparation binding'):
            validate_plan(reuse,provenance)


if __name__ == '__main__':
    unittest.main()
