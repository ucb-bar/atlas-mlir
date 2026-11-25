"""Ordinary virtual quantizer execution on the unchanged model."""
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import subprocess
import unittest
import numpy as np
from atlas_mlir_native import vector_quantize as vector
from atlas_mlir_native.activation_quantize_device import prepare_plan, virtual_program
from test_activation_quantize_device import dynamic_source, host_reference
from test_already_quantized import quantizer_source
ROOT=Path(__file__).resolve().parents[1]
BIN=Path(os.environ.get('ATLAS_OOT_BIN_DIR',ROOT/'build/bin'))


@unittest.skipUnless(os.environ.get('ATLAS_RUN_NPU_MODEL')=='1','set ATLAS_RUN_NPU_MODEL=1')
class VectorQuantizeModelTest(unittest.TestCase):
    def test_directed_vectors_exponents_padding_and_guards_on_unchanged_model(self):
        import dataclasses
        import tempfile
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.logging import LoggerConfig
        from npu_model.simulation import Simulation
        from npu_model.software.program import InstantiableProgram
        from tools.check_program_decode import decode_binary
        from tools.model_instructions import model_instruction
        torch.set_num_threads(1)
        inventory=json.loads((ROOT/'docs/selected-variant-inventory.json').read_text())
        with tempfile.TemporaryDirectory(prefix='atlas-vector-quant-model-') as directory:
            cases=(quantizer_source(count=32,fixed=-127),dynamic_source(shape=(1,1,64),reduce='head'),
                   dynamic_source(shape=(1,1,96),reduce='head_block',rule='nc'),
                   dynamic_source(shape=(1,1,96),reduce='head',rule='ref'))
            for index,source in enumerate(cases):
                with self.subTest(index=index):
                    plan=vector.prepare_plan(source)
                    values=np.resize(np.array([0,0x8000,1,127,128,0x3f88,0x3f98,0x3ff8,
                                               0x7f7f,0xff7f]+([0x7f80,0xff80,0x7fc1,0xffff] if index!=2 else []),dtype='<u2'),plan.count)
                    raw=values.tobytes()
                    compiled=Path(directory)/str(index)
                    vector.compile_source(None,source,target_source=ROOT,destination=compiled,limits=None)
                    binary=(compiled/'program.bin').read_bytes()
                    decoded=decode_binary(inventory,binary)
                    program=InstantiableProgram([model_instruction(row,inventory,isa) for row in decoded['instructions']])
                    program.dram_base=0x90000000
                    initial=bytearray([0xa5]*(plan.workspace_bytes+64))
                    initial[:plan.input_bytes]=raw+bytes(plan.input_bytes-len(raw))
                    wanted=bytearray(initial)
                    codes,exps=host_reference(source,raw)
                    ca,ea=plan.code_address-program.dram_base,plan.exponent_address-program.dram_base
                    wanted[ca:ca+plan.count]=codes
                    wanted[ca+plan.count:ea]=bytes(ea-ca-plan.count)
                    wanted[ea:ea+len(exps)]=exps
                    wanted[ea+len(exps):plan.workspace_bytes]=bytes(plan.exponent_bytes-len(exps))
                    program.memory_regions=[(0,torch.tensor(list(initial),dtype=torch.uint8))]
                    config=DefaultHardwareConfig()
                    config.arch_state_config=dataclasses.replace(config.arch_state_config,numerics='rtl',randomize_init=True,init_seed=997)
                    sim=Simulation(config,LoggerConfig(filename=str(Path(directory)/f'model-{index}.json')),program,verbose=False)
                    state=sim.core.arch_state
                    try:
                        state.vmem.fill_(0xcd)
                        sim.run()
                        self.assertTrue(state.halted)
                        self.assertEqual(sim.get_stats().runtime_errors,0)
                        self.assertEqual(bytes(state.read_dram(program.dram_base,len(initial)).tolist()),bytes(wanted))
                        if os.environ.get('ATLAS_VECTOR_QUANT_RECEIPTS'):
                            import hashlib
                            receipts=Path(os.environ['ATLAS_VECTOR_QUANT_RECEIPTS'])
                            receipts.mkdir(parents=True,exist_ok=True)
                            (receipts/f'model-{index}.json').write_text(json.dumps(dict(
                                status='PASS',shape=list(plan.x.type.shape),attrs=plan.attrs,
                                program_sha256=hashlib.sha256(binary).hexdigest(),program_bytes=len(binary),
                                cycles=sim.get_stats().cycles,functional_instructions=sim.get_stats().total_instructions,
                                runtime_errors=sim.get_stats().runtime_errors,checked_bytes=len(initial),
                                checked_guards_bytes=64,simulation_max_cycles='default 10000',
                                numerical_oracle='phase1 host C codes and signed int32 exponents'),indent=2)+'\n')
                    finally:
                        state.close()

if __name__=="__main__": unittest.main()
