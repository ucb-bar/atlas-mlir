"""Run packed-boundary fixtures on the authenticated instruction model."""
from dataclasses import fields
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import torch
from npu_model.configs import isa_definition as isa
from npu_model.configs.hardware.default import DefaultHardwareConfig
from npu_model.functional import FunctionalConfig
from npu_model.functional.program import FunctionalInstruction, FunctionalProgram


def main():
    spec_path,receipt_path,atlas_root=map(Path,sys.argv[1:])
    spec=json.loads(spec_path.read_text())
    sys.path.insert(0,str(atlas_root))
    sys.path.insert(0,str(atlas_root.parent/'merlin/examples/atlas/whole-model/unchanged_fx'))
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction
    from functional_runtime import PhysicalFunctionalInterpreter
    from model_runtime import authenticate_model

    torch.set_num_threads(1)
    model=Path(os.environ['IMPORT_MODEL_ROOT']);before=authenticate_model(model)
    inventory=json.loads((atlas_root/'docs/selected-variant-inventory.json').read_text())
    receipts=[]
    for group in spec:
        binary=bytes.fromhex(group['binary']);entries=[]
        for i,row in enumerate(decode_binary(inventory,binary)['instructions']):
            instruction=model_instruction(row,inventory,isa)
            entries.append(FunctionalInstruction(i,instruction,i+1,str(instruction)))
        config={field.name:getattr(DefaultHardwareConfig().arch_state_config,field.name) for field in fields(FunctionalConfig)}
        config.update(dram_base=0x90000000,dram_size=len(bytes.fromhex(group['cases'][0]['memory'])),numerics='rtl')
        interpreter=PhysicalFunctionalInterpreter(FunctionalConfig(**config))
        for case in group['cases']:
            memory=bytes.fromhex(case['memory']);expected=bytes.fromhex(case['expected_memory'])
            result=interpreter.run(FunctionalProgram(entries),dram={0x90000000:torch.tensor(list(memory),dtype=torch.uint8)})
            actual=bytes(result.state.read_dram(0x90000000,len(memory)).tolist())
            assert result.state.halted,case['label']
            assert actual==expected,(group['kind'],case['label'],next((i,a,b) for i,(a,b) in enumerate(zip(actual,expected,strict=True)) if a!=b))
            receipts.append(dict(kind=group['kind'],case=case['label'],instructions=result.instructions_executed,
                                 program_words=len(entries),binary_sha256=sha256(binary).hexdigest()))
    assert before==authenticate_model(model)
    receipt_path.write_text(json.dumps(dict(cases=receipts,model_sources=before),indent=2)+'\n')


if __name__=='__main__':
    main()
