"""Pinned-model half of test_exponent_restoration, using the model's Python."""
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
    spec=json.loads(spec_path.read_text());layout=spec['layout']
    sys.path.insert(0,str(atlas_root))
    sys.path.insert(0,str(atlas_root.parent/'merlin/examples/atlas/whole-model/unchanged_fx'))
    from tools.check_program_decode import decode_binary
    from tools.model_instructions import model_instruction
    from functional_runtime import PhysicalFunctionalInterpreter
    from model_runtime import authenticate_model

    torch.set_num_threads(1)
    model=Path(os.environ['IMPORT_MODEL_ROOT']);before=authenticate_model(model)
    binary=bytes.fromhex(spec['binary'])
    inventory=json.loads((atlas_root/'docs/selected-variant-inventory.json').read_text())
    decoded=decode_binary(inventory,binary)
    entries=[]
    for i,row in enumerate(decoded['instructions']):
        instruction=model_instruction(row,inventory,isa)
        entries.append(FunctionalInstruction(i,instruction,i+1,str(instruction)))
    program=FunctionalProgram(entries)
    config_values={field.name:getattr(DefaultHardwareConfig().arch_state_config,field.name) for field in fields(FunctionalConfig)}
    base=0x90000000
    memory_bytes=layout['status_address']-base+layout['status_bytes']
    config_values.update(dram_base=base,dram_size=memory_bytes,numerics='rtl')
    interpreter=PhysicalFunctionalInterpreter(FunctionalConfig(**config_values))
    receipts=[]
    for case in spec['cases']:
        memory=bytes.fromhex(case['memory']);label=case['label']
        result=interpreter.run(program,dram={base:torch.tensor(list(memory),dtype=torch.uint8)})
        actual=bytes(result.state.read_dram(base,memory_bytes).tolist())
        assert result.state.halted,label
        at=layout['status_address']-base
        status=int.from_bytes(actual[at:at+4],'little')
        assert bool(status)==case['rejected'],(label,status)
        prefix=layout['output_addresses'][0]-base
        assert actual[:prefix]==memory[:prefix],label
        if case['expected'] is not None:
            expected=bytes.fromhex(case['expected'])
            assert actual[prefix:prefix+len(expected)]==expected,label
        receipts.append(dict(case=label,rejected=case['rejected'],instructions=result.instructions_executed))
    assert before==authenticate_model(model)
    receipt_path.write_text(json.dumps(dict(cases=receipts,model_sources=before,program_words=len(entries),
                                            binary_sha256=sha256(binary).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    main()
