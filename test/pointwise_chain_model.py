"""Execute the generated pointwise-chain C ABI with the original model."""
import ctypes as ct
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    spec_path,receipt_path,atlas_root=map(Path,sys.argv[1:])
    sys.path.insert(0,str(atlas_root))
    sys.path.insert(0,str(atlas_root.parent/'merlin/examples/atlas/whole-model/unchanged_fx'))
    from functional_runtime import FunctionalModelRuntime
    spec=json.loads(spec_path.read_text());receipts=[]
    torch.set_num_threads(1)
    for index,group in enumerate(spec):
        binary=bytes.fromhex(group['binary']);digest=sha256(binary).hexdigest()
        library=ct.CDLL(group['library'])
        for case_index,case in enumerate(group['cases']):
            runtime=FunctionalModelRuntime(atlas_root,Path(os.environ['IMPORT_MODEL_ROOT']),[binary],
                spec_path.parent/f'model-{index}-{case_index}',image_plans={digest:group['execution']},
                trace_payloads=False,memory_bytes=max(262144,group['arena_bytes']+2048))
            library.call.argtypes=[type(callback) for callback in runtime.callbacks]+[ct.POINTER(ct.c_void_p),ct.c_void_p]
            inputs=[np.asarray(value,np.float32) for value in case['inputs']]
            frozen=[value.tobytes() for value in inputs]
            output=np.empty(group['output_shape'],np.float32);output.view(np.uint8).fill(0xa5)
            pointers=(ct.c_void_p*len(inputs))(*(value.ctypes.data for value in inputs))
            code=library.call(*runtime.callbacks,pointers,output.ctypes.data)
            execution=runtime.finish()
            assert code==case['status'],(group['name'],case['name'],code)
            assert frozen==[value.tobytes() for value in inputs]
            if code:
                assert np.all(output.view(np.uint8)==0xa5),(group['name'],case['name'],'non-atomic rejection')
            else:
                assert output.tobytes()==bytes.fromhex(case['expected']), (group['name'],case['name'],output)
            assert len(execution['launches'])==case.get('launches',1)
            receipts.append(dict(group=group['name'],case=case['name'],return_code=code,
                                 launches=len(execution['launches']),instructions=execution['total_instructions'],
                                 immutable_inputs=True,atomic_rejection=bool(code)))
    receipt_path.write_text(json.dumps(receipts,indent=2)+'\n')


if __name__=='__main__':main()
