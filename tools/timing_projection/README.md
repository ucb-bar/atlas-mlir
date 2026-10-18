# Schedule timing projection

`predict.py` predicts the cycle count of a compiled Atlas program without running its arithmetic. It reads the `atlas.physical_program.v1` JSON from `atlas-emit --program-json` and a memory profile, and models what can delay issue: the compiler's explicit delays, scalar address and loop registers, the eight DMA channels with their credits, and VMEM bank conflicts. It reuses npu_model's own memory backend.

This works because a legal Atlas schedule spaces compute with explicit delays; the only dynamic stall is a DMA wait. On the attention, MLP and combined attention→MLP programs it matched npu_model's measured cycles exactly on every tested memory profile, in about 10 ms per case instead of about 1 s for a numerical run.

It assumes the program already passed the compiler's checks; it does not check hazards or numerical results. Programs must end with the compiler's completion CSR write followed by ECALL.

```sh
PYTHONPATH=/path/to/npu_model python3 tools/timing_projection/predict.py program.json --latency 43 --spacing 1
cd tools/timing_projection && PYTHONPATH=/path/to/npu_model python3 -m unittest test_predict
```
