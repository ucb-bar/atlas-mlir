"""Run upstream cosim tests on the selected AtlasCore with an explicit halt state.

The selected ScalarCore drives io.halted from scalar/halt_now, and optimization can drop the public
io_halted port. Runners that take `halt_signal` receive it directly; older runners get a peek alias.
"""
import inspect

HALT_SIGNAL = "scalar/halt_now"


def run_selected_program(cosim_atlas, model_path, manifest_path, words, **kwargs):
    runner = cosim_atlas.run_program
    if "halt_signal" in inspect.signature(runner).parameters:
        return runner(model_path, manifest_path, words, halt_signal=HALT_SIGNAL, **kwargs)
    original = cosim_atlas.CosimCore

    class SelectedHaltCore(original):
        def peek(self, name):
            return super().peek(HALT_SIGNAL if name == "io_halted" else name)

    cosim_atlas.CosimCore = SelectedHaltCore
    try:
        return runner(model_path, manifest_path, words, **kwargs)
    finally:
        cosim_atlas.CosimCore = original
