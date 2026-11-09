"""Original-model proof that BF16 accumulator pop/reload preserves every bit."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import unittest


@unittest.skipUnless(os.environ.get("NPU_MODEL_ROOT"), "select original47 model")
class BF16AccumulatorRoundTripReferenceTest(unittest.TestCase):
    def test_all_65536_bf16_patterns_on_both_units(self):
        import torch
        from npu_model.configs import isa_definition as isa
        from npu_model.configs.hardware.default import DefaultHardwareConfig
        from npu_model.hardware.arch_state import ArchState
        from test_mxu1_cap448 import authenticate_model

        root = Path(os.environ["NPU_MODEL_ROOT"])
        torch.set_num_threads(1)
        state = ArchState(replace(DefaultHardwareConfig().arch_state_config, numerics="rtl"))
        before = authenticate_model(root)
        patterns = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        try:
            for unit, pop, push in (
                ("mxu0", isa.VMATPOP_BF16_ACC_MXU0, isa.VMATPUSH_ACC_BF16_MXU0),
                ("mxu1", isa.VMATPOP_BF16_ACC_MXU1, isa.VMATPUSH_ACC_BF16_MXU1),
            ):
                for offset in range(0, 65536, 1024):
                    with self.subTest(unit=unit, first_pattern=offset):
                        expected = patterns[offset:offset + 1024]
                        state.write_acc_bf16(unit, 0, expected.view(torch.bfloat16).reshape(32, 32))
                        pop(vd=32, vs2=0).exec(state)
                        push(vd=1, vs1=32).exec(state)
                        actual = state.read_acc_bf16(unit, 1).view(torch.uint16).flatten()
                        # Integer comparison also checks signed zero, subnormals,
                        # infinities, and every signaling/quiet NaN payload.
                        self.assertTrue(torch.equal(actual, expected))
        finally:
            state.close()
        self.assertEqual(authenticate_model(root), before)


if __name__ == "__main__":
    unittest.main()
