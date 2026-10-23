"""Selected-model identity, decoding and execution failures must fail closed."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_scorer import ModelScorer, model_instruction, selected_model
from check_program_decode import decode_binary


class ScorerGuardsTest(unittest.TestCase):
    def test_assertions_cannot_be_disabled(self):
        with patch("atlas_model_scorer.sys.flags") as flags:
            flags.optimize = 1
            with self.assertRaisesRegex(ValueError, "assertions"):
                ModelScorer(Path("/tmp"))({})

    def test_each_model_identity_component_is_required(self):
        keys = ("source_sha256", "numerical_data_sha256", "configuration_sha256")
        for key in keys:
            evidence = dict(zip(keys, ("source", "data", "config")))
            evidence[key] = "changed"
            with self.subTest(key=key), patch("atlas_model_scorer.selected_model",
                                             return_value=(Path("/tmp"), "source", "data", "config")):
                with self.assertRaisesRegex(ValueError, "differs from provider evidence"):
                    ModelScorer(Path("/tmp"))({"model_evidence": evidence})


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for model execution")
class ModelScorerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        _, source, data, config = selected_model(root)
        cls.scorer = ModelScorer(root)
        cls.evidence = dict(source_sha256=source, numerical_data_sha256=data, configuration_sha256=config)
        cls.inventory = json.loads((Path(__file__).parents[1] / "docs/selected-variant-inventory.json").read_text())

    def program(self, words):
        return dict(model_evidence=self.evidence, instructions=[dict(word_u32=word) for word in words])

    def test_shift_immediate_decodes_without_function_bits(self):
        from npu_model.configs import isa_definition as isa
        for word in (0x40115093, 0x00115093, 0x00111093):
            with self.subTest(word=word):
                row = decode_binary(self.inventory, word.to_bytes(4, "little"))["instructions"][0]
                instruction = model_instruction(row, self.inventory, isa)
                self.assertEqual(instruction.to_bytecode(), word)
                self.assertEqual(instruction.imm & 31, 1)

    def test_cold_cycles_profiles_and_exhaustion(self):
        # ADDI x1, x0, 1; CSRRW x0, completion, x1; ECALL.
        program = self.program([0x00100093, 0xc1009073, 0x00000073])
        for profile in ("fifo43", "curve"):
            self.assertEqual(self.scorer(program, profile), 3)
        with self.assertRaisesRegex(ValueError, "exhausted before ECALL"):
            self.scorer(self.program([0x00100093]))


if __name__ == "__main__":
    unittest.main()
