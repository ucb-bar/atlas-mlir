"""Native scoring preserves words and refuses incomplete or incorrect results."""
from io import StringIO
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_model_scorer import (ModelScorer, bind_curve_configuration,
                                bind_fifo43_configuration, model_instruction)
from atlas_model_timing import sha256
from check_program_decode import decode_binary
from model_evidence import NativeModelFixture


class ScorerGuardsTest(unittest.TestCase):
    def test_assertions_required(self):
        with patch("atlas_model_scorer.sys.flags") as flags:
            flags.optimize = 1
            with self.assertRaisesRegex(ValueError, "assertions"):
                ModelScorer(Path("/tmp"))({})

    def test_curve_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "curve.json"
            raw = b'{"curves":{"100":[[100,20]]}}\n'
            selected.write_bytes(raw)
            cfg = SimpleNamespace(dma_memory_params={"curves": str(selected)})
            with patch("atlas_model_scorer.resolve_curve_path", return_value=selected):
                for evidence in ({}, {"dma_curve_sha256": "changed"}):
                    with self.assertRaisesRegex(ValueError, "DMA curve differs"):
                        bind_curve_configuration(cfg, evidence)
                bind_curve_configuration(cfg, {"dma_curve_sha256": sha256(raw)})
            self.assertEqual(cfg.dma_memory_params["curves"], json.loads(raw))

    def test_fifo43_binding(self):
        cfg = SimpleNamespace()
        bind_fifo43_configuration(cfg)
        self.assertEqual(cfg.dma_memory_backend, "fixed")
        self.assertEqual(cfg.dma_memory_params["latency"], 43)
        self.assertEqual(cfg.dma_memory_params["cycles_per_beat"], 1)
        accept = pickle.loads(pickle.dumps(cfg.dma_memory_params["accept_fn"]))
        self.assertEqual([accept(cycle) for cycle in range(20, 45)],
                         [cycle % 11 < 4 for cycle in range(20, 45)])


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT")
class ModelScorerTest(NativeModelFixture, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from atlas_model_facts import regenerate
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        facts = Path(temporary.name) / "facts/model-timing.json"
        regenerate(os.environ["ATLAS_DERIVE_MODEL_ROOT"], facts.parent)
        cls.evidence = dict(op_timing=str(facts), op_timing_sha256=sha256(facts.read_bytes()),
                            dma_curve_sha256=json.loads(facts.read_text())["dma_timing"]["curve_sha256"])

    def setUp(self):
        super().setUp()
        self.root = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        self.scorer = ModelScorer(self.root)
        self.inventory = json.loads((Path(__file__).parents[1] /
                                     "docs/selected-variant-inventory.json").read_text())

    def program(self, assembly):
        from npu_model.util.converter import stream_to_instrs
        return dict(instructions=[dict(word_u32=instruction.to_bytecode())
                                  for instruction in stream_to_instrs(StringIO(assembly))])

    def test_reencoding(self):
        from npu_model.configs import isa_definition as isa
        words = [int(mode["rtl_bitpat"].replace("?", "0"), 2) for mode in self.inventory["variants"]]
        words += [row["word_u32"] for row in self.program(
            "srai x1, x2, 1\nvpack.bf16.fp8 m7, m34, e31\nvload m35, -16(x3)")["instructions"]]
        for word in words:
            with self.subTest(word=hex(word)):
                row, = decode_binary(self.inventory, word.to_bytes(4, "little"))["instructions"]
                self.assertEqual(model_instruction(row, self.inventory, isa).to_bytecode(), word)

    def test_model_mismatch(self):
        with self.assertRaisesRegex(ValueError, "imported npu_model differs"):
            ModelScorer(self.root.parent)({}, "fifo43")

    def test_completion_and_drain(self):
        program = self.program("addi x1, x0, 1\necall")
        program["model_evidence"] = self.evidence
        for profile in ("curve", "fifo43"):
            self.assertEqual(self.scorer(program, profile), 2)
        with self.assertRaisesRegex(ValueError, "exhausted before ECALL"):
            self.scorer(self.program("addi x1, x0, 1"), "fifo43")
        with self.assertRaisesRegex(ValueError, "ECALL before engines and DMA drained"):
            self.scorer(self.program("vadd.bf16 m4, m0, m2\necall"), "fifo43")

    def test_expected_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "region.json"
            path.write_text(json.dumps(dict(address=0x80000000, initial_hex="0001", expected_hex="0001")))
            program = self.program("addi x1, x0, 1\necall")
            self.assertEqual(ModelScorer(self.root, (path,))(program, "fifo43"), 2)
            path.write_text(json.dumps(dict(address=0x80000000, initial_hex="0001", expected_hex="0002")))
            with self.assertRaisesRegex(ValueError, "differ from expected bytes"):
                ModelScorer(self.root, (path,))(program, "fifo43")


if __name__ == "__main__":
    unittest.main()
