"""Direct Atlas IMEM packaging checks independent of LLVM object generation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from atlas_direct_pack import checked_words, pack  # noqa: E402


REVISION = "0079c0541111197741a231c002e3843fa6f545b2"


def program() -> dict:
    return {
        "schema": "atlas.physical_program.v1", "selected_rtl_revision": REVISION,
        "pc_unit": "instruction_word", "word_endianness": "little",
        "entry_word_index": 0, "word_count": 3,
        "instructions": [
            {"word_index": 0, "word_u32": 0x0000006F, "word_hex": "0000006f",
             "control": {"kind": "jump_direct", "target_word_index": 2,
                         "delay_slot_word_index": 1}},
            {"word_index": 1, "word_u32": 0x00000013, "word_hex": "00000013",
             "control": {"kind": "sequential"}},
            {"word_index": 2, "word_u32": 0x00000073, "word_hex": "00000073",
             "control": {"kind": "trap"}},
        ],
    }


class DirectPackTest(unittest.TestCase):
    def test_closed_program_and_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "physical.mlir"
            source.write_text("module {}\n")
            layout = root / "layout.json"
            layout.write_text(json.dumps({
                "schema": "atlas.reset-entry-layout.v1",
                "selected_rtl_revision": REVISION,
                "dram_window_base": "0x90000000",
                "regions": [
                    {"name": "input", "kind": "input", "address": "0x90000000", "size_bytes": 32},
                    {"name": "output", "kind": "output", "address": "0x90000020", "size_bytes": 32},
                ],
            }))
            emitter = root / "atlas-emit"
            emitter.write_text("#!/usr/bin/env python3\nimport json\nprint(json.dumps(" + repr(program()) + "))\n")
            emitter.chmod(0o755)
            manifest = pack(source, emitter, layout, root / "image")
            self.assertEqual((root / "image/program.bin").read_bytes(),
                             bytes.fromhex("6f0000001300000073000000"))
            self.assertEqual(manifest["completion"]["ecall_pc_word"], 2)
            self.assertEqual(manifest["program_words"], 3)

    def test_refuses_unresolved_or_escaping_control(self):
        candidate = program()
        candidate["instructions"][0]["control"]["target_word_index"] = 3
        with self.assertRaisesRegex(ValueError, "escapes"):
            checked_words(candidate, REVISION)
        candidate["instructions"][0]["control"]["kind"] = "jump_register"
        with self.assertRaisesRegex(ValueError, "resolved"):
            checked_words(candidate, REVISION)

    def test_refuses_target_revision_or_early_completion(self):
        candidate = program()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            checked_words(candidate, "f" * 40)
        candidate["instructions"][1]["word_u32"] = 0x73
        candidate["instructions"][1]["word_hex"] = "00000073"
        with self.assertRaisesRegex(ValueError, "early ECALL"):
            checked_words(candidate, REVISION)


if __name__ == "__main__":
    unittest.main()
