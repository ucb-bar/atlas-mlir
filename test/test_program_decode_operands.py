"""Known instruction words exercise operand decoding without the emitter."""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.check_program_decode import _operands, decode_binary
from tools.check_variant_inventory import INVENTORY


class ProgramDecodeOperandsTest(unittest.TestCase):
    def test_branch_uses_signed_split_immediate(self) -> None:
        self.assertEqual(
            _operands("atlas.branch", 0xFCE6C4E3, {"kind": "blt"}),
            {"lhs": 13, "rhs": 14, "offset_bytes": -56},
        )

    def test_vector_pack_and_scalar_store_fields(self) -> None:
        self.assertEqual(
            _operands("atlas.vpu_pack", 0x89106057, {"direction": "bf16_to_fp8"}),
            {"dst": 0, "scale_reg": 3, "src": 34},
        )
        self.assertEqual(
            _operands("atlas.scalar_store", 0x01062023, {"kind": "sw"}),
            {"src": 16, "base": 12, "offset": 0},
        )

    def test_delay_count_is_unsigned(self) -> None:
        self.assertEqual(_operands("atlas.delay", 0xFFF01067, {}), {"cycles": 4095})

    def test_binary_only_decode_uses_selected_modes_and_operand_bits(self) -> None:
        inventory = json.loads(INVENTORY.read_text())
        words = (0x0000007F, 0x40000113, 0x00030007, 0x89106057, 0xFCE6C4E3)
        decoded = decode_binary(inventory, struct.pack("<5I", *words))
        self.assertEqual(decoded["word_count"], 5)
        self.assertEqual(
            [(row["mode_id"], row["fields"]) for row in decoded["instructions"]],
            [
                ("DMA_CONFIG_ANY", {"base_reg": 0, "channel": 0}),
                ("ADDI", {"kind": "addi", "dst": 2, "src": 0, "immediate": 1024}),
                ("VLOAD", {"format": "raw", "base": 6, "offset": 0, "dst": 0}),
                ("VFP8PACK", {"direction": "bf16_to_fp8", "dst": 0, "scale_reg": 3, "src": 34}),
                ("BLT", {"kind": "blt", "lhs": 13, "rhs": 14, "offset_bytes": -56}),
            ],
        )
        self.assertEqual([row["word_index"] for row in decoded["instructions"]], list(range(5)))
        with self.assertRaisesRegex(ValueError, "partial instruction word"):
            decode_binary(inventory, b"\x7f")
        with self.assertRaisesRegex(ValueError, "expected one selected RTL decode mode"):
            decode_binary(inventory, struct.pack("<I", 0xFFFFFFFF))

    def test_every_selected_mode_has_a_binary_decode_route(self) -> None:
        inventory = json.loads(INVENTORY.read_text())
        for mode in inventory["variants"]:
            # Fill unbound BitPat positions with zero. This checks route
            # coverage, not the legality of every operand combination.
            word = int(mode["rtl_bitpat"].replace("?", "0"), 2)
            with self.subTest(mode=mode["id"]):
                decoded = decode_binary(inventory, struct.pack("<I", word))
                self.assertEqual(decoded["instructions"][0]["mode_id"], mode["id"])


if __name__ == "__main__":
    unittest.main()
