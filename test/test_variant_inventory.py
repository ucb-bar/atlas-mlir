"""Frozen selected decoder modes and bounded evidence are kept separate."""

from __future__ import annotations

import copy
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_variant_inventory import (
    counts, load_inventory, validate_domains, validate_rows, validate_sources,
)
from test_dialect import scalar_variants, variants


def synthetic_sources(inventory: dict) -> tuple[str, str]:
    instructions = "\n".join(
        f'def {row["id"]} = BitPat("b{row["rtl_bitpat"]}")'
        for row in inventory["variants"]
    )
    decode = "\n".join(
        f'{row["id"]} -> List({row["rtl_decode_controls"]})'
        for row in inventory["variants"]
    )
    return instructions, decode


class VariantInventoryTest(unittest.TestCase):
    def test_finite_parameter_domains_keep_pc_units_separate(self) -> None:
        inventory = load_inventory()
        validate_domains(inventory)
        self.assertEqual(len(inventory["parameter_domains"]), 19)
        self.assertFalse(any(domain["reviewed"] for domain in inventory["parameter_domains"].values()))
        by_name = {row["id"]: row for row in inventory["variants"]}
        self.assertIn("branch_byte_displacement", by_name["BEQ"]["parameter_domains"])
        self.assertIn("jal_byte_displacement", by_name["JAL"]["parameter_domains"])
        self.assertIn("jalr_word_offset", by_name["JALR"]["parameter_domains"])

        changed = copy.deepcopy(inventory)
        changed["parameter_domains"]["bf16_even_pair_0_62"]["intervals"][0]["max"] = 63
        with self.assertRaisesRegex(ValueError, "invalid interval"):
            validate_domains(changed)
        changed = copy.deepcopy(inventory)
        changed["parameter_domains"]["mreg_0_63"]["evidence_sources"] = ["../unselected/MregFile.scala"]
        with self.assertRaisesRegex(ValueError, "invalid parameter evidence path"):
            validate_domains(changed)

    def test_declared_modes_match_typed_dialect_fixtures(self) -> None:
        inventory = load_inventory()
        by_name = {row["id"]: row for row in inventory["variants"]}
        fixtures = {name: (op, attrs) for op, attrs, name in variants() + scalar_variants()}
        self.assertEqual(len(by_name), 99)
        self.assertEqual(set(by_name), set(fixtures))
        self.assertEqual(by_name["ECALL"]["parameter_domains"], [])
        self.assertEqual(by_name["EBREAK"]["parameter_domains"], [])
        self.assertIn("e8m0_scale_reg_0_31", by_name["SELD"]["parameter_domains"])
        self.assertIn("e8m0_scale_reg_0_31", by_name["SELI"]["parameter_domains"])
        self.assertNotIn("scalar_gpr_0_31", by_name["SELI"]["parameter_domains"])
        for name, (op, attrs) in fixtures.items():
            with self.subTest(name=name):
                row = by_name[name]
                self.assertEqual(row["dialect_op"], f"atlas.{op}")
                self.assertEqual(row["mode_attrs"], {
                    key: value for key, value in attrs.items()
                    if key in ("kind", "direction", "format", "mode", "unit", "accumulate")
                })
        self.assertEqual(counts(inventory), {
            "required": 99, "software_admitted": 0, "represented": 99,
            "word_emitted": 99, "llvm_word_emitted": 99,
            "independent_semantic_test": 99, "standalone_core_executed": 99,
            "blocked": 99, "denominator": 99,
        })

    def test_census_check_rejects_pattern_decoder_and_requirement_drift(self) -> None:
        inventory = load_inventory()
        instructions, decode = synthetic_sources(inventory)
        validate_rows(inventory, instructions, decode)
        original_bits = inventory["variants"][0]["rtl_bitpat"]
        flipped = original_bits[:-1] + ("0" if original_bits[-1] == "1" else "1")
        changed_pattern = instructions.replace(
            'BitPat("b' + original_bits + '")', 'BitPat("b' + flipped + '")', 1)
        self.assertNotEqual(changed_pattern, instructions)
        with self.assertRaisesRegex(ValueError, "BitPat drift"):
            validate_rows(inventory, changed_pattern, decode)
        changed_decode = decode.replace("LSU_VLOAD", "LSU_VSTORE", 1)
        self.assertNotEqual(changed_decode, decode)
        with self.assertRaisesRegex(ValueError, "decoder control drift"):
            validate_rows(inventory, instructions, changed_decode)
        removed = copy.deepcopy(inventory)
        removed["variants"].pop()
        with self.assertRaisesRegex(ValueError, "99 frozen"):
            validate_rows(removed, instructions, decode)
        missing_domain = copy.deepcopy(inventory)
        next(row for row in missing_domain["variants"] if row["id"] == "SELI")["parameter_domains"] = []
        with self.assertRaisesRegex(ValueError, "variable encoding lacks parameter domain"):
            validate_rows(missing_domain, instructions, decode)
        added = decode + "\nNEW_OPCODE -> List(Y)\n"
        with self.assertRaisesRegex(ValueError, "required variant drift"):
            validate_rows(inventory, instructions, added)

    def test_selected_source_pins_and_model_identities(self) -> None:
        rtl = os.environ.get("ATLAS_RTL_ROOT")
        model = os.environ.get("ATLAS_MODEL_ROOT")
        if not rtl or not model:
            self.skipTest("set ATLAS_RTL_ROOT and ATLAS_MODEL_ROOT for source-bound census")
        inventory = load_inventory()
        validate_sources(inventory, pathlib.Path(rtl), pathlib.Path(model))


if __name__ == "__main__":
    unittest.main()
