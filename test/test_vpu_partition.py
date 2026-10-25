"""Independent finite-domain checks for complete legal operand partitions."""
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from atlas_overlap_rules import Register
from atlas_vpu_partition import (operand_domain, relation, standalone_domain, preserves_inputs,
                                 require_preserved_inputs)
import atlas_derive_overlap as derive


def role(operand, write=False, count=2):
    return dict(operand=operand, write=write, registers=count, offset=0)


def footprints(roles):
    return tuple(Register(r["write"], ((2 if r["write"] else 0, i),), 4)
                 for r in roles for i in range(r["registers"]))


def registers(roles, values):
    return [values[r["operand"]] + r["offset"] + i
            for r in roles for i in range(r["registers"])]


def native_port_oracle(regs, plans):
    """Direct cycle/port interpretation, independent of pairwise constraints."""
    accesses = {}
    read_cache = set()
    for reg, plan in zip(regs, plans):
        for age, row in plan.requests:
            if not plan.write:
                if (age, reg, row) in read_cache:
                    continue
                read_cache.add((age, reg, row))
            bank, physical_row = reg % 32, 32 * (reg // 32) + row
            key = (age, plan.write, bank)
            if key in accesses or accesses.get((age, not plan.write, bank)) == physical_row:
                return False
            accesses[key] = physical_row
    return True


class VpuPartitionTests(unittest.TestCase):
    def check_complete(self, first, second):
        fa, fb = footprints(first), footprints(second)
        actual = operand_domain(first, second, fa, fb)
        expected = set()
        # Four operand groups need at most four distinguishable bank-pairs.
        # Both halves and singleton parities are concrete architectural values.
        choices = [range(0, 8, 2) if r["registers"] == 2 else range(8)
                   for r in first + second]
        choices = [tuple(c) + tuple(value + 32 for value in c) for c in choices]
        for values in itertools.product(*choices):
            av = dict(zip((r["operand"] for r in first), values))
            bv = dict(zip((r["operand"] for r in second), values[len(first):]))
            a, b = registers(first, av), registers(second, bv)
            if native_port_oracle(a, fa) and native_port_oracle(b, fb):
                expected.add(relation(a, b))
        self.assertEqual({key for key, _, _ in actual}, expected)
        self.assertEqual(len(actual), len(expected))
        for key, av, bv in actual:
            a, b = registers(first, av), registers(second, bv)
            self.assertTrue(native_port_oracle(a, fa))
            self.assertTrue(native_port_oracle(b, fb))
            self.assertEqual(key, relation(a, b))

    def test_complete_unary_pair_partition_against_concrete_exhaustion(self):
        self.check_complete([role("rs1"), role("rd", True)],
                            [role("rs1"), role("rd", True)])

    def test_complete_pack_unpack_partition_including_singleton_parity(self):
        self.check_complete([role("rs2"), role("rd", True, 1)],
                            [role("rs2", count=1), role("rd", True)])

    def test_singletons_support_xlu_including_odd_and_tier_aliases(self):
        self.check_complete([role("rs1", count=1), role("rd", True, 1)],
                            [role("rs1", count=1), role("rd", True, 1)])

    def test_multiple_joint_aliases_and_duplicate_sources_are_covered(self):
        roles = [role("rs1"), role("rs2"), role("rd", True)]
        rows = operand_domain(roles, roles, footprints(roles), footprints(roles))
        keys = {key for key, _, _ in rows}
        # Duplicate sources and in-place destinations are source-legal. This
        # relation also aliases several independent second-command operands.
        all_same = dict(rs1=0, rs2=0, rd=0)
        self.assertIn(relation(registers(roles, all_same), registers(roles, all_same)), keys)
        for _, values_a, values_b in rows:
            for values in (values_a, values_b):
                self.assertFalse(values["rs1"] % 32 == values["rs2"] % 32
                                 and values["rs1"] != values["rs2"])

    def test_missing_or_unknown_structure_fails_closed(self):
        roles = [role("rs1"), role("rd", True)]
        with self.assertRaisesRegex(ValueError, "requires guarded"):
            derive.mreg_domain("vpu.relu", "vpu.exp2")
        with self.assertRaisesRegex(ValueError, "footprints"):
            operand_domain(roles, roles, (), ())
        with self.assertRaisesRegex(ValueError, "unsupported operand partition role"):
            operand_domain([role("rs1", count=3)], roles, (), ())

    def test_pack_second_source_half_is_rejected_by_actual_row_collision(self):
        roles = [role("rs2"), role("rd", True, 1)]
        plans = (Register(False, tuple((age, age) for age in range(32)), 64),
                 Register(False, tuple((age + 32, age) for age in range(32)), 64),
                 Register(True, tuple((3 + 2 * row, row) for row in range(32)), 66))
        classes = {key for key, _ in standalone_domain(roles, plans)}
        invalid = registers(roles, dict(rs2=0, rd=1))
        self.assertNotIn(relation(invalid, invalid), classes)
        self.assertFalse(native_port_oracle(invalid, plans))
        self.assertFalse(preserves_inputs(roles, plans, dict(rs2=0, rd=1)))
        with self.assertRaisesRegex(ValueError, "before its first sampled read"):
            require_preserved_inputs(roles, plans, dict(rs2=0, rd=1))
        for _, values in standalone_domain(roles, plans):
            self.assertTrue(preserves_inputs(roles, plans, values))

    def test_static_input_preservation_uses_first_column_reduction_read(self):
        roles = [role("rs1", count=1), role("rd", True, 1)]
        plans = (Register(False, ((0, 0), (64, 0)), 128), Register(True, ((32, 0),), 130))
        self.assertTrue(preserves_inputs(roles, plans, dict(rs1=0, rd=0)))
        self.assertFalse(preserves_inputs(roles, (Register(False, ((40, 0),), 64), plans[1]),
                                         dict(rs1=0, rd=0)))


if __name__ == "__main__":
    unittest.main()
