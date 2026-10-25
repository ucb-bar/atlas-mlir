"""Independent finite-domain checks for complete legal operand partitions."""
import itertools
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from atlas_overlap_rules import Register
from atlas_vpu_partition import (relation, standalone_domain, preserves_inputs,
                                 require_preserved_inputs)


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
    def test_standalone_classes_match_concrete_native_port_oracle(self):
        for roles in ([role("rs1"), role("rd", True)],
                      [role("rs2"), role("rd", True, 1)],
                      [role("rs2", count=1), role("rd", True)],
                      [role("rs1", count=1), role("rd", True, 1)],
                      [role("rs1"), role("rs2"), role("rd", True)]):
            with self.subTest(roles=roles):
                plans = footprints(roles)
                actual = standalone_domain(roles, plans)
                # Three operand groups need at most three bank-pairs. Cover
                # both logical halves and every singleton lane independently.
                choices = [range(0, 6, 2) if r["registers"] == 2 else range(6) for r in roles]
                choices = [tuple(c) + tuple(value + 32 for value in c) for c in choices]
                expected = set()
                for bases in itertools.product(*choices):
                    values = dict(zip((r["operand"] for r in roles), bases))
                    regs = registers(roles, values)
                    if native_port_oracle(regs, plans):
                        expected.add(relation(regs, regs))
                self.assertEqual({key for key, _ in actual}, expected)
                self.assertEqual(len(actual), len(expected))
                for key, values in actual:
                    regs = registers(roles, values)
                    self.assertTrue(native_port_oracle(regs, plans))
                    self.assertEqual(key, relation(regs, regs))

    def test_missing_or_unknown_structure_fails_closed(self):
        roles = [role("rs1"), role("rd", True)]
        with self.assertRaisesRegex(ValueError, "footprints"):
            standalone_domain(roles, ())
        with self.assertRaisesRegex(ValueError, "unsupported operand partition role"):
            standalone_domain([role("rs1", count=3)], ())

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
