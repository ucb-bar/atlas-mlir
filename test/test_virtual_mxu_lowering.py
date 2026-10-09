"""Explicit resident MXU chains reach checked machine streams and LLVM words."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import unittest

from test_virtual_lowering import BIN, emitted, lower, object_words, run, shorten_delay
from test_virtual_mxu_handles import (
    EXAMPLE, STATE, FP8, BF16, acc, load, reset, accumulate, readout,
)


def source(*body: str, final_state: str = "s3", tile: str = "y") -> str:
    return "\n".join([
        "module {",
        f"  func.func @resident() -> {STATE} attributes "
        "{atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2415923200 : i64} {",
        f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
        f'    %io1, %x = "atlas.virtual_input_fp8"(%io0) {{index = 0 : i32}} '
        f': ({STATE}) -> ({STATE}, {FP8})',
        f'    %io2, %w = "atlas.virtual_input_fp8"(%io1) {{index = 1 : i32}} '
        f': ({STATE}) -> ({STATE}, {FP8})',
        *body,
        f'    %out = "atlas.virtual_output_bf16"(%{final_state}, %{tile}) '
        f'{{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}',
        f"    return %out : {STATE}", "  }", "}",
    ])


def continuation(unit: int, *, reload_weight: bool = False) -> str:
    body = [load("io2", "s0", "weight", unit),
            reset("s0", "s1", "a0", unit=unit)]
    if reload_weight:
        body.append(load("s1", "refresh", "replacement", unit))
    body.extend([
        accumulate("refresh" if reload_weight else "s1", "s2", "a1", "a0",
                   "replacement" if reload_weight else "weight", unit),
        readout("s2", "s3", "y", "a1", unit),
    ])
    return source(*body)


def instructions(machine: str, *, allow_untimed: bool = False) -> list[dict]:
    exported = run("atlas-emit", machine, "--program-json", *(("--allow-untimed",) if allow_untimed else ()))
    if exported.returncode:
        raise AssertionError(exported.stderr)
    return json.loads(exported.stdout)["instructions"]


class VirtualMXULoweringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def checked(self, virtual: str, slots: int = 1) -> tuple[str, list[dict]]:
        untimed = lower(virtual, timed=False)
        self.assertIn('atlas.timing_state = "untimed"', untimed)
        self.assertNotIn('"atlas.delay"', untimed)
        machine = lower(virtual)
        self.assertIn("atlas.generated_from_virtual", machine)
        self.assertNotRegex(machine, r'(?m)^\s*%[^\n=]+\s*=\s*"atlas\.virtual_')
        for option in ("--verify-atlas-machine-stream",
                       "--verify-atlas-generated-schedule", "--verify-atlas-timing"):
            result = run("atlas-opt", machine, option)
            self.assertEqual(result.returncode, 0, result.stderr)
        entries = instructions(machine)
        self.assertEqual(tuple(entry["word_u32"] for entry in entries),
                         emitted(machine))
        for index, entry in enumerate(entries):
            operation = entry["operation"]
            if not operation.startswith("atlas.mxu_"):
                continue
            fields = entry["fields"]
            self.assertIn(fields["unit"], (0, 1))
            if operation == "atlas.mxu_matmul":
                self.assertLess(max(fields["weight_slot"], fields["acc_slot"]), slots)
                self.assertLess(fields["src"], 32)
            else:
                self.assertLess(fields["slot"], slots)
                if operation == "atlas.mxu_push":
                    self.assertEqual(fields["kind"], "weight_fp8")
                    self.assertLess(fields["src"], 32)
                else:
                    self.assertEqual(operation, "atlas.mxu_pop")
                    self.assertEqual(fields["format"], "bf16")
                    self.assertGreaterEqual(fields["dst"], 32)
                    self.assertEqual(fields["dst"] % 2, 0)
                    self.assertEqual(fields["scale_reg"], 0)
        return machine, entries

    def test_reset_only_matches_legacy_matmul_words_on_both_units(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                explicit = source(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    readout("s1", "s3", "y", "a0", unit),
                )
                legacy = source(
                    f'    %y = "atlas.virtual_mxu_matmul"(%x, %w) '
                    f'{{unit = {unit} : i32}} : ({FP8}, {FP8}) -> {BF16}',
                    final_state="io2",
                )
                machine, _ = self.checked(explicit)
                self.assertEqual(emitted(machine), emitted(lower(legacy)))

    def test_continuation_and_live_weight_reload_on_both_units(self) -> None:
        for unit in (0, 1):
            for refresh in (False, True):
                with self.subTest(unit=unit, refresh=refresh):
                    _, entries = self.checked(continuation(unit, reload_weight=refresh))
                    mxu = [entry for entry in entries
                           if entry["operation"].startswith("atlas.mxu_")]
                    expected = ["atlas.mxu_push", "atlas.mxu_matmul"]
                    if refresh:
                        expected.append("atlas.mxu_push")
                    expected.extend(["atlas.mxu_matmul", "atlas.mxu_pop"])
                    self.assertEqual([entry["operation"] for entry in mxu], expected)
                    self.assertTrue(all(entry["fields"]["unit"] == unit for entry in mxu))
                    flags = [entry["fields"]["accumulate"] for entry in mxu
                             if entry["operation"] == "atlas.mxu_matmul"]
                    self.assertEqual(flags, [False, True])

    def test_resident_weight_reused_across_completed_chains(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                _, entries = self.checked(source(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    readout("s1", "s2", "first", "a0", unit),
                    reset("s2", "s3", "a1", unit=unit),
                    readout("s3", "s4", "y", "a1", unit),
                    final_state="s4",
                ))
                self.assertEqual(sum(entry["operation"] == "atlas.mxu_push"
                                     for entry in entries), 1)
                self.assertEqual([entry["fields"]["accumulate"] for entry in entries
                                  if entry["operation"] == "atlas.mxu_matmul"],
                                 [False, False])

    def test_interleaved_units_lower_independently(self) -> None:
        _, entries = self.checked(EXAMPLE.read_text())
        for unit in (0, 1):
            mxu = [entry for entry in entries
                   if entry["operation"].startswith("atlas.mxu_")
                   and entry["fields"]["unit"] == unit]
            self.assertEqual([entry["operation"] for entry in mxu],
                             ["atlas.mxu_push", "atlas.mxu_matmul",
                              "atlas.mxu_matmul", "atlas.mxu_pop"])
            self.assertEqual([entry["fields"]["accumulate"] for entry in mxu
                              if entry["operation"] == "atlas.mxu_matmul"],
                             [False, True])

    def test_two_live_weights_and_accumulators_take_separate_slots(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                _, entries = self.checked(source(
                    load("io2", "s0", "w0", unit), load("s0", "s1", "w1", unit),
                    reset("s1", "r0", "a0", "w0", unit), reset("r0", "r1", "b0", "w1", unit),
                    accumulate("r1", "r2", "a1", "a0", "w1", unit),
                    readout("r2", "r3", "y", "a1", unit), readout("r3", "s3", "z", "b0", unit),
                ), slots=2)
                fields = [entry["fields"] for entry in entries
                          if entry["operation"].startswith("atlas.mxu_")]
                self.assertEqual([f["slot"] for f in fields[:2]], [0, 1])
                self.assertEqual([(f["weight_slot"], f["acc_slot"]) for f in fields[2:5]],
                                 [(0, 0), (1, 1), (1, 0)])
                self.assertEqual([f["slot"] for f in fields[5:]], [0, 1])

    def test_fp8_sources_can_be_reused_after_push_and_matmul(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                program = source(
                    load("io2", "s0", "weight", unit),
                    f'    %s1, %fresh = "atlas.virtual_input_fp8"(%s0) '
                    f'{{index = 2 : i32}} : ({STATE}) -> ({STATE}, {FP8})',
                    reset("s1", "s2", "a0", unit=unit),
                    f'    %s3, %second = "atlas.virtual_input_fp8"(%s2) '
                    f'{{index = 3 : i32}} : ({STATE}) -> ({STATE}, {FP8})',
                    readout("s3", "s4", "y", "a0", unit),
                    f'    %other = "atlas.virtual_mxu_matmul"(%fresh, %second) '
                    f'{{unit = {1 - unit} : i32}} : ({FP8}, {FP8}) -> {BF16}',
                    final_state="s4",
                ).replace("atlas.output_dram_base = 2415923200",
                          "atlas.output_dram_base = 2415927296")
                _, entries = self.checked(program)
                loads = [entry["fields"]["dst"] for entry in entries
                         if entry["operation"] == "atlas.vload"]
                mxu = [entry for entry in entries
                       if entry["operation"].startswith("atlas.mxu_")]
                # Tensor sources may be reused after transfer into the MXU.
                self.assertEqual(mxu[0]["fields"]["src"], loads[2])
                self.assertEqual(mxu[1]["fields"]["src"], loads[3])

    def test_bf16_readout_crosses_cfg_edge_without_resident_handles(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.checked(source(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    readout("s1", "s3", "y", "a0", unit),
                    f"    cf.br ^next(%s3, %y : {STATE}, {BF16})",
                    f"  ^next(%next_state: {STATE}, %next_tile: {BF16}):",
                    final_state="next_state", tile="next_tile",
                ))

    def test_legacy_other_unit_and_same_unit_reload_coexist(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                def legacy(name: str, target: int) -> str:
                    return (f'    %{name} = "atlas.virtual_mxu_matmul"(%x, %w) '
                            f'{{unit = {target} : i32}} : ({FP8}, {FP8}) -> {BF16}')

                _, entries = self.checked(source(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    legacy("other", 1 - unit),
                    readout("s1", "s2", "first", "a0", unit),
                    legacy("same", unit),
                    load("s2", "s3", "replacement", unit),
                    reset("s3", "s4", "a1", "replacement", unit),
                    readout("s4", "s5", "y", "a1", unit),
                    final_state="s5",
                ))
                self.assertEqual(sum(entry["operation"] == "atlas.mxu_matmul"
                                     for entry in entries), 4)

    def test_lowering_enforces_handle_and_state_verification(self) -> None:
        for unit in (0, 1):
            valid = continuation(unit)
            cases = [
                (valid.replace('(%s2, %a1)', '(%s2, %a0)'), "stale accumulator handle"),
                (valid.replace('(%s1, %x, %weight, %a0)',
                               '(%s0, %x, %weight, %a0)'), "nonlinear virtual state chain"),
                (source(load("io2", "s0", "weight", unit),
                        reset("s0", "s1", "a0", unit=unit),
                        load("s1", "r0", "replacement", unit),
                        load("r0", "r1", "third", unit),
                        accumulate("r1", "s2", "a1", "a0", "weight", unit),
                        accumulate("s2", "r2", "a2", "a1", "replacement", unit),
                        readout("r2", "s3", "y", "a2", unit)),
                 "cannot load a weight on a unit with 2 live weights"),
                (valid.replace(acc(unit), acc(1 - unit)), "unit"),
                (source(load("io2", "s0", "weight", unit),
                        reset("s0", "s1", "a0", unit=unit),
                        reset("s1", "r0", "b0", unit=unit),
                        reset("r0", "s2", "c0", unit=unit),
                        readout("s2", "s3", "y", "c0", unit)),
                 "cannot reset a unit with 2 live accumulators"),
                (source(load("io2", "s0", "weight", unit),
                        f"    cf.br ^next(%s0 : {STATE})",
                        f"  ^next(%next_state: {STATE}):",
                        reset("next_state", "s1", "a0", unit=unit),
                        readout("s1", "s3", "y", "a0", unit)),
                 "virtual MXU handles cannot cross CFG blocks"),
            ]
            # Drop the readout while keeping a well-typed BF16 output producer.
            dropped = valid.replace(readout("s2", "s3", "y", "a1", unit),
                                    f'    %y = "atlas.virtual_mxu_matmul"(%x, %w) '
                                    f'{{unit = {1 - unit} : i32}} : ({FP8}, {FP8}) -> {BF16}')
            cases.append((dropped.replace('(%s3, %y)', '(%s2, %y)'),
                          "must read out the live MXU accumulator before block exit"))
            for changed, expected in cases:
                with self.subTest(unit=unit, expected=expected):
                    self.assertNotEqual(changed, valid)
                    result = run("atlas-opt", changed, "--lower-atlas-virtual-to-machine")
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(expected, result.stderr)

    def test_generated_mxu_waits_are_enforced(self) -> None:
        machine, _ = self.checked(continuation(0, reload_weight=True))
        for producer in ("atlas.mxu_matmul", "atlas.mxu_pop"):
            changed = shorten_delay(machine, after=producer)
            self.assertNotEqual(changed, machine)
            for tool, options in (("atlas-emit", ()),
                                  ("atlas-opt", ("--verify-atlas-timing",))):
                with self.subTest(producer=producer, tool=tool):
                    result = run(tool, changed, *options)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("insufficient issue spacing", result.stderr)

    def test_structured_llvm_handoff_preserves_continuation_flags_and_words(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                machine, _ = self.checked(continuation(unit, reload_weight=True))
                structured = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
                self.assertEqual(structured.returncode, 0, structured.stderr)
                self.assertIn("accumulate = false", structured.stdout)
                self.assertIn("accumulate = true", structured.stdout)
                finalized = run("atlas-opt", structured.stdout, "--finalize-atlas-llvm-calls")
                self.assertEqual(finalized.returncode, 0, finalized.stderr)
                direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
                self.assertEqual(direct.returncode, 0, direct.stderr)
                self.assertEqual(finalized.stdout, direct.stdout)
                self.assertIn("llvm.inline_asm", finalized.stdout)

    def test_llvm_object_preserves_explicit_generated_words(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                machine = lower(continuation(unit, reload_weight=True))
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_assembler_matches_generated_mxu_encoding(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location(
            "selected_virtual_mxu_assembler", Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        for unit in (0, 1):
            for entry in instructions(lower(continuation(unit, reload_weight=True))):
                fields = entry["fields"]
                operation = entry["operation"]
                if operation == "atlas.mxu_push":
                    name, args = "VMATPUSH_W", (0, fields["src"])
                elif operation == "atlas.mxu_matmul":
                    name = "VMATMUL_ACC" if fields["accumulate"] else "VMATMUL"
                    args = (0, fields["src"], 0)
                elif operation == "atlas.mxu_pop":
                    name, args = "VMATPOP_BF16", (fields["dst"], 0)
                else:
                    continue
                with self.subTest(unit=unit, operation=name):
                    expected = getattr(assembler, f"{name}_MXU{unit}")(*args)
                    self.assertEqual(entry["word_u32"], expected)


if __name__ == "__main__":
    unittest.main()
