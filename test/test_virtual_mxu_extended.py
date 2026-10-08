"""Seeded MXU accumulators and immutable FP8 readout scales reach machine words."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

from test_virtual_lowering import BIN, ROOT, emitted, lower, object_words, run
from test_virtual_mxu_handles import STATE, FP8, BF16, acc, load, accumulate, readout
from test_virtual_mxu_lowering import instructions


SCALE = "!atlas.virtual_scale"
EXAMPLE = ROOT / "test/examples/virtual_mxu_seeded_fp8.mlir"


def scale(name: str = "scale", code: int = 129) -> str:
    return (f'    %{name} = "atlas.virtual_scale_constant"() '
            f'{{code = {code} : i32}} : () -> {SCALE}')


def seed(before: str, after: str, handle: str, unit: int = 0,
         fmt: str = "fp8", tile: str | None = None) -> str:
    ty = FP8 if fmt == "fp8" else BF16
    tile = tile or ("x" if fmt == "fp8" else "seed")
    return (f'    %{after}, %{handle} = "atlas.virtual_mxu_load_acc_{fmt}"'
            f'(%{before}, %{tile}) {{unit = {unit} : i32}} : ({STATE}, {ty}) '
            f'-> ({STATE}, {acc(unit)})')


def fp8_readout(before: str, after: str, tile: str, handle: str,
                unit: int = 0, scale_name: str = "scale") -> str:
    return (f'    %{after}, %{tile} = "atlas.virtual_mxu_readout_fp8"'
            f'(%{before}, %{handle}, %{scale_name}) : '
            f'({STATE}, {acc(unit)}, {SCALE}) -> ({STATE}, {FP8})')


def program(*body: str, final_state: str = "s3", tile: str = "y") -> str:
    return "\n".join([
        "module {",
        f"  func.func @seeded() -> {STATE} attributes "
        "{atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2415935488 : i64} {",
        f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
        f'    %io1, %seed = "atlas.virtual_input_bf16"(%io0) '
        f'{{index = 2 : i32}} : ({STATE}) -> ({STATE}, {BF16})',
        f'    %io2, %x = "atlas.virtual_input_fp8"(%io1) '
        f'{{index = 0 : i32}} : ({STATE}) -> ({STATE}, {FP8})',
        f'    %io3, %w = "atlas.virtual_input_fp8"(%io2) '
        f'{{index = 1 : i32}} : ({STATE}) -> ({STATE}, {FP8})',
        *body,
        f'    %out = "atlas.virtual_output_bf16"(%{final_state}, %{tile}) '
        f'{{index = 0 : i32}} : ({STATE}, {BF16}) -> {STATE}',
        f"    return %out : {STATE}", "  }", "}",
    ])


def seeded_chain(unit: int = 0, fmt: str = "fp8", code: int = 129) -> str:
    return program(
        scale(code=code), load("io3", "s0", "weight", unit),
        seed("s0", "s1", "a0", unit, fmt),
        accumulate("s1", "s2", "a1", "a0", unit=unit),
        fp8_readout("s2", "s3", "packed", "a1", unit),
        seed("s3", "s4", "a2", unit, tile="packed"),
        accumulate("s4", "s5", "a3", "a2", unit=unit),
        readout("s5", "s6", "y", "a3", unit), final_state="s6",
    )


class VirtualMXUExtendedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def checked(self, virtual: str) -> tuple[str, list[dict]]:
        verified = run("atlas-opt", virtual, "--verify-atlas-virtual-stream")
        self.assertEqual(verified.returncode, 0, verified.stderr)
        roundtrip = run("atlas-opt", verified.stdout, "--verify-atlas-virtual-stream")
        self.assertEqual(roundtrip.returncode, 0, roundtrip.stderr)
        self.assertEqual(roundtrip.stdout, verified.stdout)
        machine = lower(virtual)
        self.assertIn("atlas.generated_from_virtual", machine)
        self.assertNotIn("atlas.virtual_", machine)
        for option in ("--verify-atlas-machine-stream", "--verify-atlas-generated-schedule"):
            result = run("atlas-opt", machine, option)
            self.assertEqual(result.returncode, 0, result.stderr)
        entries = instructions(machine)
        self.assertEqual(tuple(entry["word_u32"] for entry in entries), emitted(machine))
        for index, entry in enumerate(entries):
            op, fields = entry["operation"], entry["fields"]
            if not op.startswith("atlas.mxu_"):
                continue
            self.assertIn(fields["unit"], (0, 1))
            if op == "atlas.mxu_matmul":
                self.assertEqual((fields["weight_slot"], fields["acc_slot"]), (0, 0))
                self.assertLess(fields["src"], 62)
                reason = "mxu_matmul_completion"
            elif op == "atlas.mxu_push":
                self.assertEqual(fields["slot"], 0)
                if fields["kind"] == "acc_bf16":
                    self.assertLess(fields["src"], 62)
                    self.assertEqual(fields["src"] % 2, 0)
                else:
                    self.assertIn(fields["kind"], ("weight_fp8", "acc_fp8"))
                    self.assertLess(fields["src"], 62)
                reason = ("mxu_weight_completion" if fields["kind"] == "weight_fp8"
                          else "mxu_accumulator_completion")
            else:
                self.assertEqual(op, "atlas.mxu_pop")
                self.assertEqual(fields["slot"], 0)
                if fields["format"] == "fp8":
                    self.assertLess(fields["dst"], 62)
                    self.assertEqual(fields["scale_reg"], 3)
                    preceding = entries[index - 1]
                    self.assertEqual(preceding["operation"], "atlas.scalar_load")
                    self.assertEqual(preceding["fields"]["kind"], "seli")
                    self.assertEqual(preceding["fields"]["dst"], 3)
                else:
                    self.assertEqual(fields["format"], "bf16")
                    self.assertLess(fields["dst"], 62)
                    self.assertEqual(fields["dst"] % 2, 0)
                    self.assertEqual(fields["scale_reg"], 0)
                reason = "mxu_readout_completion"
            following = entries[index + 1]
            self.assertEqual(following["operation"], "atlas.delay")
            self.assertEqual(following["fields"]["cycles"], 256)
            self.assertEqual(following["fields"]["atlas.delay_reason"], reason)
        return machine, entries

    def rejected(self, virtual: str, expected: str = "") -> None:
        for option in ("--verify-atlas-virtual-stream", "--lower-atlas-virtual-to-machine"):
            result = run("atlas-opt", virtual, option)
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout, "")
            if expected:
                self.assertIn(expected, result.stderr)

    def test_seed_continue_fp8_readout_and_reseed_preserve_weight_on_both_units(self) -> None:
        for unit in (0, 1):
            for fmt in ("fp8", "bf16"):
                with self.subTest(unit=unit, seed=fmt):
                    _, entries = self.checked(seeded_chain(unit, fmt))
                    mxu = [entry for entry in entries
                           if entry["operation"].startswith("atlas.mxu_")]
                    self.assertEqual([entry["operation"] for entry in mxu], [
                        "atlas.mxu_push", "atlas.mxu_push", "atlas.mxu_matmul",
                        "atlas.mxu_pop", "atlas.mxu_push", "atlas.mxu_matmul", "atlas.mxu_pop",
                    ])
                    self.assertEqual([entry["fields"]["kind"] for entry in mxu
                                      if entry["operation"] == "atlas.mxu_push"],
                                     ["weight_fp8", f"acc_{fmt}", "acc_fp8"])
                    self.assertTrue(all(entry["fields"]["unit"] == unit for entry in mxu))
                    self.assertTrue(all(entry["fields"]["accumulate"] for entry in mxu
                                        if entry["operation"] == "atlas.mxu_matmul"))

    def test_example_and_fp8_readout_to_legacy_matmul(self) -> None:
        self.checked(EXAMPLE.read_text())
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.checked(program(
                    scale(), seed("io3", "s0", "a0", unit, "bf16"),
                    fp8_readout("s0", "s3", "packed", "a0", unit),
                    f'    %y = "atlas.virtual_mxu_matmul"(%packed, %w) '
                    f'{{unit = {unit} : i32}} : ({FP8}, {FP8}) -> {BF16}',
                ))

    def test_full_scale_code_range_materializes_at_readout(self) -> None:
        for unit in (0, 1):
            for code in (0, 127, 255):
                with self.subTest(unit=unit, code=code):
                    _, entries = self.checked(seeded_chain(unit, code=code))
                    pop = next(i for i, entry in enumerate(entries)
                               if entry["operation"] == "atlas.mxu_pop"
                               and entry["fields"]["format"] == "fp8")
                    self.assertEqual(entries[pop - 1]["fields"]["offset"], code)

    def test_scale_is_rematerialized_after_pack_clobbers_e3(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                _, entries = self.checked(program(
                    scale(code=131), load("io3", "s0", "weight", unit),
                    seed("s0", "s1", "a0", unit, "bf16"),
                    fp8_readout("s1", "s2", "first", "a0", unit),
                    seed("s2", "s3", "a1", unit, tile="first"),
                    f'    %fresh = "atlas.virtual_pack_fp8"(%seed) '
                    f'{{scale_code = 127 : i32}} : ({BF16}) -> {FP8}',
                    fp8_readout("s3", "s4", "second", "a1", unit),
                    f'    %y = "atlas.virtual_mxu_matmul"(%second, %fresh) '
                    f'{{unit = {unit} : i32}} : ({FP8}, {FP8}) -> {BF16}',
                    final_state="s4",
                ))
                pops = [i for i, entry in enumerate(entries)
                        if entry["operation"] == "atlas.mxu_pop"
                        and entry["fields"]["format"] == "fp8"]
                self.assertEqual(len(pops), 2)
                self.assertEqual([entries[i - 1]["fields"]["offset"] for i in pops], [131, 131])
                self.assertTrue(any(entry["operation"] == "atlas.scalar_load"
                                    and entry["fields"]["kind"] == "seli"
                                    and entry["fields"]["dst"] == 3
                                    and entry["fields"]["offset"] == 127
                                    for entry in entries[pops[0]:pops[1]]))

    def test_constant_scale_can_dominate_a_different_block(self) -> None:
        self.checked(program(
            scale(code=128),
            f"    cf.br ^next(%io3 : {STATE})",
            f"  ^next(%next_state: {STATE}):",
            seed("next_state", "s0", "a0"),
            fp8_readout("s0", "s3", "packed", "a0"),
            f'    %y = "atlas.virtual_mxu_matmul"(%packed, %w) '
            f'{{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}',
        ))

    def test_readout_type_checks_run_without_stream_verification(self) -> None:
        source = f'''module {{
  func.func @readout(%state: {STATE}, %acc: {acc(0)}, %scale: {SCALE}) -> ({STATE}, {FP8}) {{
    %next, %tile = "atlas.virtual_mxu_readout_fp8"(%state, %acc, %scale) : ({STATE}, {acc(0)}, {SCALE}) -> ({STATE}, {FP8})
    return %next, %tile : {STATE}, {FP8}
  }}
}}'''
        valid = run("atlas-opt", source)
        self.assertEqual(valid.returncode, 0, valid.stderr)
        for changed, expected in (
            (source.replace(SCALE, "i32"), "operand #2 must be"),
            (source.replace(acc(0), acc(2)), "MXU unit must be in [0, 1]"),
        ):
            with self.subTest(expected=expected):
                result = run("atlas-opt", changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_invalid_codes_units_and_operand_types_are_rejected(self) -> None:
        valid = seeded_chain()
        cases = [valid.replace("code = 129 : i32", f"code = {code} : i32")
                 for code in (-1, 256)]
        cases.extend([
            valid.replace("code = 129 : i32", "code = 129 : i64"),
            valid.replace("unit = 0 : i32", "unit = 2 : i32"),
            valid.replace(acc(0), acc(1)),
            valid.replace(seed("s0", "s1", "a0"),
                          seed("s0", "s1", "a0").replace(f"{STATE}, {FP8}", f"{STATE}, {BF16}")),
            seeded_chain(fmt="bf16").replace(seed("s0", "s1", "a0", fmt="bf16"),
                seed("s0", "s1", "a0", fmt="bf16").replace(f"{STATE}, {BF16}", f"{STATE}, {FP8}")),
            valid.replace(f"{acc(0)}, {SCALE})", f"{acc(0)}, {FP8})"),
            valid.replace(fp8_readout("s2", "s3", "packed", "a1"),
                          fp8_readout("s2", "s3", "packed", "a1").replace(
                              f"-> ({STATE}, {FP8})", f"-> ({STATE}, {BF16})")),
        ])
        for index, changed in enumerate(cases):
            with self.subTest(case=index):
                self.assertNotEqual(changed, valid)
                self.rejected(changed)

    def test_scale_block_arguments_and_nonlinear_seed_state_are_rejected(self) -> None:
        self.rejected(program(
            scale(),
            f"    cf.br ^next(%io3, %scale : {STATE}, {SCALE})",
            f"  ^next(%next_state: {STATE}, %next_scale: {SCALE}):",
            seed("next_state", "s0", "a0"),
            fp8_readout("s0", "s3", "packed", "a0", scale_name="next_scale"),
            f'    %y = "atlas.virtual_mxu_matmul"(%packed, %w) '
            f'{{unit = 0 : i32}} : ({FP8}, {FP8}) -> {BF16}',
        ))
        for fmt in ("fp8", "bf16"):
            with self.subTest(seed=fmt):
                self.rejected(seeded_chain(fmt=fmt).replace(
                    seed("s0", "s1", "a0", fmt=fmt),
                    seed("io3", "s1", "a0", fmt=fmt)),
                    "nonlinear virtual state chain")

    def test_stale_and_live_accumulator_seeding_are_rejected(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                valid = seeded_chain(unit)
                self.rejected(valid.replace(f"(%s2, %a1, %scale)",
                                            f"(%s2, %a0, %scale)"), "stale accumulator handle")
                for fmt in ("fp8", "bf16"):
                    self.rejected(program(
                        seed("io3", "s0", "a0", unit),
                        seed("s0", "s1", "a1", unit, fmt),
                        readout("s1", "s3", "y", "a1", unit),
                    ), "live accumulator")
                self.rejected(program(
                    scale(), seed("io3", "s0", "a0", unit),
                    fp8_readout("s0", "s1", "packed", "a0", unit),
                    fp8_readout("s1", "s2", "again", "a0", unit),
                    f'    %y = "atlas.virtual_mxu_matmul"(%packed, %w) '
                    f'{{unit = {unit} : i32}} : ({FP8}, {FP8}) -> {BF16}',
                    final_state="s2",
                ), "stale accumulator handle")

    def test_generated_accumulator_load_wait_is_enforced(self) -> None:
        machine, _ = self.checked(seeded_chain(fmt="bf16"))
        lines = machine.splitlines()
        index = next(i for i, line in enumerate(lines)
                     if 'atlas.delay_reason = "mxu_accumulator_completion"' in line)
        lines[index] = lines[index].replace("cycles = 256 : i32", "cycles = 1 : i32")
        changed = "\n".join(lines)
        for tool, options in (("atlas-emit", ()),
                              ("atlas-opt", ("--verify-atlas-generated-schedule",))):
            with self.subTest(tool=tool):
                result = run(tool, changed, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("DELAY >= 256", result.stderr)

    def test_structured_llvm_and_object_preserve_generated_words(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                machine, _ = self.checked(seeded_chain(unit, "bf16"))
                structured = run("atlas-opt", machine, "--convert-atlas-to-llvm-calls")
                self.assertEqual(structured.returncode, 0, structured.stderr)
                self.assertIn('kind = "acc_bf16"', structured.stdout)
                self.assertIn('kind = "acc_fp8"', structured.stdout)
                self.assertIn('format = "fp8"', structured.stdout)
                finalized = run("atlas-opt", structured.stdout, "--finalize-atlas-llvm-calls")
                self.assertEqual(finalized.returncode, 0, finalized.stderr)
                direct = run("atlas-opt", machine, "--convert-atlas-to-llvm")
                self.assertEqual(direct.returncode, 0, direct.stderr)
                self.assertEqual(finalized.stdout, direct.stdout)
                self.assertEqual(object_words(machine), emitted(machine))

    def test_selected_assembler_matches_seed_and_fp8_readout_encodings(self) -> None:
        root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for selected assembler")
        spec = importlib.util.spec_from_file_location("selected_extended_mxu_assembler",
                                                     Path(root) / "assembler.py")
        assert spec and spec.loader
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        for unit in (0, 1):
            for fmt in ("fp8", "bf16"):
                for entry in instructions(lower(seeded_chain(unit, fmt))):
                    fields, op = entry["fields"], entry["operation"]
                    if op == "atlas.mxu_push" and fields["kind"].startswith("acc_"):
                        name = "VMATPUSH_AFP8" if fields["kind"] == "acc_fp8" else "VMATPUSH_ABF16"
                        args = (fields["slot"], fields["src"])
                    elif op == "atlas.mxu_pop" and fields["format"] == "fp8":
                        name, args = "VMATPOP_FP8", (fields["dst"], fields["scale_reg"], fields["slot"])
                    else:
                        continue
                    with self.subTest(unit=unit, seed=fmt, operation=name):
                        self.assertEqual(entry["word_u32"],
                                         getattr(assembler, f"{name}_MXU{unit}")(*args))


if __name__ == "__main__":
    unittest.main()
