"""Resident MXU handles are checked virtual SSA before explicit lowering."""

from __future__ import annotations

import unittest

from test_virtual_ssa import BIN, ROOT, run


EXAMPLE = ROOT / "test/examples/virtual_mxu_accumulation.mlir"
STATE = "!atlas.virtual_state"
FP8 = "!atlas.virtual_fp8"
BF16 = "!atlas.virtual_bf16"


def weight(unit: int) -> str:
    return f"!atlas.virtual_mxu_weight<{unit}>"


def acc(unit: int) -> str:
    return f"!atlas.virtual_mxu_acc<{unit}>"


def load(before: str, after: str, handle: str, unit: int = 0) -> str:
    return (f'    %{after}, %{handle} = "atlas.virtual_mxu_load_weight"'
            f'(%{before}, %w) {{unit = {unit} : i32}} : ({STATE}, {FP8}) '
            f'-> ({STATE}, {weight(unit)})')


def reset(before: str, after: str, result: str, handle: str = "weight",
          unit: int = 0) -> str:
    return (f'    %{after}, %{result} = "atlas.virtual_mxu_reset"'
            f'(%{before}, %x, %{handle}) : ({STATE}, {FP8}, {weight(unit)}) '
            f'-> ({STATE}, {acc(unit)})')


def accumulate(before: str, after: str, result: str, previous: str,
               handle: str = "weight", unit: int = 0) -> str:
    return (f'    %{after}, %{result} = "atlas.virtual_mxu_accumulate"'
            f'(%{before}, %x, %{handle}, %{previous}) : '
            f'({STATE}, {FP8}, {weight(unit)}, {acc(unit)}) '
            f'-> ({STATE}, {acc(unit)})')


def readout(before: str, after: str, tile: str, handle: str,
            unit: int = 0) -> str:
    return (f'    %{after}, %{tile} = "atlas.virtual_mxu_readout_bf16"'
            f'(%{before}, %{handle}) : ({STATE}, {acc(unit)}) '
            f'-> ({STATE}, {BF16})')


def program(*body: str, final_state: str = "io2", module_scope: bool = False) -> str:
    prefix = [
        f'    %io0 = "atlas.virtual_start"() : () -> {STATE}',
        f'    %seed_state, %seed = "atlas.virtual_input_bf16"(%io0) '
        f'{{index = 2 : i32}} : ({STATE}) -> ({STATE}, {BF16})',
        f'    %io1, %x = "atlas.virtual_input_fp8"(%seed_state) {{index = 0 : i32}} '
        f': ({STATE}) -> ({STATE}, {FP8})',
        f'    %io2, %w = "atlas.virtual_input_fp8"(%io1) {{index = 1 : i32}} '
        f': ({STATE}) -> ({STATE}, {FP8})',
    ]
    output = (f'    %output_state = "atlas.virtual_output_bf16"'
              f'(%{final_state}, %seed) {{index = 0 : i32}} '
              f': ({STATE}, {BF16}) -> {STATE}')
    if module_scope:
        return "\n".join(["module {", *prefix, *body, output, "}"])
    return "\n".join([
        "module {",
        f"  func.func @handles() -> {STATE} attributes "
        "{atlas.input_dram_base = 2415919104 : i64, "
        "atlas.output_dram_base = 2415923200 : i64} {",
        *prefix, *body, output, f"    return %output_state : {STATE}", "  }", "}",
    ])


def chain(unit: int = 0, module_scope: bool = False) -> str:
    return program(
        load("io2", "s0", "weight", unit),
        reset("s0", "s1", "a0", unit=unit),
        accumulate("s1", "s2", "a1", "a0", unit=unit),
        readout("s2", "s3", "y", "a1", unit),
        final_state="s3", module_scope=module_scope,
    )


class VirtualMXUHandleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue((BIN / "atlas-opt").is_file(), "build atlas-opt first")
        self.assertTrue((BIN / "atlas-emit").is_file(), "build atlas-emit first")

    def accepted(self, source: str) -> str:
        result = run("atlas-opt", source, "--verify-atlas-virtual-stream")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def rejected(self, source: str, expected: str = "", *, stream: bool = True) -> str:
        options = ("--verify-atlas-virtual-stream",) if stream else ()
        result = run("atlas-opt", source, *options)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr)
        if expected:
            self.assertIn(expected, result.stderr)
        return result.stderr

    def test_two_units_interleave_and_round_trip(self) -> None:
        first = self.accepted(EXAMPLE.read_text())
        self.assertEqual(first, self.accepted(first))
        for unit in (0, 1):
            self.assertIn(weight(unit), first)
            self.assertIn(acc(unit), first)
        self.assertEqual(first.count('"atlas.virtual_mxu_accumulate"'), 2)
        self.assertEqual(first.count('"atlas.virtual_mxu_readout_bf16"'), 2)

    def test_module_and_function_scope_round_trip_on_both_units(self) -> None:
        for unit in (0, 1):
            for module_scope in (False, True):
                with self.subTest(unit=unit, module_scope=module_scope):
                    first = self.accepted(chain(unit, module_scope))
                    self.assertEqual(first, self.accepted(first))

    def test_weight_reuse_and_reload_after_completed_chains(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.accepted(program(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    readout("s1", "s2", "y0", "a0", unit),
                    reset("s2", "s3", "a1", unit=unit),
                    readout("s3", "s4", "y1", "a1", unit),
                    load("s4", "s5", "newweight", unit),
                    reset("s5", "s6", "a2", "newweight", unit),
                    readout("s6", "s7", "y2", "a2", unit),
                    final_state="s7",
                ))

    def test_weight_refresh_during_live_accumulation(self) -> None:
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.accepted(program(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    load("s1", "s2", "replacement", unit),
                    accumulate("s2", "s3", "a1", "a0", "replacement", unit),
                    readout("s3", "s4", "y", "a1", unit),
                    final_state="s4",
                ))
                # Two weights are resident at once: the first stays usable.
                self.accepted(program(
                    load("io2", "s0", "weight", unit),
                    reset("s0", "s1", "a0", unit=unit),
                    load("s1", "s2", "replacement", unit),
                    accumulate("s2", "s3", "a1", "a0", unit=unit),
                    readout("s3", "s4", "y", "a1", unit),
                    final_state="s4",
                ))

    def test_handle_parameters_and_operation_unit_types(self) -> None:
        source = chain()
        invalid = [
            source.replace(weight(0), weight(2)),
            source.replace(acc(0), acc(2)),
            source.replace(weight(0), "!atlas.virtual_mxu_weight<-1>"),
            source.replace(acc(0), "!atlas.virtual_mxu_acc<-1>"),
            source.replace("unit = 0 : i32", "unit = 1 : i32"),
            source.replace("unit = 0 : i32", "unit = 2 : i32"),
            source.replace("unit = 0 : i32", "unit = -1 : i32"),
            source.replace(acc(0), acc(1)),
            source.replace(f"-> ({STATE}, {acc(0)})",
                           f"-> ({STATE}, {acc(1)})", 1),
            source.replace(f"-> ({STATE}, {BF16})",
                           f"-> ({STATE}, {FP8})"),
            source.replace(STATE, "!atlas.state"),
        ]
        for index, changed in enumerate(invalid):
            with self.subTest(case=index):
                self.rejected(changed, stream=False)

    def test_stale_and_forked_accumulators_are_rejected(self) -> None:
        source = chain()
        self.rejected(source.replace('(%s2, %a1)', '(%s2, %a0)'),
                      "stale accumulator handle")
        for operation in (
            accumulate("s2", "s3", "a2", "a0"),
            readout("s2", "s3", "y", "a0"),
        ):
            with self.subTest(operation=operation):
                self.rejected(program(
                    load("io2", "s0", "weight"),
                    reset("s0", "s1", "a0"),
                    accumulate("s1", "s2", "a1", "a0"),
                    operation, final_state="s3",
                ), "stale accumulator handle")
        self.rejected(program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            readout("s1", "s2", "y0", "a0"),
            readout("s2", "s3", "y1", "a0"), final_state="s3",
        ), "stale accumulator handle")

    def test_dropped_accumulator_and_overlapping_reset_are_rejected(self) -> None:
        for module_scope in (False, True):
            with self.subTest(module_scope=module_scope):
                self.rejected(program(
                    load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
                    final_state="s1", module_scope=module_scope,
                ), "must read out the live MXU accumulator before block exit")
        self.accepted(program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            reset("s1", "s2", "b0"), readout("s2", "s3", "y", "a0"),
            readout("s3", "s4", "z", "b0"), final_state="s4",
        ))
        self.rejected(program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            reset("s1", "s2", "b0"), reset("s2", "s3", "c0"),
            readout("s3", "s4", "y", "c0"), final_state="s4",
        ), "cannot reset a unit with 2 live accumulators")

    def test_stale_state_and_a_third_live_weight_are_rejected(self) -> None:
        self.rejected(chain().replace('(%s1, %x, %weight, %a0)',
                                      '(%s0, %x, %weight, %a0)'),
                      "nonlinear virtual state chain")
        self.rejected(program(
            load("io2", "s0", "weight"),
            load("s0", "s1", "replacement"),
            load("s1", "s2", "third"),
            reset("s2", "s3", "a0"), readout("s3", "s4", "y", "a0"),
            reset("s4", "s5", "b0", "replacement"),
            readout("s5", "s6", "z", "b0"), final_state="s6",
        ), "cannot load a weight on a unit with 2 live weights")

    def test_legacy_matmul_coexists_and_clobbers_same_unit_handles(self) -> None:
        def legacy(unit: int) -> str:
            return (f'    %legacy = "atlas.virtual_mxu_matmul"(%x, %w) '
                    f'{{unit = {unit} : i32}} : ({FP8}, {FP8}) -> {BF16}')

        self.accepted(program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            legacy(1), readout("s1", "s2", "y", "a0"), final_state="s2",
        ))
        self.rejected(program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            legacy(0), readout("s1", "s2", "y", "a0"), final_state="s2",
        ), "cannot overwrite a live accumulator")
        self.rejected(program(
            load("io2", "s0", "weight"), legacy(0),
            reset("s0", "s1", "a0"), readout("s1", "s2", "y", "a0"),
            final_state="s2",
        ), "legacy virtual_mxu_matmul cannot overwrite a live weight")
        self.accepted(program(
            load("io2", "s0", "weight"), legacy(0),
            load("s0", "s1", "replacement"),
            reset("s1", "s2", "a0", "replacement"),
            readout("s2", "s3", "y", "a0"), final_state="s3",
        ))

    def test_handles_cannot_cross_blocks_by_capture_or_block_argument(self) -> None:
        captured_weight = program(
            load("io2", "s0", "weight"),
            f"    cf.br ^next(%s0 : {STATE})",
            f"  ^next(%next_state: {STATE}):",
            reset("next_state", "s1", "a0"),
            readout("s1", "s2", "y", "a0"), final_state="s2",
        )
        self.rejected(captured_weight, "virtual MXU handles cannot cross CFG blocks")
        passed_weight = captured_weight.replace(
            f"^next(%s0 : {STATE})",
            f"^next(%s0, %weight : {STATE}, {weight(0)})",
        ).replace(
            f"%next_state: {STATE}):",
            f"%next_state: {STATE}, %next_weight: {weight(0)}):",
        ).replace('(%next_state, %x, %weight)',
                  '(%next_state, %x, %next_weight)')
        self.rejected(passed_weight, "virtual block arguments must be BF16 tiles")
        captured_acc = program(
            load("io2", "s0", "weight"), reset("s0", "s1", "a0"),
            f"    cf.br ^next(%s1 : {STATE})",
            f"  ^next(%next_state: {STATE}):",
            readout("next_state", "s2", "y", "a0"), final_state="s2",
        )
        self.rejected(captured_acc, "must read out the live MXU accumulator before block exit")
        passed_acc = captured_acc.replace(
            f"^next(%s1 : {STATE})",
            f"^next(%s1, %a0 : {STATE}, {acc(0)})",
        ).replace(
            f"%next_state: {STATE}):",
            f"%next_state: {STATE}, %next_acc: {acc(0)}):",
        ).replace('(%next_state, %a0)', '(%next_state, %next_acc)')
        self.rejected(passed_acc, "must read out the live MXU accumulator before block exit")

    def test_virtual_handles_require_explicit_lowering_before_physical_stages(self) -> None:
        source = EXAMPLE.read_text()
        self.accepted(source)
        for tool, options in (
            ("atlas-emit", ()),
            ("atlas-opt", ("--verify-atlas-machine-stream",)),
            ("atlas-opt", ("--convert-atlas-to-llvm",)),
        ):
            with self.subTest(tool=tool, options=options):
                result = run(tool, source, *options)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, "")
        lowered = run("atlas-opt", source, "--lower-atlas-virtual-to-machine")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        self.assertNotIn("!atlas.virtual_", lowered.stdout)
        emitted = run("atlas-emit", lowered.stdout)
        self.assertNotEqual(emitted.returncode, 0)
        self.assertIn("untimed", emitted.stderr)
        timed = run("atlas-opt", lowered.stdout, "--insert-atlas-delays")
        self.assertEqual(timed.returncode, 0, timed.stderr)
        emitted = run("atlas-emit", timed.stdout)
        self.assertEqual(emitted.returncode, 0, emitted.stderr)
        self.assertTrue(emitted.stdout)


if __name__ == "__main__":
    unittest.main()
