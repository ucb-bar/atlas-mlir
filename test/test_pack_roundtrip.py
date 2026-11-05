"""Exact PACK relayout through quantized BF16 strips and an MXU copy/pop."""

from __future__ import annotations

import re
import unittest

from test_dialect import EMIT, OPT
from test_virtual_lowering import run
from verification_support import UNTIMED, assert_boundaries, rewrite_line

PACK = '"atlas.virtual_pack_fp8"'
ROUNDTRIP = '"atlas.virtual_pack_roundtrip_bf16"'
COPY = '"atlas.virtual_mxu_load_acc_bf16"'
POP = '"atlas.virtual_mxu_readout_fp8"'
LATE = "--lower-atlas-pack-relayout"
MACHINE = ("--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits")


def source(scale: int = 127, *, explicit: bool = False, live_units: tuple[int, ...] = ()) -> str:
    """Keep each requested accumulator live across the conversion in source order."""
    lines = ['module {', '  func.func @pack() -> !atlas.virtual_state attributes {'
        'atlas.input_dram_base = 2415919104 : i64, ' 'atlas.output_dram_base = 2415984640 : i64} {',
        '    %s0 = "atlas.virtual_start"() : () -> !atlas.virtual_state',
        '    %s1, %x = "atlas.virtual_input_bf16"(%s0) {index = 0 : i32}'
        ' : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)',]
    state = "%s1"
    for index, unit in enumerate(live_units):
        following = f"%load{index}"
        lines.append(f'    {following}, %acc{index} = {COPY}({state}, %x) '
            f'{{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) '
            f'-> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)')
        state = following
    operation, result_type = ((ROUNDTRIP, "bf16") if explicit else (PACK, "fp8"))
    lines.append(f'    %p = {operation}(%x) {{scale_code = {scale} : i32}} '
        f': (!atlas.virtual_bf16) -> !atlas.virtual_{result_type}')
    for index, unit in enumerate(live_units):
        following = f"%read{index}"
        lines.append(f'    {following}, %unused{index} = "atlas.virtual_mxu_readout_bf16"'
            f'({state}, %acc{index}) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) '
            '-> (!atlas.virtual_state, !atlas.virtual_bf16)')
        state = following
    if explicit:
        lines.append(f'    %done = "atlas.virtual_output_bf16"({state}, %p) {{index = 0 : i32}} '
            ': (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state')
    else:
        lines.extend(['    %out = arith.constant -1878982656 : i32', '    %bytes = arith.constant 1024 : i32',
            f'    %done = "atlas.virtual_dma_store_fp8"({state}, %p, %out, %bytes) '
            ': (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> !atlas.virtual_state',])
    lines.extend(['    return %done : !atlas.virtual_state', '  }', '}'])
    return "\n".join(lines) + "\n"


def increment_field(line: str, field: str, step: int = 1) -> str:
    return re.sub(rf"\b{field} = (\d+) : i32", lambda match: f"{field} = {int(match[1]) + step} : i32", line, count=1)


@unittest.skipUnless(OPT.is_file() and EMIT.is_file(), "build atlas-opt and atlas-emit")
class PackRoundTripTests(unittest.TestCase):
    def checked(self, text: str, *passes: str) -> str:
        result = run("atlas-opt", text, *passes)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_explicit_operation_compiles_all_admitted_scales(self) -> None:
        for scale in range(122, 129):
            with self.subTest(scale=scale):
                machine = self.checked(source(scale, explicit=True), "--atlas-compile")
                packs = [line for line in machine.splitlines() if '"atlas.vpu_pack"' in line]
                self.assertEqual(len(packs), 2)
                self.assertIn('direction = "bf16_to_fp8"', packs[0])
                self.assertIn('dst = 62 : i32', packs[0])
                self.assertIn('direction = "fp8_to_bf16"', packs[1])
                self.assertIn('src = 62 : i32', packs[1])
                self.assertNotIn('"atlas.scalar_store"', machine)
                assert_boundaries(self, machine)

    def test_automatic_pipeline_removes_scalar_relayout_for_unknown_bf16(self) -> None:
        # Input BF16 carries no finiteness assumption, including at nonunit scale.
        for scale in (122, 127, 128):
            with self.subTest(scale=scale):
                virtual = self.checked(source(scale), LATE)
                self.assertNotIn(PACK, virtual)
                for operation in (ROUNDTRIP, COPY, POP):
                    self.assertEqual(virtual.count(operation), 1)
                self.assertIn('code = 127 : i32', virtual)
                self.assertEqual(self.checked(virtual, LATE), virtual)
                machine = self.checked(source(scale), "--atlas-compile-unscheduled")
                self.assertNotIn('"atlas.scalar_store"', machine)
                self.assertNotIn('"atlas.branch"', machine)
                self.assertEqual(machine.count('"atlas.vpu_pack"'), 2)
                self.assertIn('kind = "acc_bf16"', machine)
                self.assertIn('format = "fp8"', machine)
                assert_boundaries(self, machine)

    def test_second_unit_is_used_when_first_has_two_live_accumulators(self) -> None:
        virtual = self.checked(source(live_units=(0, 0)), LATE)
        self.assertNotIn(PACK, virtual)
        self.assertIn(ROUNDTRIP, virtual)
        copied = next(line for line in virtual.splitlines() if COPY in line and 'unit = 1 : i32' in line)
        self.assertIn('virtual_mxu_acc<1>', copied)
        machine = self.checked(virtual, *MACHINE, "--insert-atlas-delays")
        self.assertNotIn('"atlas.scalar_store"', machine)
        assert_boundaries(self, machine)

    def test_all_accumulators_live_preserves_scalar_fallback(self) -> None:
        original = source(live_units=(0, 0, 1, 1))
        virtual = self.checked(original, LATE)
        self.assertIn(PACK, virtual)
        self.assertNotIn(ROUNDTRIP, virtual)
        self.assertEqual(virtual.count(COPY), 4)
        machine = self.checked(virtual, *MACHINE, "--insert-atlas-delays")
        self.assertIn('"atlas.scalar_store"', machine)
        self.assertIn('"atlas.branch"', machine)
        assert_boundaries(self, machine)

    def test_scale_types_ranges_and_unqualified_scale_codes_are_rejected(self) -> None:
        cases = [(source(-1, explicit=True), ()), (source(256, explicit=True), ()),
            (source(explicit=True).replace('scale_code = 127 : i32', 'scale_code = 127 : i64'), ()),
            (source(explicit=True).replace('scale_code = 127 : i32', 'scale_code = "127"'), ()),]
        for scale in (0, 121, 129, 255):
            cases.append((source(scale, explicit=True), MACHINE))
            # The automatic rewrite must preserve the original scale admission.
            cases.append((source(scale), (LATE, *MACHINE)))
        for index, (text, passes) in enumerate(cases):
            with self.subTest(case=index):
                result = run("atlas-opt", text, *passes)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertTrue(result.stderr)

    def test_generated_cfg_checks_bind_both_conversion_beats(self) -> None:
        machine = self.checked(source(122, explicit=True), *MACHINE)
        assert_boundaries(self, machine, UNTIMED)
        conversion = lambda direction: lambda line: ('"atlas.vpu_pack"' in line and
                                                     f'direction = "{direction}"' in line)
        scale_load = lambda code: lambda line: ('"atlas.scalar_load"' in line and 'kind = "seli"' in line
            and f'offset = {code} : i32' in line)
        mutations = [(scale_load(122), lambda line: increment_field(line, "offset")), (scale_load(127),
            lambda line: increment_field(line, "offset", -1)),]
        for direction in ("bf16_to_fp8", "fp8_to_bf16"):
            for field in ("src", "dst", "scale_reg"):
                # Keep tensor registers even and in range so source correspondence,
                # rather than the physical instruction's field schema, is checked.
                step = 2 if field in ("src", "dst") else 1
                mutations.append((conversion(direction), lambda line, field=field, step=step: increment_field(line,
                    field, -step if f'{field} = 62 : i32' in line else step)))
            other = "fp8_to_bf16" if direction == "bf16_to_fp8" else "bf16_to_fp8"
            mutations.append((conversion(direction), lambda line, direction=direction,
                other=other: line.replace(f'direction = "{direction}"', f'direction = "{other}"')))
        for index, (where, change) in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = rewrite_line(machine, where, change)
                assert_boundaries(self, changed, UNTIMED, rejects="CFG contract")


if __name__ == "__main__":
    unittest.main()
