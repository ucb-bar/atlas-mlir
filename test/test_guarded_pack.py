"""Finite-source proofs, resource fallback and ordinary lowering checks."""
import unittest

from test_dialect import OPT, EMIT, run

PACK = '"atlas.virtual_pack_fp8"'
ACC_LOAD = '"atlas.virtual_mxu_load_acc_bf16"'
MACHINE = ("--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
           "--insert-atlas-delays", "--verify-atlas-generated-schedule")


def source(producer=None, scale=127, *, before="", after="", state="%t", value="%x"):
    producer = producer or '%x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16'
    return f'''module {{
  func.func @tile() -> !atlas.virtual_state attributes {{
      atlas.input_dram_base = 2415919104 : i64,
      atlas.output_dram_base = 2415984640 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %t, %input = "atlas.virtual_input_bf16"(%s) {{index = 0 : i32}}
      : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    {producer}
    {before}
    %p = "atlas.virtual_pack_fp8"(%x) {{scale_code = {scale} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    {after}
    %u = "atlas.virtual_output_bf16"({state}, {value}) {{index = 0 : i32}}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %u : !atlas.virtual_state
  }}
}}
'''

UNKNOWN = '%x = "atlas.virtual_vpu_unary"(%input) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'


@unittest.skipUnless(OPT.is_file() and EMIT.is_file(), "build atlas-opt and atlas-emit")
class GuardedPackTests(unittest.TestCase):
    def checked(self, text, *passes, tool=OPT):
        result = run(tool, text, *passes)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def lower(self, text, *passes):
        return self.checked(text, "--lower-atlas-guarded-pack", *passes)

    def emit(self, text, *passes):
        self.checked(self.checked(text, *passes), tool=EMIT)

    def test_pass_is_opt_in_and_ignores_unproved_attributes(self):
        self.assertIn(PACK, run(OPT, source()).stdout)
        text = source(UNKNOWN).replace('{scale_code', '{finite_src, scale_code')
        self.assertIn(PACK, self.lower(text))

    def test_raw_fp8_reset_is_finite_on_both_units(self):
        for unit in (0, 1):
            producer = f'''%a, %fp8 = "atlas.virtual_input_fp8"(%t) {{index = 1 : i32}}
      : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %b, %weight = "atlas.virtual_mxu_load_weight"(%a, %fp8) {{unit = {unit} : i32}}
      : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<{unit}>)
    %c, %acc = "atlas.virtual_mxu_reset"(%b, %fp8, %weight)
      : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>)
      -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
    %d, %x = "atlas.virtual_mxu_readout_bf16"(%c, %acc)
      : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
            text = source(producer, state="%d")
            self.assertNotIn(PACK, self.lower(text))
            accumulated = text.replace('    %d, %x =', f'''    %e, %next = "atlas.virtual_mxu_accumulate"(%c, %fp8, %weight, %acc)
      : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>, !atlas.virtual_mxu_acc<{unit}>)
      -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
    %d, %x =''').replace('"atlas.virtual_mxu_readout_bf16"(%c, %acc)', '"atlas.virtual_mxu_readout_bf16"(%e, %next)')
            self.assertIn(PACK, self.lower(accumulated))
            pure = f'''%a, %fp8 = "atlas.virtual_input_fp8"(%t) {{index = 1 : i32}}
      : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %x = "atlas.virtual_mxu_matmul"(%fp8, %fp8) {{unit = {unit} : i32}}
      : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16'''
            text = source(pure, state="%a")
            self.assertNotIn(PACK, self.lower(text))

    def test_bf16_accumulator_copy_preserves_only_proven_finiteness(self):
        for operand, finite in (('%input', False), ('%finite', True)):
            producer = f'''%finite = "atlas.virtual_vpu_splat"() {{bits = 16256 : i32}} : () -> !atlas.virtual_bf16
    %a, %acc = "atlas.virtual_mxu_load_acc_bf16"(%t, {operand}) {{unit = 0 : i32}}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %b, %x = "atlas.virtual_mxu_readout_bf16"(%a, %acc)
      : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
            text = source(producer, state="%b")
            self.assertEqual(PACK not in self.lower(text), finite)

    def test_all_scale_constants_and_clamp_boundaries(self):
        for scale in range(256):
            with self.subTest(scale=scale):
                text = self.lower(source(scale=scale))
                self.assertNotIn(PACK, text)
                self.assertNotIn("atlas.guarded_pack_bound", text)
                if scale == 127:
                    for operation in (ACC_LOAD, '"atlas.virtual_mxu_readout_fp8"'):
                        self.assertEqual(text.count(operation), 1)
                    for attribute in ('unit = 0 : i32', 'kind = "min"', 'kind = "max"'):
                        self.assertIn(attribute, text)
                    self.assertEqual(self.lower(text), text)
                self.assertIn(f'code = {254 - min(scale, 254)} : i32', text)
                if scale <= 246:
                    cap = ((scale + 8) << 7) | 96
                    self.assertIn(f'bits = {cap} : i32', text)
                    self.assertIn(f'bits = {cap | 0x8000} : i32', text)
                    self.assertEqual(text.count('"atlas.virtual_vpu_binary"'), 2)
                else:
                    self.assertEqual(text.count('"atlas.virtual_vpu_splat"'), 1)
                    self.assertNotIn('"atlas.virtual_vpu_binary"', text)

        duplicate = '%q = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'
        lowered = self.lower(source(after=duplicate))
        for fragment in ('bits = 50144 : i32', 'bits = 17376 : i32'):
            self.assertEqual(lowered.count(fragment), 1)
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 2)
        self.assertEqual(lowered.count("atlas.guarded_pack_bound"), 2)

    def test_repeated_conversions_share_only_equal_clamp_bounds(self):
        packs = '\n'.join(
            f'%p{i} = "atlas.virtual_pack_fp8"(%x) {{scale_code = {127 + i % 2} : i32}} '
            ': (!atlas.virtual_bf16) -> !atlas.virtual_fp8'
            for i in range(16))
        lowered = self.lower(source(after=packs))
        # The original source splat plus two bounds for each of the two scales.
        self.assertEqual(lowered.count('"atlas.virtual_vpu_splat"'), 5)
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 17)
        for scale in (127, 128):
            cap = ((scale + 8) << 7) | 96
            for bits in (cap, cap | 0x8000):
                self.assertEqual(lowered.count(f'bits = {bits} : i32'), 1)
        # Equal bits in distinct functions must have separate dominating defs.
        body = source().removeprefix('module {').removesuffix('}\n')
        two_functions = 'module {' + body + body.replace('@tile()', '@other()') + '}\n'
        separate = self.lower(two_functions)
        self.assertEqual(separate.count('bits = 17376 : i32'), 2)
        self.assertEqual(separate.count('bits = 50144 : i32'), 2)

    def test_unproved_sources_fall_back(self):
        producers = [UNKNOWN] + [
            f'%x = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16'
            for bits in (0x7fc0, 0xffc0, 0x7f80, 0xff80)]
        producers.append('''%a = "atlas.virtual_vpu_splat"() {bits = 32639 : i32} : () -> !atlas.virtual_bf16
    %x = "atlas.virtual_vpu_binary"(%a, %a) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16''')
        for producer in producers:
            for scale in (0, 126, 127, 128, 246, 247, 254, 255):
                with self.subTest(producer=producer, scale=scale):
                    lowered = self.lower(source(producer, scale))
                    self.assertIn(PACK, lowered)
                    self.assertNotIn(ACC_LOAD, lowered)

    def test_machine_pipelines_and_consumed_results(self):
        consumer = '''%result = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32}
      : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16'''
        for scale in range(122, 129):
            text = source(scale=scale, after=consumer, value="%result")
            for flags in (('--atlas-compile',), ('--atlas-compile=guarded-pack=true',),
                          ('--lower-atlas-guarded-pack=unit=1', '--atlas-compile')):
                self.emit(text, *flags)
        for producer, scale, unit in [(None, s, u) for s in (0, 127, 246, 247, 254, 255) for u in (0, 1)] + [(UNKNOWN, 127, 0)]:
            with self.subTest(producer=producer, scale=scale, unit=unit):
                self.emit(source(producer, scale), f'--lower-atlas-guarded-pack=unit={unit}', *MACHINE)

    def test_register_pressure_is_recovered_by_spilling(self):
        # Fourteen original tiles fit beside the native PACK source, but
        # conversion needs additional BF16 bounds/intermediates while they live.
        tiles = '\n'.join(
            f'    %v{i} = "atlas.virtual_vpu_splat"() {{bits = {16257+i} : i32}} : () -> !atlas.virtual_bf16'
            for i in range(14))
        outputs = '\n'.join(f'''%o{i} = "atlas.virtual_output_bf16"(%o{i-1}, %v{i}) {{index = {i+1} : i32}}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state''' for i in range(14))
        text = source(before=tiles).replace('    return %u', outputs.replace('%o-1', '%u') + '\n    return %o13')
        self.checked(text, '--lower-atlas-virtual-to-machine')
        for unit in (0, 1):
            for scale in (0, 127, 246, 255):
                with self.subTest(unit=unit, scale=scale):
                    candidate = text.replace('scale_code = 127', f'scale_code = {scale}')
                    converted = self.checked(candidate, f'--lower-atlas-guarded-pack=unit={unit}')
                    self.assertNotIn(PACK, converted)
                    self.assertIn(ACC_LOAD, converted)
                    spilled = self.checked(converted, '--spill-atlas-virtual')
                    if scale <= 246:
                        self.assertIn('"atlas.virtual_spill_bf16"', spilled)
                    self.emit(spilled, *MACHINE)

    def test_allocation_rematerializes_shared_bounds_under_pressure(self):
        # 15 middle tiles fill the BF16 pairs between the PACKs: per-use bounds
        # fit, bounds shared across both PACKs would not.
        tiles = '\n'.join(
            f'%v{i} = "atlas.virtual_vpu_splat"() {{bits = {16257 + i} : i32}} '
            ': () -> !atlas.virtual_bf16' for i in range(15))
        outputs = '\n'.join(
            f'%o{i} = "atlas.virtual_output_bf16"(%o{i - 1}, %v{i}) '
            f'{{index = {i + 1} : i32}} : '
            '(!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state'
            for i in range(15)).replace('%o-1', '%first_out')
        after = f'''%first = "atlas.virtual_mxu_matmul"(%p, %p) {{unit = 1 : i32}}
      : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16
    %first_out = "atlas.virtual_output_bf16"(%t, %first) {{index = 0 : i32}}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    {tiles}
    {outputs}
    %last = "atlas.virtual_vpu_splat"() {{bits = 16256 : i32}} : () -> !atlas.virtual_bf16
    %q = "atlas.virtual_pack_fp8"(%last) {{scale_code = 127 : i32}}
      : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %result = "atlas.virtual_mxu_matmul"(%q, %q) {{unit = 1 : i32}}
      : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16'''
        text = source(after=after, state="%o14", value="%result")
        text = text.replace('"atlas.virtual_output_bf16"(%o14, %result) {index = 0',
                            '"atlas.virtual_output_bf16"(%o14, %result) {index = 16')
        # Allocate directly; a spill pass would hide the lifetime regression.
        self.checked(text, '--lower-atlas-virtual-to-machine')
        # Both PACKs must share one bound pair, or the test is vacuous.
        shared = self.lower(text)
        self.assertEqual(shared.count('bits = 50144 : i32'), 1)
        self.assertEqual(shared.count("atlas.guarded_pack_bound"), 2)
        for unit in (0, 1):
            with self.subTest(unit=unit):
                self.emit(text, f'--lower-atlas-guarded-pack=unit={unit}', *MACHINE)

    def test_live_accumulator_slots_fallback(self):
        text = source()
        text = text.replace('    %p =', '''    %as, %aa = "atlas.virtual_mxu_load_acc_bf16"(%t, %x) {unit = 1 : i32}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %bs, %ba = "atlas.virtual_mxu_load_acc_bf16"(%as, %x) {unit = 1 : i32}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %p =''')
        text = text.replace('    %u =', '''    %cs, %cv = "atlas.virtual_mxu_readout_bf16"(%bs, %aa)
      : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %ds, %dv = "atlas.virtual_mxu_readout_bf16"(%cs, %ba)
      : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %u =''').replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%ds,')
        for scale in (0, 127, 246, 247, 254, 255):
            with self.subTest(scale=scale):
                candidate = text.replace('scale_code = 127', f'scale_code = {scale}')
                candidate = candidate.replace('unit = 1', 'unit = 0').replace('virtual_mxu_acc<1>', 'virtual_mxu_acc<0>')
                self.assertIn(PACK, self.lower(candidate))

    def test_options_and_scale_range(self):
        self.assertIn('unit = 1 : i32', self.checked(source(), '--lower-atlas-guarded-pack=unit=1'))
        incoming = source().replace('@tile()', '@tile(%a: !atlas.virtual_mxu_acc<0>, %b: !atlas.virtual_mxu_acc<0>)')
        cases = [(source(scale=s), ("--lower-atlas-guarded-pack",), "") for s in (-1, 256)] + [
            (source(), ('--lower-atlas-guarded-pack=unit=2',), ""),
            (incoming, ('--lower-atlas-guarded-pack',), 'virtual entry arguments must be i1/i32 controls'),
            (source(UNKNOWN, 0), ('--lower-atlas-guarded-pack', '--lower-atlas-virtual-to-machine'), 'scale codes 122..128 only')]
        for text, flags, message in cases:
            result = run(OPT, text, *flags)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)

    def test_multiblock_function_is_unchanged(self):
        text = '''module {
  func.func @tile() -> !atlas.virtual_state {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16
    cf.br ^done(%s, %x : !atlas.virtual_state, !atlas.virtual_bf16)
  ^done(%t: !atlas.virtual_state, %value: !atlas.virtual_bf16):
    %p = "atlas.virtual_pack_fp8"(%value) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u = "atlas.virtual_output_bf16"(%t, %value) {index = 0 : i32}
      : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %u : !atlas.virtual_state
  }
}'''
        self.assertEqual(self.lower(text), self.checked(text))


if __name__ == "__main__":
    unittest.main()
