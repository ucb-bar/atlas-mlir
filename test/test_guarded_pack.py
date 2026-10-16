"""Opt-in conversion contracts, proof boundaries and downstream verifiers."""
import unittest

from test_dialect import OPT, EMIT, run


def source(producer=None, scale=127, contract=""):
    producer = producer or '%x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16'
    return f'''module {{
  func.func @tile() -> !atlas.virtual_state attributes {{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %t, %input = "atlas.virtual_input_bf16"(%s) {{index = 0 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    {producer}
    %p = "atlas.virtual_pack_fp8"(%x) {{scale_code = {scale} : i32{contract}}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u = "atlas.virtual_output_bf16"(%t, %x) {{index = 0 : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %u : !atlas.virtual_state
  }}
}}
'''


UNKNOWN = '%x = "atlas.virtual_vpu_unary"(%input) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'


@unittest.skipUnless(OPT.is_file() and EMIT.is_file(), "build atlas-opt and atlas-emit")
class GuardedPackTests(unittest.TestCase):
    def lower(self, text, *passes):
        result = run(OPT, text, "--lower-atlas-guarded-pack", *passes)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_opt_in_required(self):
        result = run(OPT, source(contract=", finite_src"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"atlas.virtual_pack_fp8"', result.stdout)

    def test_static_finite_and_state(self):
        text = self.lower(source())
        self.assertNotIn('"atlas.virtual_pack_fp8"', text)
        self.assertEqual(text.count('"atlas.virtual_mxu_load_acc_bf16"'), 1)
        self.assertEqual(text.count('"atlas.virtual_mxu_readout_fp8"'), 1)
        self.assertIn('unit = 0 : i32', text)
        self.assertIn('kind = "min"', text)
        self.assertIn('kind = "max"', text)
        self.assertEqual(self.lower(text), text)

    def test_explicit_exact_source_contract(self):
        text = self.lower(source(UNKNOWN, contract=", finite_src"))
        self.assertNotIn('"atlas.virtual_pack_fp8"', text)
        # Output still uses the original producer, not the clamped temporary.
        self.assertIn('"atlas.virtual_output_bf16"', text)

    def test_unknown_source_fallback_at_scale_boundaries(self):
        for scale in (0, 126, 127, 128, 246, 247, 254, 255):
            with self.subTest(scale=scale):
                lowered = self.lower(source(UNKNOWN, scale=scale))
                self.assertIn('"atlas.virtual_pack_fp8"', lowered)
                self.assertNotIn('"atlas.virtual_mxu_load_acc_bf16"', lowered)

    def test_all_scale_constants_and_clamp_boundaries(self):
        for scale in range(256):
            with self.subTest(scale=scale):
                text = self.lower(source(UNKNOWN, scale=scale, contract=", finite_src"))
                self.assertNotIn('"atlas.virtual_pack_fp8"', text)
                self.assertIn(f'code = {254 - min(scale, 254)} : i32', text)
                if scale <= 246:
                    cap = ((scale + 8) << 7) | 96
                    self.assertIn(f'bits = {cap} : i32', text)
                    self.assertIn(f'bits = {cap | 0x8000} : i32', text)
                    self.assertEqual(text.count('"atlas.virtual_vpu_binary"'), 2)
                else:
                    self.assertNotIn('"atlas.virtual_vpu_splat"', text)
                    self.assertNotIn('"atlas.virtual_vpu_binary"', text)

    def test_scale_code_range_is_verified(self):
        for scale in (-1, 256):
            result = run(OPT, source(scale=scale), '--lower-atlas-guarded-pack')
            self.assertNotEqual(result.returncode, 0)

    def test_nan_inf_and_overflow_capable_fallback(self):
        for scale in (0, 127, 246, 247, 254, 255):
            for bits in (0x7fc0, 0xffc0, 0x7f80, 0xff80):
                with self.subTest(bits=bits, scale=scale):
                    producer = f'%x = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16'
                    self.assertIn('"atlas.virtual_pack_fp8"', self.lower(source(producer, scale=scale)))
        producer = '''%a = "atlas.virtual_vpu_splat"() {bits = 32639 : i32} : () -> !atlas.virtual_bf16
    %x = "atlas.virtual_vpu_binary"(%a, %a) {kind = "mul"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
        self.assertIn('"atlas.virtual_pack_fp8"', self.lower(source(producer)))

    def test_contract_is_typed(self):
        result = run(OPT, source(contract=", finite_src = true"), "--lower-atlas-guarded-pack")
        self.assertNotEqual(result.returncode, 0)

    def test_contract_does_not_authorize_another_pack(self):
        text = source(UNKNOWN, contract=", finite_src")
        text = text.replace('    %u =', '''    %q = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u =''')
        lowered = self.lower(text)
        self.assertEqual(lowered.count('"atlas.virtual_pack_fp8"'), 1)
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 1)

    def test_clamp_constants_are_shared_per_block(self):
        text = source().replace('    %u =', '''    %q = "atlas.virtual_pack_fp8"(%x) {scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u =''')
        lowered = self.lower(text)
        self.assertEqual(lowered.count('bits = 50144 : i32'), 1)
        self.assertEqual(lowered.count('bits = 17376 : i32'), 1)
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 2)

    def test_mixed_scales_have_independent_constants(self):
        text = source(UNKNOWN, scale=0, contract=", finite_src")
        packs = '\n'.join(f'    %q{index} = "atlas.virtual_pack_fp8"(%x) {{finite_src, scale_code = {scale} : i32}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8'
                          for index, scale in enumerate((246, 247, 254, 255, 0, 246)))
        lowered = self.lower(text.replace('    %u =', packs + '\n    %u ='))
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 7)
        for scale in (0, 246):
            cap = ((scale + 8) << 7) | 96
            self.assertEqual(lowered.count(f'bits = {cap} : i32'), 1)
            self.assertEqual(lowered.count(f'bits = {cap | 0x8000} : i32'), 1)
        self.assertEqual(lowered.count('"atlas.virtual_vpu_binary"'), 8)
        for inverse in (254, 8, 7):
            self.assertEqual(lowered.count(f'code = {inverse} : i32'), 1)
        self.assertEqual(lowered.count('code = 0 : i32'), 2)

    def test_live_accumulator_slots_fallback(self):
        text = source(contract=", finite_src")
        text = text.replace('    %p =', '''    %as, %aa = "atlas.virtual_mxu_load_acc_bf16"(%t, %x) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %bs, %ba = "atlas.virtual_mxu_load_acc_bf16"(%as, %x) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %p =''')
        text = text.replace('    %u =', '''    %cs, %cv = "atlas.virtual_mxu_readout_bf16"(%bs, %aa) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %ds, %dv = "atlas.virtual_mxu_readout_bf16"(%cs, %ba) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %u =''').replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%ds,')
        for scale in (0, 127, 246, 247, 254, 255):
            with self.subTest(scale=scale):
                candidate = text.replace('scale_code = 127', f'scale_code = {scale}')
                candidate = candidate.replace('unit = 1', 'unit = 0').replace('virtual_mxu_acc<1>', 'virtual_mxu_acc<0>')
                self.assertIn('"atlas.virtual_pack_fp8"', self.lower(candidate))

    def test_engine_option(self):
        result = run(OPT, source(), '--lower-atlas-guarded-pack=unit=1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('unit = 1 : i32', result.stdout)
        result = run(OPT, source(), '--lower-atlas-guarded-pack=unit=2')
        self.assertNotEqual(result.returncode, 0)

    def test_multiblock_function_is_unchanged(self):
        text = '''module {
  func.func @tile() -> !atlas.virtual_state {
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %x = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16
    cf.br ^done(%s, %x : !atlas.virtual_state, !atlas.virtual_bf16)
  ^done(%t: !atlas.virtual_state, %value: !atlas.virtual_bf16):
    %p = "atlas.virtual_pack_fp8"(%value) {finite_src, scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u = "atlas.virtual_output_bf16"(%t, %value) {index = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> !atlas.virtual_state
    return %u : !atlas.virtual_state
  }
}'''
        original = run(OPT, text)
        self.assertEqual(original.returncode, 0, original.stderr)
        self.assertEqual(self.lower(text), original.stdout)

    def test_incoming_accumulators_are_outside_virtual_contract(self):
        text = source().replace('@tile()', '@tile(%a: !atlas.virtual_mxu_acc<0>, %b: !atlas.virtual_mxu_acc<0>)')
        result = run(OPT, text, '--lower-atlas-guarded-pack')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('virtual entry arguments must be i1/i32 controls', result.stderr)

    def test_existing_machine_metadata_and_timing_verifiers(self):
        for producer, contract in ((None, ""), (UNKNOWN, ", finite_src"), (UNKNOWN, "")):
            with self.subTest(producer=producer, contract=contract):
                text = self.lower(source(producer, contract=contract),
                                  "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits",
                                  "--insert-atlas-delays", "--verify-atlas-generated-schedule")
                emitted = run(EMIT, text)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)

    def test_scale_boundaries_reach_existing_machine_verifiers(self):
        for scale in (0, 246, 247, 254, 255):
            for unit in (0, 1):
                result = run(OPT, source(UNKNOWN, scale=scale, contract=", finite_src"),
                             f'--lower-atlas-guarded-pack=unit={unit}',
                             '--lower-atlas-virtual-to-machine', '--insert-atlas-dma-waits',
                             '--insert-atlas-delays', '--verify-atlas-generated-schedule')
                self.assertEqual(result.returncode, 0, result.stderr)
                emitted = run(EMIT, result.stdout)
                self.assertEqual(emitted.returncode, 0, emitted.stderr)

    def test_retained_pack_can_still_fail_native_scale_admission(self):
        result = run(OPT, source(UNKNOWN, scale=0), '--lower-atlas-guarded-pack',
                     '--lower-atlas-virtual-to-machine')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('scale codes 122..128 only', result.stderr)


if __name__ == "__main__":
    unittest.main()
