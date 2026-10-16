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

    def test_unknown_and_wrong_scale_fallback(self):
        for text in (source(UNKNOWN), source(scale=126), source(scale=128, contract=", finite_src")):
            with self.subTest(text=text):
                lowered = self.lower(text)
                self.assertIn('"atlas.virtual_pack_fp8"', lowered)
                self.assertNotIn('"atlas.virtual_mxu_load_acc_bf16"', lowered)

    def test_nan_inf_and_overflow_capable_fallback(self):
        for bits in (0x7fc0, 0xffc0, 0x7f80, 0xff80):
            with self.subTest(bits=bits):
                producer = f'%x = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16'
                self.assertIn('"atlas.virtual_pack_fp8"', self.lower(source(producer)))
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

    def test_live_accumulator_slots_fallback(self):
        text = source(contract=", finite_src")
        text = text.replace('    %p =', '''    %as, %aa = "atlas.virtual_mxu_load_acc_bf16"(%t, %x) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %bs, %ba = "atlas.virtual_mxu_load_acc_bf16"(%as, %x) {unit = 1 : i32} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>)
    %p =''')
        text = text.replace('    %u =', '''    %cs, %cv = "atlas.virtual_mxu_readout_bf16"(%bs, %aa) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %ds, %dv = "atlas.virtual_mxu_readout_bf16"(%cs, %ba) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<1>) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %u =''').replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%ds,')
        self.assertIn('"atlas.virtual_pack_fp8"', self.lower(text.replace('unit = 1', 'unit = 0').replace('virtual_mxu_acc<1>', 'virtual_mxu_acc<0>')))

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


if __name__ == "__main__":
    unittest.main()
