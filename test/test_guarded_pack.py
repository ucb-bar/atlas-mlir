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
        # Splat(1) proves the unit interval at scale 127: both clamps vanish.
        self.assertNotIn('kind = "min"', text)
        self.assertNotIn('kind = "max"', text)
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
        text = source(UNKNOWN, contract=", finite_src").replace('    %u =', '''    %q = "atlas.virtual_pack_fp8"(%x) {finite_src, scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
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

    def annotated(self, text):
        result = run(OPT, text, "--lower-atlas-guarded-pack=annotate-facts")
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def splat(self, bits):
        return f'%x = "atlas.virtual_vpu_splat"() {{bits = {bits} : i32}} : () -> !atlas.virtual_bf16'

    def test_range_clamps_parameterized_by_scale(self):
        for bits in (0, 0x8000, 1, 0x7f, 0x3f00, 0x3f80):
            for scale in (0, 118, 119, 127, 246, 247, 255):
                with self.subTest(bits=bits, scale=scale):
                    text = self.annotated(source(self.splat(bits), scale=scale))
                    self.assertIn('atlas.derived_finite = true', text)
                    self.assertIn('atlas.derived_nonnegative = true', text)
                    self.assertIn('atlas.derived_unit_interval = true', text)
                    self.assertIn('atlas.pack_finite_proof = "derived"', text)
                    self.assertNotIn('kind = "max"', text)
                    self.assertEqual('kind = "min"' in text, scale < 119)
        # A negative subnormal is finite but not nonnegative or unit interval.
        text = self.annotated(source(self.splat(0x8001)))
        self.assertIn('atlas.derived_finite = true', text)
        self.assertIn('atlas.derived_nonnegative = false', text)
        self.assertIn('kind = "max"', text)
        self.assertIn('kind = "min"', text)

    def test_nonnegative_does_not_imply_finite(self):
        text = self.annotated(source(self.splat(0x7f80)))
        self.assertIn('atlas.derived_nonnegative = true', text)
        self.assertIn('atlas.derived_finite = false', text)
        self.assertIn('atlas.derived_unit_interval = false', text)
        self.assertIn('"atlas.virtual_pack_fp8"', text)
        for bits in (0x7fc0, 0xffc0):
            text = self.annotated(source(self.splat(bits)))
            self.assertIn('atlas.derived_nonnegative = false', text)
            self.assertIn('atlas.derived_finite = false', text)

    def test_unknown_total_order_clamp_proves_unit_interval(self):
        producer = '''%lo = "atlas.virtual_vpu_splat"() {bits = 0 : i32} : () -> !atlas.virtual_bf16
    %hi = "atlas.virtual_vpu_splat"() {bits = 16256 : i32} : () -> !atlas.virtual_bf16
    %a = "atlas.virtual_vpu_binary"(%input, %lo) {kind = "max"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16
    %x = "atlas.virtual_vpu_binary"(%a, %hi) {kind = "min"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
        text = self.annotated(source(producer))
        self.assertNotIn('"atlas.virtual_pack_fp8"', text)
        self.assertIn('atlas.derived_unit_interval = true', text)
        self.assertEqual(text.count('"atlas.virtual_vpu_binary"'), 2)
        for kind in ('row_min', 'row_max'):
            reduced = producer.replace('%x =', '%bounded =') + f'''
    %x = "atlas.virtual_vpu_reduce"(%bounded) {{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'''
            text = self.annotated(source(reduced))
            self.assertIn('atlas.derived_unit_interval = true', text)
        # One-sided clamps can still select a signed NaN, not a finite value.
        for kind in ('min', 'max'):
            one = self.splat(0x3f80).replace('%x =', '%a =') + f'''
    %x = "atlas.virtual_vpu_binary"(%input, %a) {{kind = "{kind}"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
            self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(one)))

    def test_unproved_attrs_are_not_analysis_inputs(self):
        producer = UNKNOWN.replace('{kind = "mov"}', '{kind = "mov", atlas.derived_finite = true, atlas.derived_unit_interval = true}')
        text = self.annotated(source(producer))
        self.assertIn('"atlas.virtual_pack_fp8"', text)
        self.assertIn('atlas.pack_finite_proof = "unproved"', text)

    def test_caller_contract_provenance_and_no_range_invention(self):
        text = self.annotated(source(UNKNOWN, contract=", finite_src"))
        self.assertIn('atlas.pack_finite_proof = "caller_finite_src"', text)
        self.assertIn('atlas.derived_finite = false', text)
        self.assertIn('atlas.derived_nonnegative = false', text)
        self.assertIn('atlas.caller_contract = true', text)
        self.assertIn('kind = "max"', text)
        self.assertIn('kind = "min"', text)

    def test_facts_are_ssa_based_and_preserved_through_mov_min_max(self):
        producer = self.splat(0x3f80).replace('%x =', '%odd_name =') + '''
    %renamed = "atlas.virtual_vpu_unary"(%odd_name) {kind = "mov"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16
    %x = "atlas.virtual_vpu_binary"(%renamed, %odd_name) {kind = "min"} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
        text = self.annotated(source(producer))
        self.assertIn('atlas.derived_unit_interval = true', text)
        self.assertNotIn('"atlas.virtual_pack_fp8"', text)
        self.assertEqual(text.count('"atlas.virtual_vpu_binary"'), 1)
        for kind in ('min', 'max'):
            producer = self.splat(0x3f80).replace('%x =', '%a =') + f'''
    %x = "atlas.virtual_vpu_binary"(%input, %a) {{kind = "{kind}"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
            self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(producer)))
            producer = producer.replace('%input, %a', '%a, %input')
            self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(producer)))

    def test_overflow_and_zero_reciprocal_do_not_gain_finite_facts(self):
        for kind in ('add', 'mul', 'sub'):
            producer = self.splat(0x7f7f).replace('%x =', '%a =') + f'''
    %x = "atlas.virtual_vpu_binary"(%a, %a) {{kind = "{kind}"}} : (!atlas.virtual_bf16, !atlas.virtual_bf16) -> !atlas.virtual_bf16'''
            self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(producer)))
        for bits, kind in ((0, 'recip'), (0x8000, 'recip'), (0x7f7f, 'exp')):
            producer = self.splat(bits).replace('%x =', '%a =') + f'''
    %x = "atlas.virtual_vpu_unary"(%a) {{kind = "{kind}"}} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'''
            self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(producer)))

    def test_relu_finite_source_proves_nonnegative(self):
        producer = self.splat(0x4500).replace('%x =', '%a =') + '''
    %x = "atlas.virtual_vpu_unary"(%a) {kind = "relu"} : (!atlas.virtual_bf16) -> !atlas.virtual_bf16'''
        text = self.annotated(source(producer))
        self.assertIn('atlas.derived_nonnegative = true', text)
        self.assertNotIn('kind = "max"', text)
        self.assertIn('kind = "min"', text)
        unknown = UNKNOWN.replace('kind = "mov"', 'kind = "relu"')
        self.assertIn('"atlas.virtual_pack_fp8"', self.annotated(source(unknown)))

    def test_same_scale_lazily_materializes_only_required_constants(self):
        text = source().replace('    %u =', '''    %q = "atlas.virtual_pack_fp8"(%input) {finite_src, scale_code = 127 : i32} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %u =''')
        lowered = self.lower(text)
        self.assertEqual(lowered.count('"atlas.virtual_mxu_readout_fp8"'), 2)
        self.assertEqual(lowered.count('bits = 50144 : i32'), 1)
        self.assertEqual(lowered.count('bits = 17376 : i32'), 1)
        self.assertEqual(lowered.count('"atlas.virtual_scale_constant"'), 1)

    def test_wrong_dtype_and_shape_are_rejected(self):
        for ty in ('!atlas.virtual_fp8', 'tensor<16x32xbf16>', 'tensor<32x32xbf16>'):
            text = source().replace('!atlas.virtual_bf16', ty)
            result = run(OPT, text, '--lower-atlas-guarded-pack')
            self.assertNotEqual(result.returncode, 0)

    def test_bounded_raw_fp8_reset_is_finite_on_either_unit(self):
        for unit in (0, 1):
            producer = f'''%f, %raw = "atlas.virtual_input_fp8"(%t) {{index = 1 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %w, %weight = "atlas.virtual_mxu_load_weight"(%f, %raw) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<{unit}>)
    %r, %acc = "atlas.virtual_mxu_reset"(%w, %raw, %weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
    %z, %x = "atlas.virtual_mxu_readout_bf16"(%r, %acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
            text = self.annotated(source(producer).replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%z,'))
            self.assertIn('atlas.derived_finite = true', text)
            self.assertIn('32-term raw FP8 reset contraction', text)
            self.assertNotIn('"atlas.virtual_pack_fp8"', text)
            self.assertIn('kind = "max"', text)
            self.assertIn('kind = "min"', text)

    def test_accumulate_is_unknown_even_after_finite_reset(self):
        producer = '''%f, %raw = "atlas.virtual_input_fp8"(%t) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)
    %w, %weight = "atlas.virtual_mxu_load_weight"(%f, %raw) {unit = 0 : i32} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_weight<0>)
    %r, %acc = "atlas.virtual_mxu_reset"(%w, %raw, %weight) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %r2, %acc2 = "atlas.virtual_mxu_accumulate"(%r, %raw, %weight, %acc) : (!atlas.virtual_state, !atlas.virtual_fp8, !atlas.virtual_mxu_weight<0>, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>)
    %z, %x = "atlas.virtual_mxu_readout_bf16"(%r2, %acc2) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<0>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
        text = self.annotated(source(producer).replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%z,'))
        self.assertIn('"atlas.virtual_pack_fp8"', text)
        self.assertIn('atlas.derived_finite = false', text)

    def test_accumulator_copy_and_pure_matmul_facts(self):
        for unit in (0, 1):
            producer = self.splat(0x3f80).replace('%x =', '%a =') + f'''
    %r, %acc = "atlas.virtual_mxu_load_acc_bf16"(%t, %a) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_bf16) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
    %z, %x = "atlas.virtual_mxu_readout_bf16"(%r, %acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
            text = self.annotated(source(producer).replace('"atlas.virtual_output_bf16"(%t,', '"atlas.virtual_output_bf16"(%z,'))
            self.assertIn('atlas.derived_unit_interval = true', text)
            for kind in ('copy', 'matmul'):
                input_ = '''%f, %raw = "atlas.virtual_input_fp8"(%t) {index = 1 : i32} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_fp8)'''
                if kind == 'copy':
                    op = f'''%r, %acc = "atlas.virtual_mxu_load_acc_fp8"(%f, %raw) {{unit = {unit} : i32}} : (!atlas.virtual_state, !atlas.virtual_fp8) -> (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>)
    %z, %x = "atlas.virtual_mxu_readout_bf16"(%r, %acc) : (!atlas.virtual_state, !atlas.virtual_mxu_acc<{unit}>) -> (!atlas.virtual_state, !atlas.virtual_bf16)'''
                    state = '%z'
                else:
                    op = f'''%x = "atlas.virtual_mxu_matmul"(%raw, %raw) {{unit = {unit} : i32}} : (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16'''
                    state = '%f'
                text = self.annotated(source(input_ + '\n    ' + op).replace('"atlas.virtual_output_bf16"(%t,', f'"atlas.virtual_output_bf16"({state},'))
                self.assertIn('atlas.derived_finite = true', text)
                self.assertNotIn('"atlas.virtual_pack_fp8"', text)

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
