"""Consumed PACK bytes, with a scalar oracle independent of the lowering.

Set ATLAS_PACK_BYTE_RUNNER to a JSON command array. The runner receives one
case-directory argument and executes its program.json/program.hex on both the
model and RTL, writing model-output-*.bin and rtl-output-*.bin. This test owns
the source/fixture/oracle and compares every returned output byte itself.
ATLAS_PACK_BYTE_OUTPUT_DIR optionally retains cases and execution receipts.
"""
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

from test_dialect import OPT, EMIT, ROOT, run


def pack_oracle(bits, scale):
    """Atlas VPU PACK: BF16 flush, integer nearest-even, E4M3 saturation.

    This uses the native PACK scalar definition, never MXU pop or clamps.
    Atlas flushes the rounded result below exponent -6 instead of producing
    IEEE FP8 subnormals, and maps both signed zeros and NaNs to positive zero.
    E8M0 codes 254 and 255 share the saturated signed exponent shift.
    """
    exponent, fraction = (bits >> 7) & 255, bits & 127
    if exponent == 0 or (exponent == 255 and fraction):
        return 0
    sign = (bits >> 8) & 128
    if exponent == 255:
        return sign | 126
    retained, remainder = divmod(128 + fraction, 16)
    retained += remainder > 8 or (remainder == 8 and retained % 2)
    power = exponent - min(scale, 254)
    if retained == 16:
        power += 1
        retained = 8
    if power < -6:
        return 0
    return sign | min(126, ((power + 7) * 8 + retained - 8))


def fixture(scale, special=False):
    # Include both signs around the rounding/flush/saturation boundaries and
    # each boundary's complete 128-word mantissa sweep. Shuffle so a lane or
    # half-tile relayout error changes the byte sequence.
    words = [0, 0x8000, 1, 0x8001, 0x7f, 0x807f, 0x80, 0x8080,
             0x7f7f, 0xff7f, 0x3f80, 0xbf80]
    effective = min(scale, 254)
    for exponent in sorted({max(1, effective - 7), max(1, effective - 6),
                            min(254, effective + 8)}):
        words += [sign | (exponent << 7) | mantissa
                  for sign in (0, 0x8000) for mantissa in range(128)]
    if special:
        words += [0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81, 0xff81]
    words = list(dict.fromkeys(words))
    assert len(words) <= 1024
    unique = len(words)
    words += [words[index % unique] for index in range(1024 - unique)]
    random.Random(0x5041434b + scale + 256 * special).shuffle(words)
    physical = b''.join(struct.pack('<H', words[row * 32 + half * 16 + col])
                        for half in (0, 1) for row in range(32) for col in range(16))
    return words, physical


def source(scale, special=False):
    contract = '' if special else ', finite_src'
    return f'''module {{
  func.func @consumed_pack() -> !atlas.virtual_state attributes {{atlas.input_dram_base = 2415919104 : i64, atlas.output_dram_base = 2415984640 : i64}} {{
    %s = "atlas.virtual_start"() : () -> !atlas.virtual_state
    %t, %x = "atlas.virtual_input_bf16"(%s) {{index = 0 : i32}} : (!atlas.virtual_state) -> (!atlas.virtual_state, !atlas.virtual_bf16)
    %p = "atlas.virtual_pack_fp8"(%x) {{scale_code = {scale} : i32{contract}}} : (!atlas.virtual_bf16) -> !atlas.virtual_fp8
    %address = arith.constant -1878982656 : i32
    %bytes = arith.constant 1024 : i32
    %u = "atlas.virtual_dma_store_fp8"(%t, %p, %address, %bytes) : (!atlas.virtual_state, !atlas.virtual_fp8, i32, i32) -> !atlas.virtual_state
    return %u : !atlas.virtual_state
  }}
}}
'''


class PackOracleTests(unittest.TestCase):
    def test_atlas_special_values_and_rounding(self):
        for bits in (0, 0x8000, 1, 0x807f, 0x7fc0, 0xffc0):
            self.assertEqual(pack_oracle(bits, 127), 0)
        for bits, expected in ((0x3f80, 0x38), (0xbf80, 0xb8),
                               (0x3f88, 0x38), (0x3f89, 0x39),
                               (0x3f98, 0x3a), (0x43e0, 0x7e),
                               (0x7f80, 0x7e), (0xff80, 0xfe)):
            self.assertEqual(pack_oracle(bits, 127), expected)
        self.assertEqual(pack_oracle(0x7f7f, 254), pack_oracle(0x7f7f, 255))


@unittest.skipUnless(OPT.is_file() and EMIT.is_file(), 'build atlas-opt and atlas-emit')
class ConsumedPackTests(unittest.TestCase):
    def test_many_scales_allocate_native_and_guarded(self):
        text = (ROOT / 'test/guarded-pack-many-scales.mlir').read_text()
        for flags in ([], ['--lower-atlas-guarded-pack=unit=0'],
                      ['--lower-atlas-guarded-pack=unit=1'],
                      ['--lower-atlas-guarded-pack', '--schedule-atlas-virtual']):
            with self.subTest(flags=flags):
                result = run(OPT, text, *flags, '--lower-atlas-virtual-to-machine',
                             '--insert-atlas-dma-waits', '--insert-atlas-delays',
                             '--verify-atlas-generated-schedule')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(run(EMIT, result.stdout).returncode, 0)

    @unittest.skipUnless(os.environ.get('ATLAS_PACK_BYTE_RUNNER'), 'select model+RTL byte runner')
    def test_model_and_rtl_consumed_bytes(self):
        runner = json.loads(os.environ['ATLAS_PACK_BYTE_RUNNER'])
        self.assertIsInstance(runner, list)
        self.assertTrue(runner)
        with tempfile.TemporaryDirectory(prefix='atlas-pack-') as temporary:
            output = Path(os.environ.get('ATLAS_PACK_BYTE_OUTPUT_DIR', temporary))
            output.mkdir(parents=True, exist_ok=True)
            # Wider scale codes require the finite contract. Native fallback
            # is admitted only at codes 122..128, including NaNs/infinities.
            scales = (0, 1, 121, 122, 126, 127, 128, 129, 245, 246, 247, 248, 253, 254, 255)
            cases = [(scale, unit, False) for scale in scales for unit in (0, 1)]
            cases += [(scale, unit, True) for scale in (122, 127, 128) for unit in (0, 1)]
            for scale, unit, special in cases:
                with self.subTest(scale=scale, unit=unit, fallback=special):
                    target = output / f's{scale:03d}-u{unit}-' / ('fallback' if special else 'finite')
                    target.mkdir(parents=True, exist_ok=False)
                    text = source(scale, special)
                    (target / 'source.mlir').write_text(text)
                    rewritten = run(OPT, text, f'--lower-atlas-guarded-pack=unit={unit}')
                    self.assertEqual(rewritten.returncode, 0, rewritten.stderr)
                    self.assertEqual('"atlas.virtual_pack_fp8"' in rewritten.stdout, special)
                    (target / 'rewritten.mlir').write_text(rewritten.stdout)
                    machine = run(OPT, rewritten.stdout,
                                  f'--lower-atlas-virtual-to-machine=allocation-report={target.resolve()}/allocation.json',
                                  '--insert-atlas-dma-waits', '--insert-atlas-delays',
                                  '--verify-atlas-machine-stream', '--verify-atlas-timing',
                                  '--verify-atlas-generated-schedule')
                    self.assertEqual(machine.returncode, 0, machine.stderr)
                    (target / 'machine.mlir').write_text(machine.stdout)
                    for name, flags in (('program.hex', []), ('program.json', ['--program-json'])):
                        emitted = run(EMIT, machine.stdout, *flags)
                        self.assertEqual(emitted.returncode, 0, emitted.stderr)
                        (target / name).write_text(emitted.stdout)
                    logical, physical = fixture(scale, special)
                    expected = bytes(pack_oracle(bits, scale) for bits in logical)
                    (target / 'input.bf16.bin').write_bytes(physical)
                    (target / 'expected.bin').write_bytes(expected)
                    (target / 'fixture.json').write_text(json.dumps({'scale': scale, 'unit': unit,
                        'finite_src': not special, 'logical_bf16_words': logical}) + '\n')
                    executed = subprocess.run([*runner, str(target.resolve())],
                                              capture_output=True, text=True)
                    (target / 'execution.log').write_text(executed.stdout + executed.stderr)
                    self.assertEqual(executed.returncode, 0, executed.stderr)
                    for backend in ('model', 'rtl'):
                        actuals = sorted(target.glob(f'{backend}-output-*.bin'))
                        self.assertGreaterEqual(len(actuals), 1, f'missing {backend} consumed bytes')
                        for actual in actuals:
                            self.assertEqual(actual.read_bytes(), expected, str(actual))


if __name__ == '__main__':
    unittest.main()
