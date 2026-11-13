"""Paired raw-word unary kernels preserve every encoding and atomic rejection."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from atlas_mlir_native import exact_float_device as device
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_exact_float_device import request
from test_exact_float_vector_preflight import VectorPreflightTest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                     'select Rust simulator and Atlas tools')
class UnaryWordPairTest(unittest.TestCase):
    def test_all_encodings_odd_tails_and_rejected_last_tiles(self):
        compare = VectorPreflightTest().compare_runs
        runner = os.environ['ATLAS_RUST_NPU_MODEL']
        for operation in ('bf16_to_f32', 'bf16_neg'):
            codes = np.arange(65536, dtype='<u2')
            if operation == 'bf16_to_f32':
                codes = codes[(codes & 0x7fff) <= 0x7f80]
            np.random.default_rng(80321).shuffle(codes)
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ATLAS_EXACT_CAST_VECTOR_PREFLIGHT='1'):
                root = Path(directory)
                for count in (1, 35, 1025, len(codes)):
                    with self.subTest(operation=operation, count=count):
                        source = request(operation, count)
                        programs = []
                        for expanded in (False, True):
                            with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                    execution_instruction_limit=128000000, exact_float_unroll=expanded)):
                                plan = device.prepare_plan(source)
                                target = root / f'{count}-{expanded}'
                                device.compile_source(None, source, target_source=ROOT,
                                    destination=target, limits=None)
                                bound = json.loads((target / 'float-instruction-bound.json').read_text())
                                self.assertTrue(bound['admitted'])
                                self.assertLessEqual((target / 'program.bin').stat().st_size, 131072)
                                programs.append(target / 'program.bin')
                        raw = codes[:count].tobytes()
                        compare(root, runner, source, plan, programs, [raw])
                        if operation == 'bf16_to_f32':
                            for index in sorted({0, count // 2, count - 1}):
                                invalid = bytearray(raw)
                                invalid[2 * index:2 * index + 2] = b'\x81\xff'
                                compare(root, runner, source, plan, programs, [bytes(invalid)])


if __name__ == '__main__':
    unittest.main()
