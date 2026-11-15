"""Larger operand panels retain ordered rounding and reduce partial traffic."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atlas_mlir_native import quantized_panel_reuse as panel
from atlas_mlir_native.already_quantized import prepare_source
from atlas_mlir_native.binding import DRAM_BASE
from atlas_mlir_native.quantized_loop import execution_layout, virtual_source
from atlas_mlir_native.restoration_budget import instruction_bound
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from test_already_quantized import source_request
from test_exponent_restoration import compile_fixture
from test_quantized_prefix_reuse import memory_fixture


@unittest.skipUnless(os.environ.get('ATLAS_RUST_NPU_MODEL') and os.environ.get('ATLAS_OOT_BIN_DIR'),
                     'select Rust simulator and Atlas tools')
class LargerPanelTest(unittest.TestCase):
    def test_rust_panel_boundaries_tails_row_pairs_and_rejected_exponents(self):
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            for m, n, k in ((33, 35, 1024), (33, 35, 1025), (97, 65, 2049)):
                with self.subTest(m=m, n=n, k=k), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    plan = prepare_source(source_request(m=m, n=n, k=k, batch=()))
                    self.assertTrue(panel.admitted(plan))
                    binaries = []
                    for width in (32, panel.PANEL_K_TILES):
                        with patch.object(panel, 'PANEL_K_TILES', width):
                            layout = execution_layout(plan, operand_reuse='paired-cache')
                            virtual = virtual_source(plan, layout)
                            machine, physical = compile_fixture(virtual)
                        self.assertTrue(instruction_bound(virtual, machine, physical)['admitted'])
                        binary = b''.join(row['word_u32'].to_bytes(4, 'little')
                                          for row in physical['instructions'])
                        self.assertLessEqual(len(binary), 131072)
                        program = root / f'panel-{width}.bin'
                        program.write_bytes(binary)
                        binaries.append(program)
                    initial = memory_fixture(plan, layout)
                    initial.extend(bytes([0xa5]) * (-len(initial) % 4096))
                    config = root / 'config.json'
                    config.write_text(json.dumps({'compute': 'cpu', 'pack_layout': 'register_rows',
                                                  'dram_base': DRAM_BASE, 'dram_size': len(initial)}))
                    for rejected in (False, True):
                        if rejected:
                            at = layout['exponent_addresses'][1] - DRAM_BASE
                            initial[at:at + 4] = (1000).to_bytes(4, 'little')
                        memory = root / 'memory.bin'
                        memory.write_bytes(initial)
                        outputs, cycles = [], []
                        for index, program in enumerate(binaries):
                            state = root / f'state-{int(rejected)}-{index}'
                            result = subprocess.run([os.environ['ATLAS_RUST_NPU_MODEL'], str(program),
                                '--config', str(config), '--load', f'{DRAM_BASE}:{memory}',
                                '--dump-state', str(state)], check=True, capture_output=True, text=True)
                            cycles.append(json.loads(result.stdout)['cycles'])
                            actual = b''.join((state / f'dram_{a:016x}.bin').read_bytes()
                                             for a in range(DRAM_BASE, DRAM_BASE + len(initial), 4096))
                            start = layout['output_addresses'][0] - DRAM_BASE
                            end = layout['status_address'] - DRAM_BASE + layout['status_bytes']
                            self.assertEqual(actual[:start], initial[:start])
                            self.assertEqual(actual[end:], initial[end:])
                            status = layout['status_address'] - DRAM_BASE
                            self.assertEqual(bool(int.from_bytes(actual[status:status + 4], 'little')), rejected)
                            outputs.append(actual)
                        self.assertEqual(outputs[0], outputs[1])
                        if k > 1024 and not rejected:
                            self.assertLess(cycles[1], cycles[0])


if __name__ == '__main__':
    unittest.main()
