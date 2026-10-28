"""Full-image relocation, semantic stop, and incomplete UART prefix controls."""

from pathlib import Path
import struct
import tempfile
import json
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tools/host_iss')]
from tiny_prefix import live_mapping, verify_prefix_uart, PrefixBoundary, digest, EPILOGUE
from test_host_iss_tiny_segments import image_fixture
import run


def full_image():
    image, data = image_fixture()
    image.symbols['workspace'] = image.symbols.pop('tiny_full_segment_workspace')
    image.symbols['tiny_full_state_global'] = dict(address=0x83800100, size=48)
    layout = image.bytes_at.return_value
    state = struct.pack('<5QII', 0x83000000, 0x83000000 + 1802240,
                        0x83400000, 0x83400000 + 1802240, 0, 0, 0)
    image.bytes_at.side_effect = lambda address, size: state if size == 48 else layout
    return image, data


class TinyPrefixTest(unittest.TestCase):
    def test_full_image_mapping_uses_actual_state_pointers_and_workspace(self):
        image, data = full_image()
        mapping, _ = live_mapping(image, data)
        self.assertEqual(len(mapping), 143)
        self.assertEqual(mapping['hidden_1'], (0x81000000 + 16778880, 524288))
        self.assertEqual(mapping['keys_1'], (0x83000000 + 1802240, 1802240))
        image.symbols['tiny_full_keys']['address'] = 123
        self.assertEqual(live_mapping(image, data)[0], mapping)

    def test_full_mapping_rejects_alias_outside_load_or_readonly(self):
        for kind in ('alias', 'outside', 'readonly'):
            image, data = full_image()
            original = image.bytes_at.side_effect
            if kind == 'alias':
                image.bytes_at.side_effect = lambda address, size: (struct.pack('<5QII', *([0x83000000]*4), 0, 0, 0)
                                                                  if size == 48 else original(address, size))
            elif kind == 'outside':
                image.symbols['workspace']['address'] = 0x100000000
            else:
                image.segments[1]['flags'] = 4
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                live_mapping(image, data)

    def test_uart_prefix_requires_begin_then_exact_layer_progress(self):
        build = dict(result_name='tiny-full', result_progress_launches=[12, 24, 840])
        begin = b'ATLAS_KERNEL_BEGIN name=tiny-full launches_expected=840 launches_completed=0\n'
        progress = b'ATLAS_KERNEL_PROGRESS name=tiny-full launches_expected=840 launches_completed=12 boundary=layer\n'
        self.assertEqual(verify_prefix_uart(begin + progress, build, 12)['launches_completed'], 12)
        for raw in (begin, progress, begin + progress.replace(b'completed=12', b'completed=11'),
                    begin + progress + progress, begin + progress.replace(b'PROGRESS', b'FAIL')):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                verify_prefix_uart(raw, build, 12)

    def test_twelfth_device_completion_does_not_establish_layer_boundary(self):
        prefix = PrefixBoundary.__new__(PrefixBoundary)
        prefix.launches = 12
        prefix.image = SimpleNamespace(symbols={name:dict(address=value) for name, value in
            [('tiny_full_state_global', 100), ('tiny_full_projection_state', 200), ('workspace', 300),
             ('__stack_bottom', 400), ('__stack_top', 800)]})
        registers = {11:100, 12:500, 13:0, 15:200, 16:300}
        remote = SimpleNamespace(reg=lambda r:registers[r])
        self.assertFalse(prefix.at_boundary(remote, SimpleNamespace(completed=[])))
        with self.assertRaises(ValueError):
            prefix.at_boundary(remote, SimpleNamespace(completed=[{}]*12))
        registers[13] = 1
        self.assertTrue(prefix.at_boundary(remote, SimpleNamespace(completed=[{}]*12)))
        registers[12] = 790
        with self.assertRaises(ValueError):
            prefix.at_boundary(remote, SimpleNamespace(completed=[{}]*12))

    def test_all_regions_dumped_and_scalar_corruption_or_cycle_disagreement_rejected(self):
        for failure in (None, 'hidden', 'cycles'):
            image, data = full_image()
            mapping, _ = live_mapping(image, data)
            image.symbols.update({name:dict(address=address) for name, address in
                [('tiny_full_projection_state', 0x83800200), ('tiny_full_mailbox', 0x83800300),
                 ('tiny_full_host', 0x83800400), ('tohost', 0x83800500)]})
            prefix = PrefixBoundary.__new__(PrefixBoundary)
            prefix.launches, prefix.pc, prefix.phase_address = 12, 0x8000030c, 0x83800600
            prefix.image, prefix.mapping, prefix.heartbeat = image, mapping, lambda _:None
            phase = struct.pack('<Q6I', 0, 0, 0, 1, 128, 1, 0)
            control = struct.pack('<8Q', 0, 1, 0, 0, 0, 128, 7, 12)
            payloads = {name:bytes([127 if name == 'hidden_1' else 0])*size for name, (_, size) in mapping.items()}
            payloads.update(phase=phase, control=control)
            prefix.chain = SimpleNamespace(boundaries=[{}, dict(regions={name:name for name in payloads})],
                                           read=lambda name:payloads[name])
            memory = {address:payloads[name] for name, (address, _) in mapping.items()}
            if failure == 'hidden':
                address, _ = mapping['hidden_1']
                memory[address] = memory[address][:-1] + b'X'
            state = image.bytes_at(image.symbols['tiny_full_state_global']['address'], 48)
            context = struct.pack('<9Q', 1, 840, 12, 0x100000000, 6750720, 2, 0, 7, 3)
            mailbox = struct.pack('<16Q', 0x54494E5946554C4C, 1, 12, 11, *([0]*12))
            host = bytearray(112)
            struct.pack_into('<6I', host, 88, 1, 0, 0, 0, 1, 12)
            memory.update({image.symbols['tiny_full_state_global']['address']:state,
                image.symbols['tiny_full_projection_state']['address']:context,
                image.symbols['tiny_full_mailbox']['address']:mailbox,
                image.symbols['tiny_full_host']['address']:bytes(host), prefix.phase_address:phase,
                image.symbols['tohost']['address']:bytes(8)})
            prefix.physical = SimpleNamespace(read=lambda address, size:memory[address][:size])
            prefix.bundle = SimpleNamespace(read=lambda _:bytes.fromhex('73000000'), sums={'expected':digest(b'E'*32)})
            item = dict(original='program', initial_sha256=digest(b'I'), expected='expected',
                        requested_dma=dict(load_bytes=32, store_bytes=32))
            prefix.data = dict(kernels=[item]*12)
            completed = [dict(index=i, sequence=i+1, reset_count=1, cycles=1, exact=True, drained=True,
                program_sha256=digest(EPILOGUE), host_input_sha256=digest(b'I'),
                actual_output_sha256=digest(b'E'*32), expected_output_sha256=digest(b'E'*32),
                observed_dma=dict(load_bytes=32, store_bytes=32, response_beats=2)) for i in range(12)]
            device = SimpleNamespace(completed=completed, core=SimpleNamespace(cycle_count=13 if failure == 'cycles' else 12,
                arch_state=SimpleNamespace(read_csrf=lambda _:12)))
            remote = SimpleNamespace(reg=lambda r:prefix.pc if r == 32 else 1)
            with tempfile.TemporaryDirectory() as directory:
                prefix.out = Path(directory)
                if failure:
                    with self.assertRaisesRegex(ValueError, 'live prefix' if failure == 'hidden' else 'cycle counters'):
                        prefix.verify(remote, device)
                else:
                    self.assertTrue(prefix.verify(remote, device)['all_live_regions_exact'])
                self.assertEqual(len(json.loads((prefix.out/'boundary-comparison.json').read_text())), 145)
                self.assertEqual(len(list((prefix.out/'raw').iterdir())), 149)

    def test_prefix_flag_is_opt_in_and_only_full_tiny(self):
        base = ['--out','x','--elf','e','--bundle-root','b','--model-root','m','--qemu','q']
        self.assertIsNone(run.arguments(base).tiny_prefix_chain)
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            run.arguments(base + ['--tiny-prefix-chain','chain'])
        self.assertEqual(run.arguments(base + ['--workload','tiny-full','--wall-seconds','1',
            '--tiny-prefix-chain','chain']).tiny_prefix_chain, 'chain')


if __name__ == '__main__':
    unittest.main()
