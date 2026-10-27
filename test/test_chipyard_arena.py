"""Regression controls for DMA relocation and static physical ELF admission."""
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

TOOLS = Path(__file__).absolute().parents[1] / 'tools'
for directory in (TOOLS, TOOLS / 'host_iss', TOOLS / 'chipyard'):
    sys.path.insert(0, str(directory))
import arena
from firesim_target import package

BASE = 0x80000000


def elf():
    raw = bytearray(1024)
    struct.pack_into('<16sHHIQQQIHHHHHH', raw, 0, b'\x7fELF\x02\x01\x01', 2, 243, 1,
                     BASE, 64, 256, 4, 64, 56, 1, 64, 4, 0)
    struct.pack_into('<IIQQQQQQ', raw, 64, 1, 7, 0, BASE, BASE, 128, 128, 8)
    struct.pack_into('<IIQQQQIIQQ', raw, 320, 0, 8, 3, BASE + 16, 0, 32, 0, 0, 8, 0)
    names = ['tohost', 'fromhost', '__stack_start', '__stack_size', '_end', '__heap_end']
    table = b'\0' + b''.join(name.encode() + b'\0' for name in names)
    raw[512:512 + len(table)] = table
    struct.pack_into('<IIQQQQIIQQ', raw, 384, 0, 3, 0, 0, 512, len(table), 0, 0, 1, 0)
    struct.pack_into('<IIQQQQIIQQ', raw, 448, 0, 2, 0, 0, 640, 7 * 24, 2, 0, 8, 24)
    for i, value in enumerate([BASE + 64, BASE + 72, BASE + 256, 64, BASE + 384, BASE + 512], 1):
        struct.pack_into('<IBBHQQ', raw, 640 + i * 24, table.index(names[i - 1].encode()), 0, 0, 1, value, 8)
    return raw


def assemble(source):
    from npu_model.configs import isa_definition  # Register assembler metadata.
    from npu_model.util.converter import input_to_program
    return b''.join(struct.pack('<I', word & 0xffffffff) for word in input_to_program(io.StringIO(source)).assemble())


class ArenaTest(unittest.TestCase):







    def test_physical_dram_checks_memsz_sections_stack_and_heap(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'host.elf').write_bytes(elf())
            self.assertEqual(arena.audit(out, BASE, 1024)['reserved'],
                             [('stack', BASE + 256, BASE + 320), ('heap', BASE + 384, BASE + 512)])
            for offset, value, message in [(104, 1025, 'PT_LOAD'), (352, 1025, 'allocated section'),
                                            (640 + 3 * 24 + 8, BASE + 1000, 'stack'),
                                            (640 + 6 * 24 + 8, BASE + 1025, 'heap')]:
                raw = elf(); struct.pack_into('<Q', raw, offset, value)
                (out / 'host.elf').write_bytes(raw)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    arena.audit(out, BASE, 1024)

    def test_package_binds_selected_dram_and_rejects_abi_and_htif(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'host.elf').write_bytes(elf())
            contract = package(out, 'fixture', BASE, 1024)
            self.assertEqual((contract['dram_base'], contract['dram_bytes']), (BASE, 1024))
            self.assertFalse(contract['hardware_execution'])
            for key in ('cpu_mhz', 'atlas_mhz', 'vmem_bytes', 'imem_bytes', 'dma'):
                self.assertNotIn(key, contract)
            with self.assertRaises(TypeError):
                package(out, 'fixture')
            self.assertEqual(json.loads((out / 'marshal.json').read_text())['bin'], str(out / 'host.elf'))
            for offset, value, fmt, message in [(48, 0, '<I', 'ABI'), (672, BASE + 65, '<Q', 'HTIF'),
                                                (696, BASE + 64, '<Q', 'aliased HTIF')]:
                raw = elf(); struct.pack_into(fmt, raw, offset, value)
                (out / 'host.elf').write_bytes(raw)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    package(out, 'fixture', BASE, 1024)


if __name__ == '__main__':
    unittest.main()
