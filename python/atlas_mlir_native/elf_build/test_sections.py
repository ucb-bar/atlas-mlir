"""Tests for ELF section accounting without an external toolchain."""

import hashlib
from pathlib import Path
import struct
import tempfile
import unittest

from atlas_mlir_native.elf_build.sections import measure


def _elf(*, machine=243, elf_type=2):
    """Build an ELF64 image with named allocated and unallocated sections."""
    names = b"\0.text\0.bss\0.debug_info\0.empty\0.shstrtab\0"
    header_size = 64
    text_offset = header_size
    debug_offset = text_offset + 4
    names_offset = debug_offset + 3
    section_offset = (names_offset + len(names) + 7) & ~7

    def section(name, kind, flags, address, offset, size, alignment):
        return struct.pack(
            "<IIQQQQIIQQ", names.index(name) if name else 0, kind, flags,
            address, offset, size, 0, 0, alignment, 0,
        )

    sections = [
        bytes(64),
        section(b".text", 1, 6, 0x80000000, text_offset, 4, 16),
        section(b".bss", 8, 3, 0x80000040, section_offset, 96, 64),
        section(b".debug_info", 1, 0, 0, debug_offset, 3, 1),
        section(b".empty", 1, 2, 0x800000A0, text_offset, 0, 32),
        section(b".shstrtab", 3, 0, 0, names_offset, len(names), 1),
    ]
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", ident, elf_type, machine, 1, 0x80000000,
        0, section_offset, 0, header_size, 0, 0, 64, len(sections), 5,
    )
    payload = header + b"\x13\0\0\0" + b"dbg" + names
    return payload.ljust(section_offset, b"\0") + b"".join(sections)


class MeasureTests(unittest.TestCase):
    def measure_bytes(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.elf"
            path.write_bytes(content)
            return measure(path)

    def test_allocated_sections_include_nobits_and_zero_size(self):
        content = _elf()
        result = self.measure_bytes(content)
        self.assertEqual(result["sha256"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["allocated_bytes"], 100)
        sections = {section["name"]: section for section in result["sections"]}
        self.assertEqual(set(sections), {".text", ".bss", ".empty"})
        self.assertEqual(sections[".text"]["size"], 4)
        self.assertEqual(sections[".text"]["address"], 0x80000000)
        self.assertEqual(sections[".text"]["alignment"], 16)
        self.assertEqual(sections[".bss"]["size"], 96)
        self.assertEqual(sections[".bss"]["alignment"], 64)
        self.assertEqual(sections[".empty"]["size"], 0)
        self.assertEqual(sections[".empty"]["alignment"], 32)
        for section in sections.values():
            self.assertIn("type", section)
            self.assertIn("flags", section)
        self.assertIn("elf_type", result)

    def test_rejects_invalid_or_truncated_input(self):
        content = _elf()
        for invalid in (b"", b"not an ELF file", content[:32], content[:-1]):
            with self.subTest(length=len(invalid)):
                with self.assertRaises(ValueError):
                    self.measure_bytes(invalid)

    def test_rejects_wrong_machine(self):
        with self.assertRaises(ValueError):
            self.measure_bytes(_elf(machine=62))

    def test_rejects_unsupported_class_or_byte_order(self):
        for index, value in ((4, 1), (5, 2)):
            content = bytearray(_elf())
            content[index] = value
            with self.subTest(ident_index=index):
                with self.assertRaises(ValueError):
                    self.measure_bytes(bytes(content))


if __name__ == "__main__":
    unittest.main()
