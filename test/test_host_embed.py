"""Host header generation binds checked words to the selected map."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("atlas_host_embed", ROOT / "tools/atlas_host_embed.py")
assert spec and spec.loader
embed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(embed)


class HostEmbedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.words = struct.pack("<III", 0x13, 0x73, 0x8067)
        self.host_map = json.loads((ROOT / "runtime/selected-host-map.json").read_text())
        self.manifest = {
            "schema": "atlas.boot-capsule.v1",
            "program_sha256": hashlib.sha256(self.words).hexdigest(),
            "program_words": 3,
            "entry_pc_word": 0,
            "completion": {"kind": "ecall_halt"},
            "imem_tl_byte_base": 0x20000,
            "imem_capacity_bytes": 0x20000,
            "start_csr_tl_byte_address": 0x18,
            "dram": {
                "selected_rtl_revision": self.host_map["rtl_revision"],
                "regions": [
                    {"name": "input", "address": 0x90000000, "size_bytes": 0x1000},
                    {"name": "output", "address": 0x90010000, "size_bytes": 0x800},
                ],
            },
        }
        self.memmap = {"mapping": [
            {"base": [0x20000], "size": [0x20000], "r": [True], "w": [True],
             "x": [False], "c": [False], "names": ["atlas-imem"]},
            {"base": [0x40000], "size": [0x1000], "r": [True], "w": [True],
             "x": [False], "c": [False], "names": ["atlas-csr"]},
            {"base": [0x90000000], "size": [0x20000], "r": [True], "w": [True],
             "x": [False], "c": [True], "names": ["dram"]},
        ]}

    def test_header_binds_words_and_selected_map(self) -> None:
        self.assertEqual(
            self.host_map["soc_config_source"],
            "generators/chipyard/src/main/scala/config/AtlasConfigs.scala::EE290SimConfig",
        )
        self.assertEqual(self.host_map["soc_source_revision"],
                         "245f2a9d203be685176cc01e26e28029c5593202")
        rendered = embed.render(self.manifest, self.words, self.host_map)
        self.assertIn("0x00000073u", rendered)
        self.assertIn("0x40000u", rendered)
        self.assertIn("ATLAS_PROGRAM_BITSTREAM_BOUND 0", rendered)
        self.assertIn("::EE290SimConfig", rendered)
        self.assertIn(self.host_map["soc_source_revision"], rendered)
        self.assertIn(self.manifest["program_sha256"], rendered)

    def test_rejects_changed_program_and_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "hash"):
            embed.render(self.manifest, self.words[:-4] + b"\x00" * 4, self.host_map)
        changed = dict(self.host_map, start_csr_offset=0x14)
        with self.assertRaisesRegex(ValueError, "MMIO"):
            embed.render(self.manifest, self.words, changed)
        changed = dict(self.host_map, rtl_revision="0" * 40)
        with self.assertRaisesRegex(ValueError, "different RTL"):
            embed.render(self.manifest, self.words, changed)
        changed = dict(self.host_map, soc_source_revision="unknown")
        with self.assertRaisesRegex(ValueError, "SoC source revision"):
            embed.render(self.manifest, self.words, changed)

    def test_generated_soc_map_covers_host_visible_regions(self) -> None:
        checked = embed.check_soc_memmap(self.host_map, self.manifest, self.memmap)
        self.assertEqual(checked["imem"]["mapping_names"], ["atlas-imem"])
        self.assertEqual(checked["csr"]["mapping_names"], ["atlas-csr"])
        self.assertEqual([row["name"] for row in checked["dram"]], ["input", "output"])
        self.assertTrue(all(row["cacheable"] for row in checked["dram"]))

    def test_bitstream_handoff_pins_exact_generated_map_bytes(self) -> None:
        map_bytes = json.dumps(self.memmap).encode()
        bound_map = dict(self.host_map,
                         soc_memmap_sha256=hashlib.sha256(map_bytes).hexdigest())
        embed.check_soc_memmap_hash(bound_map, map_bytes)
        with self.assertRaisesRegex(ValueError, "memmap hash"):
            embed.check_soc_memmap_hash(bound_map, map_bytes + b" ")
        with self.assertRaisesRegex(ValueError, "memmap hash"):
            embed.check_soc_memmap_hash(self.host_map, map_bytes)

    def test_soc_map_rejects_missing_or_misconfigured_ranges(self) -> None:
        with self.assertRaisesRegex(ValueError, "Atlas IMEM"):
            embed.check_soc_memmap(
                self.host_map, self.manifest,
                {"mapping": self.memmap["mapping"][1:]},
            )
        with self.assertRaisesRegex(ValueError, "DRAM output"):
            embed.check_soc_memmap(
                self.host_map, self.manifest,
                {"mapping": self.memmap["mapping"][:2] + [
                    dict(self.memmap["mapping"][2], size=[0x10000])]},
            )
        with self.assertRaisesRegex(ValueError, "Atlas status/start CSR"):
            embed.check_soc_memmap(
                self.host_map, self.manifest,
                {"mapping": [self.memmap["mapping"][0],
                             dict(self.memmap["mapping"][1], c=[True]),
                             self.memmap["mapping"][2]]},
            )
        with self.assertRaisesRegex(ValueError, "SoC memmap has an invalid"):
            embed.check_soc_memmap(
                self.host_map, self.manifest,
                {"mapping": [dict(self.memmap["mapping"][0], base=[True])]},
            )

    def test_final_host_elf_entry_and_load_segments_use_mapped_dram(self) -> None:
        elf = bytearray(0x104)
        elf[:7] = b"\x7fELF\x02\x01\x01"
        struct.pack_into("<HHIQQQIHHHHHH", elf, 16,
                         2, 243, 1, 0x80000000, 64, 0, 0, 64, 56, 1, 0, 0, 0)
        struct.pack_into("<IIQQQQQQ", elf, 64,
                         1, 5, 0x100, 0x80000000, 0x80000000, 4, 8, 0x100)
        elf[0x100:] = b"\x13\x00\x00\x00"
        dram = {"base": [0x80000000], "size": [0x1000], "r": [True],
                "w": [True], "x": [True], "c": [True], "names": ["host-dram"]}
        memmap = {"mapping": self.memmap["mapping"] + [dram]}
        checked = embed.check_host_elf(bytes(elf), memmap)
        self.assertEqual(checked["entry"], 0x80000000)
        self.assertEqual(checked["load_segments"][0]["mapping_names"], ["host-dram"])
        with self.assertRaisesRegex(ValueError, "host ELF LOAD segment 0"):
            embed.check_host_elf(bytes(elf), self.memmap)
        no_execute = {"mapping": self.memmap["mapping"] + [dict(dram, x=[False])]}
        with self.assertRaisesRegex(ValueError, "host ELF LOAD segment 0"):
            embed.check_host_elf(bytes(elf), no_execute)
        with self.assertRaisesRegex(ValueError, "entry is not"):
            changed = bytearray(elf)
            struct.pack_into("<Q", changed, 24, 0x80000040)
            embed.check_host_elf(bytes(changed), memmap)
        with self.assertRaisesRegex(ValueError, "invalid for bare-metal"):
            changed = bytearray(elf)
            struct.pack_into("<I", changed, 68, 1)  # remove PF_R from the segment
            embed.check_host_elf(bytes(changed), memmap)
        with self.assertRaisesRegex(ValueError, "overlap"):
            changed = bytearray(elf)
            struct.pack_into("<H", changed, 56, 2)
            struct.pack_into("<IIQQQQQQ", changed, 120,
                             1, 6, 0x104, 0x80000004, 0x80000004, 0, 8, 1)
            embed.check_host_elf(bytes(changed), memmap)
        with self.assertRaisesRegex(ValueError, "truncated"):
            embed.check_host_elf(bytes(elf[:80]), memmap)


if __name__ == "__main__":
    unittest.main()
