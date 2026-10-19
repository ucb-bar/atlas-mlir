"""DMA wait insertion diagnoses malformed tile metadata without asserting."""

from __future__ import annotations

import unittest

from test_virtual_lowering import run


def pending(tiles: str) -> str:
    return ("module attributes {atlas.dma_waits_pending, "
            "atlas.virtual_dma_contract = [], "
            f"atlas.virtual_tile_contract = {tiles}, "
            "atlas.virtual_cfg_contract = {operations = []}} {}")


class DMAWaitMetadataTest(unittest.TestCase):
    def rejected(self, tiles: str, diagnostic: str) -> None:
        result = run("atlas-opt", pending(tiles), "--insert-atlas-dma-waits")
        self.assertGreater(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertIn(diagnostic, result.stderr)

    def test_malformed_records(self) -> None:
        for record in ("0 : i32", '"bad"', "[]"):
            with self.subTest(record=record):
                self.rejected(f"[{record}]", "DMA wait insertion cannot read a tile contract record")

    def test_required_fields_and_unique_nonnegative_ids(self) -> None:
        fields = {"id": "0 : i32", "kind": '"vload"', "vmem_byte": "0 : i32",
                  "bytes": "1024 : i32", "transfer": "-1 : i32"}

        def record(values: dict[str, str]) -> str:
            return "{" + ", ".join(f"{key} = {value}" for key, value in values.items()) + "}"

        for name in fields:
            for value in (None, "0 : i64"):
                changed = fields.copy()
                if value is None:
                    del changed[name]
                else:
                    changed[name] = value
                with self.subTest(field=name, value=value):
                    self.rejected(f"[{record(changed)}]", "DMA wait insertion cannot read a tile contract record")
        for identity in (-1, -2, 2147483647):
            with self.subTest(identity=identity):
                self.rejected(f"[{record(dict(fields, id=f'{identity} : i32'))}]",
                              "DMA wait insertion cannot read a tile contract record")
        self.rejected(f"[{record(fields)}, {record(fields)}]", "DMA wait insertion requires unique tile contract ids")

    def test_reserved_dma_and_cfg_ids(self) -> None:
        for old, new, diagnostic in (
            ("atlas.virtual_dma_contract = []",
             "atlas.virtual_dma_contract = [{id = 2147483647 : i32, channel = 0 : i32, "
             "staging_word = 0 : i32, dram_byte = 0 : i32, size_bytes = 1024 : i32}]",
             "DMA wait insertion cannot read a DMA contract record"),
            ("operations = []",
             "operations = [{id = 0 : i32, block = 0 : i32, tile_commands = array<i32: 2147483647>}]",
             "DMA wait insertion cannot read a CFG tile command"),
        ):
            with self.subTest(diagnostic=diagnostic):
                result = run("atlas-opt", pending("[]").replace(old, new), "--insert-atlas-dma-waits")
                self.assertGreater(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn(diagnostic, result.stderr)


if __name__ == "__main__":
    unittest.main()
