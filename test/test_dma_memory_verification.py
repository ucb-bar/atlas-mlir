"""Captured DMA ranges versus actual selected-RTL memory addresses.

DMA bases are word indices; VLS immediates advance by 32 words. Both select 16-bit VMEM line indices after
dropping three word-address bits.
"""

from __future__ import annotations

import unittest

from generated_fixture import artifact as program, branch_to, jump_to, label
from verification_support import DELAY, TRANSFER, assert_boundaries, constant, dma_wait as wait


CONFLICT = "DMA memory conflict"
UNKNOWN = "cannot prove DMA memory disjointness"
# Operands no contract can describe (unknown values, non-tile lengths) still exercise the memory check, which runs
# before the DMA contract check that then rejects them.
UNCONTRACTED = "DMA contract"


def launch(direction: str = "load", *, channel: int = 0, identity: int | None = 0, base: int = 4, dram: int = 7) -> tuple[str, str]:
    tag = "" if identity is None else f", {TRANSFER} = {identity} : i32"
    return "dma", f'direction = "{direction}", channel = {channel} : i32, reg = {base} : i32, dram = {dram} : i32, size = 9 : i32{tag}'


def vector(kind: str = "vload", *, base: int = 5, offset: int = 0) -> list[tuple[str, str]]:
    operand = "dst" if kind == "vload" else "src"
    return [(kind, f'{operand} = 0 : i32, base = {base} : i32, offset = {offset} : i32, format = "raw"'), DELAY]


def at(address: int, kind: str = "vload", offset: int = 0) -> list[tuple[str, str]]:
    return [*constant(5, address), *vector(kind, offset=offset)]


def prefix(*, base: int | None = 0x2000, size: int | None = 1024, shared: bool = False) -> list[tuple[str, str]]:
    ops = constant(7, 0x90000000)
    if base is not None:
        ops += constant(4, base)
    if size is not None:
        ops += constant(9, size)
    if shared:
        ops += constant(6, 0)
        ops += [("dma_config", f"channel = {channel} : i32, base_reg = 6 : i32") for channel in (0, 1)]
    return ops


def artifact(work=(), *, direction="load", base=0x2000, size=1024) -> str:
    return program([*prefix(base=base, size=size), launch(direction), *work, wait(), ("trap", 'kind = "ecall"')])


class DMAMemoryVerificationTest(unittest.TestCase):
    def checked(self, source: str, diagnostic: str | None = None) -> None:
        assert_boundaries(self, source, rejects=diagnostic)

    def test_empty_generated_artifact_has_no_dma_memory_obligations(self) -> None:
        self.checked(program([]))

    def test_vector_access_against_one_pending_transfer(self) -> None:
        cases = [
            # The captured base register may be overwritten for a disjoint access.
            ([*constant(4, 0x2100), *vector("vload", base=4)], {}, None),
            ([*constant(4, 0x2100), *vector("vstore", base=4)], {}, None),
            # Actual ranges detect aliases independently of register names.
            (at(0x2000), {}, CONFLICT), (at(0x2000, "vstore"), {}, CONFLICT), (at(0x2000, "vstore"), dict(direction="store"), CONFLICT),
            # The memory check admits a VLOAD of the half a pending store reads; the later buffer check rejects the fixture
            # because no single writer serves a DMA load reader and a VSTORE reader.
            (at(0x2000), dict(direction="store"), "buffer contract: DMA store captures a word its VSTORE did not write"),
            # Signed VLS offsets are 128 bytes each.
            (at(0x2000, offset=8), {}, None), (at(0x1F00, offset=8), {}, CONFLICT),
            (at(0x2000, offset=-8), {}, None), (at(0x2100, offset=-8), {}, CONFLICT),
            # VMEM line and DMA length masks apply before comparison.
            (at(0x82000), {}, CONFLICT), (at(0x2000), dict(base=0x82000), CONFLICT),
            (at(0x2100), dict(size=0x2400), UNCONTRACTED), (at(0x2000), dict(size=0x2400), CONFLICT),
            ((), dict(size=8192), "DMA memory transfer has zero complete beats"),
            ((), dict(base=0x5FF00), None), ((), dict(base=0x60000), "DMA memory transfer exceeds VMEM capacity"), (at(0x5FF00), {}, None),
            (at(0x2100, offset=1), {}, "vector DMA memory access has invalid VMEM span"),
            (at(0x60000), {}, "vector DMA memory access has invalid VMEM span"),
            # Unknown operands require proof only for potential write conflicts.
            ((), dict(base=None, size=None), UNCONTRACTED), (vector(), dict(direction="store", base=None, size=None), UNCONTRACTED),
            (vector(), {}, UNKNOWN), (at(0x2100), dict(base=None), UNKNOWN), (at(0x2100), dict(size=None), UNKNOWN),
        ]
        for index, (work, options, diagnostic) in enumerate(cases):
            with self.subTest(case=index, options=options):
                self.checked(artifact(work, **options), diagnostic)

    def test_matching_wait_releases_memory_ownership(self) -> None:
        self.checked(program([*constant(5, 0x2000), *prefix(), launch(), wait(), *vector(base=4), *constant(4, 0x2100),
                              launch(identity=1), *vector(base=5), wait(identity=1)]))

    def test_cfg_merges_and_backedges_require_constants_from_every_predecessor(self) -> None:
        for right, diagnostic in ((0x2000, None), (0x2100, UNKNOWN)):
            with self.subTest(merge=right):
                self.checked(program([*prefix(base=None), *constant(5, 0x2100), branch_to("right"), *constant(4, 0x2000), jump_to("join"),
                                      label("right"), *constant(4, right), label("join"), launch(), *vector(), wait()]), diagnostic)
        for next_base, diagnostic in ((0x2000, None), (0x2100, UNKNOWN)):
            with self.subTest(backedge=next_base):
                self.checked(program([*prefix(), *constant(5, 0x2100), label("loop"), launch(), *vector(), wait(),
                                      *constant(4, next_base), branch_to("loop")]), diagnostic)

    def test_two_pending_captured_windows_and_only_matching_wait_release(self) -> None:
        for base, diagnostic in ((0x2000, None), (0x2100, CONFLICT)):
            with self.subTest(released=base):
                self.checked(program([*prefix(), launch(), *constant(4, 0x2100), launch(channel=1, identity=1),
                                      wait(), *at(base), wait(channel=1, identity=1)]), diagnostic)
        for base in (0x2000, 0x82000):
            with self.subTest(alias_base=base):
                self.checked(program([*prefix(), *constant(5, base), launch(), launch(channel=1, identity=1, base=5),
                                      wait(channel=1, identity=1), wait()]), CONFLICT)

    def test_concurrent_dram_ranges_and_read_only_vmem_aliases(self) -> None:
        # The memory check admits concurrent reads of one VMEM range; the timing provider, which runs last, still
        # serializes them. Untagged stores may capture one staged half, which each explicit transfer would own.
        for direction, address, base, diagnostic in (("store", 0x90001000, 0x2000, "may still be in flight"),
                                                     ("store", 0x9000001F, 0x2000, CONFLICT + " in DRAM"),
                                                     ("load", 0x90000000, 0x2100, CONFLICT + " in DRAM")):
            with self.subTest(direction=direction, dram=address):
                self.checked(program([*prefix(shared=True), *constant(5, base), *constant(8, address), launch("store", identity=None),
                                      launch(direction, channel=1, identity=None, base=5, dram=8), wait(channel=1, identity=None),
                                      wait(identity=None)]), diagnostic)
        dram = constant(8, 0x90001000)
        second = [*constant(4, 0x2100), launch("store", channel=1, identity=1, dram=8), wait(channel=1, identity=1), wait()]
        self.checked(program([*prefix(base=None, shared=True), *dram, launch("store"), *second]), UNCONTRACTED)
        self.checked(program([*prefix(base=None, shared=True), *dram, launch(), *second]), UNKNOWN)
        # A nonzero upper DMA base depends on negotiated TileLink address width;
        # this checker cannot use it to prove disjointness of writes.
        self.checked(program([*prefix(), *constant(6, 1), ("dma_config", "channel = 0 : i32, base_reg = 6 : i32"),
                              *dram, launch("store"), *second]), UNKNOWN + " in DRAM")
        for size, diagnostic in ((32, UNCONTRACTED), (64, UNKNOWN + " in DRAM")):
            with self.subTest(dram_end_crosses_32_bits=size == 64):
                self.checked(program([*prefix(size=size, shared=True), *constant(7, 0xFFFFFFE0), *dram, launch("store"), *second]), diagnostic)


if __name__ == "__main__":
    unittest.main()
