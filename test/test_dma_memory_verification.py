"""Captured DMA ranges versus actual selected-RTL memory addresses.

DMA bases are word indices; VLS immediates advance by 32 words. Both select
16-bit VMEM line indices after dropping three word-address bits. Tests use the
compiler verifier, emitter, and LLVM handoff, without a runtime model bundle.
"""

from __future__ import annotations

import re
import unittest

import generated_fixture
from test_virtual_lowering import run
from verification_support import DELAY, NOP, STATE, TRANSFER, assert_boundaries, dma_wait as wait


CONFLICT = "DMA memory conflict"
UNKNOWN = "cannot prove DMA memory disjointness"
# Generated artifacts must also match every captured operand to a source tile contract.
# Operands no contract can describe (unknown values, non-tile lengths) still exercise the
# memory check, which runs first; the contract check then rejects them.
UNCONTRACTED = "DMA contract"


def constant(reg: int, value: int) -> list[tuple[str, str]]:
    low = ((value + 2048) & 4095) - 2048
    return [("upper", f'kind = "lui", dst = {reg} : i32, immediate = {((value + 2048) >> 12) & 0xFFFFF} : i32'),
            ("alu_imm", f'kind = "addi", dst = {reg} : i32, src = {reg} : i32, immediate = {low} : i32')]


def launch(direction: str = "load", *, channel: int = 0, identity: int | None = 0, base: int = 4, dram: int = 7) -> tuple[str, str]:
    tag = "" if identity is None else f", {TRANSFER} = {identity} : i32"
    return "dma", f'direction = "{direction}", channel = {channel} : i32, reg = {base} : i32, dram = {dram} : i32, size = 9 : i32{tag}'


def vector(kind: str = "vload", *, base: int = 5, offset: int = 0) -> list[tuple[str, str]]:
    operand = "dst" if kind == "vload" else "src"
    return [(kind, f'{operand} = 0 : i32, base = {base} : i32, offset = {offset} : i32, format = "raw"'), DELAY]


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


def program(operations: list[tuple[str, str] | str]) -> str:
    """A generated artifact; strings are labels and `@label` redirect targets become source edges."""
    stream = []
    for item in operations:
        if isinstance(item, str):
            stream.append(generated_fixture.label(item))
        elif (target := re.search(r"@(\w+)", item[1])) is not None:
            stream.append(generated_fixture.branch_to(target[1]) if item[0] == "branch" else generated_fixture.jump_to(target[1]))
        else:
            stream.append(item)
    return generated_fixture.artifact(stream)


def hand_written(operations: list[tuple[str, str]]) -> str:
    lines = [f'%s0 = "atlas.start"() : () -> {STATE}']
    lines += [f'%s{pc + 1} = "atlas.{name}"(%s{pc}) {{{fields}}} : ({STATE}) -> {STATE}' for pc, (name, fields) in enumerate(operations)]
    return "module {\n" + "\n".join(lines) + "\n}"


def artifact(work=(), *, direction="load", base=0x2000, size=1024) -> str:
    return program([*prefix(base=base, size=size), launch(direction), *work, wait(), ("trap", 'kind = "ecall"')])


class DMAMemoryVerificationTest(unittest.TestCase):
    def checked(self, source: str, diagnostic: str | None = None) -> None:
        assert_boundaries(self, source, rejects=diagnostic)

    def test_empty_generated_artifact_has_no_dma_memory_obligations(self) -> None:
        self.checked(program([]))

    def test_hand_written_stream_without_metadata_has_no_generated_obligations(self) -> None:
        ops = [*constant(31, 0), *constant(4, 0x2000), ("alu_reg", 'kind = "add", dst = 5 : i32, lhs = 4 : i32, rhs = 4 : i32'),
               ("jump", 'kind = "jalr", dst = 0 : i32, base = 31 : i32, offset = 0 : i32'), NOP]
        source = hand_written(ops)
        result = run("atlas-opt", source, "--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected an Atlas virtual-to-machine artifact", result.stderr)
        for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",))):
            result = run(tool, source, *options)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout)

    def test_captured_base_can_be_overwritten_for_disjoint_vector_access(self) -> None:
        for kind in ("vload", "vstore"):
            with self.subTest(kind=kind):
                self.checked(artifact([*constant(4, 0x2100), *vector(kind, base=4)]))

    def test_actual_ranges_detect_aliases_independently_of_register_names(self) -> None:
        for direction, kind in (("load", "vload"), ("load", "vstore"), ("store", "vstore")):
            with self.subTest(direction=direction, kind=kind):
                self.checked(artifact([*constant(5, 0x2000), *vector(kind)], direction=direction), CONFLICT)
        # The memory check admits a VLOAD of the half a pending store reads; the buffer check, which runs
        # later, rejects the fixture because no single writer serves a DMA load reader and a VSTORE reader.
        self.checked(artifact([*constant(5, 0x2000), *vector()], direction="store"), "buffer contract: DMA store captures a word its VSTORE did not write")

    def test_signed_vls_offsets_are_128_bytes_each_and_boundary_ranges_are_disjoint(self) -> None:
        for offset, disjoint, aliased in ((8, 0x2000, 0x1F00), (-8, 0x2000, 0x2100)):
            with self.subTest(offset=offset):
                self.checked(artifact([*constant(5, disjoint), *vector(offset=offset)]))
                self.checked(artifact([*constant(5, aliased), *vector(offset=offset)]), CONFLICT)

    def test_vmem_line_mask_and_dma_length_mask_are_applied_before_comparison(self) -> None:
        self.checked(artifact([*constant(5, 0x82000), *vector()]), CONFLICT)
        self.checked(artifact([*constant(5, 0x2000), *vector()], base=0x82000), CONFLICT)
        self.checked(artifact([*constant(5, 0x2100), *vector()], size=0x2400), UNCONTRACTED)
        self.checked(artifact([*constant(5, 0x2000), *vector()], size=0x2400), CONFLICT)
        self.checked(artifact(size=8192), "DMA memory transfer has zero complete beats")
        self.checked(artifact(base=0x5FF00))
        self.checked(artifact(base=0x60000), "DMA memory transfer exceeds VMEM capacity")
        self.checked(artifact([*constant(5, 0x5FF00), *vector()]))
        for base, offset in ((0x2100, 1), (0x60000, 0)):
            with self.subTest(invalid_vector_base=base, offset=offset):
                self.checked(artifact([*constant(5, base), *vector(offset=offset)]), "vector DMA memory access has invalid VMEM span")

    def test_unknown_operands_require_proof_only_for_potential_write_conflicts(self) -> None:
        self.checked(artifact(base=None, size=None), UNCONTRACTED)
        self.checked(artifact(vector(), direction="store", base=None, size=None), UNCONTRACTED)
        self.checked(artifact(vector()), UNKNOWN)
        self.checked(artifact([*constant(5, 0x2100), *vector()], base=None), UNKNOWN)
        self.checked(artifact([*constant(5, 0x2100), *vector()], size=None), UNKNOWN)

    def test_matching_wait_releases_memory_ownership(self) -> None:
        ops = [*prefix(), launch(), wait(), *vector(base=4), *constant(4, 0x2100),
               launch(identity=1), *vector(base=5), wait(identity=1)]
        ops[0:0] = constant(5, 0x2000)
        self.checked(program(ops))

    def test_cfg_merges_and_backedges_require_constants_from_every_predecessor(self) -> None:
        branch = ("branch", 'kind = "beq", lhs = 10 : i32, rhs = 11 : i32, offset_bytes = @right : i32')
        jump = ("jump", 'kind = "jal", dst = 0 : i32, base = 0 : i32, offset = @join : i32')
        for right, diagnostic in ((0x2000, None), (0x2100, UNKNOWN)):
            with self.subTest(merge=right):
                ops = [*prefix(base=None), *constant(5, 0x2100), branch, NOP,
                       *constant(4, 0x2000), jump, NOP, "right", *constant(4, right), "join",
                       launch(), *vector(), wait()]
                self.checked(program(ops), diagnostic)
        for next_base, diagnostic in ((0x2000, None), (0x2100, UNKNOWN)):
            with self.subTest(backedge=next_base):
                ops = [*prefix(), *constant(5, 0x2100), "loop", launch(), *vector(), wait(),
                       *constant(4, next_base), ("branch", 'kind = "beq", lhs = 10 : i32, rhs = 11 : i32, offset_bytes = @loop : i32'), NOP]
                self.checked(program(ops), diagnostic)

    def test_two_pending_captured_windows_and_only_matching_wait_release(self) -> None:
        ops = [*prefix(), launch(), *constant(4, 0x2100), launch(channel=1, identity=1),
               wait(), *constant(5, 0x2000), *vector(), wait(channel=1, identity=1)]
        self.checked(program(ops))
        changed = [*prefix(), launch(), *constant(4, 0x2100), launch(channel=1, identity=1),
                   wait(), *constant(5, 0x2100), *vector(), wait(channel=1, identity=1)]
        self.checked(program(changed), CONFLICT)
        for base in (0x2000, 0x82000):
            with self.subTest(alias_base=base):
                self.checked(program([*prefix(), *constant(5, base), launch(), launch(channel=1, identity=1, base=5),
                                      wait(channel=1, identity=1), wait()]), CONFLICT)

    def test_concurrent_dram_ranges_and_read_only_vmem_aliases(self) -> None:
        # The memory check admits concurrent reads of one VMEM range; the timing provider,
        # which runs last, still serializes them. Untagged stores may capture one staged half,
        # which each explicit transfer would own.
        for direction, address, base, diagnostic in (("store", 0x90001000, 0x2000, "may still be in flight"), ("store", 0x9000001F, 0x2000, CONFLICT + " in DRAM"), ("load", 0x90000000, 0x2100, CONFLICT + " in DRAM")):
            with self.subTest(direction=direction, dram=address):
                ops = [*prefix(shared=True), *constant(5, base), *constant(8, address), launch("store", identity=None),
                       launch(direction, channel=1, identity=None, base=5, dram=8), wait(channel=1, identity=None), wait(identity=None)]
                self.checked(program(ops), diagnostic)
        self.checked(program([*prefix(base=None, shared=True), launch("store"), *constant(4, 0x2100), *constant(8, 0x90001000),
                              launch("store", channel=1, identity=1, dram=8), wait(channel=1, identity=1), wait()]), UNCONTRACTED)
        self.checked(program([*prefix(base=None, shared=True), launch(), *constant(4, 0x2100), *constant(8, 0x90001000),
                              launch("store", channel=1, identity=1, dram=8), wait(channel=1, identity=1), wait()]), UNKNOWN)
        # A nonzero upper DMA base depends on negotiated TileLink address width;
        # this checker cannot use it to prove disjointness of writes.
        ops = [*prefix(), *constant(6, 1), ("dma_config", "channel = 0 : i32, base_reg = 6 : i32"),
               *constant(8, 0x90001000), launch("store"), *constant(4, 0x2100),
               launch("store", channel=1, identity=1, dram=8), wait(channel=1, identity=1), wait()]
        self.checked(program(ops), UNKNOWN + " in DRAM")
        for size, diagnostic in ((32, UNCONTRACTED), (64, UNKNOWN + " in DRAM")):
            with self.subTest(dram_end_crosses_32_bits=size == 64):
                ops = [*prefix(size=size, shared=True), *constant(7, 0xFFFFFFE0), *constant(8, 0x90001000),
                       launch("store"), *constant(4, 0x2100), launch("store", channel=1, identity=1, dram=8),
                       wait(channel=1, identity=1), wait()]
                self.checked(program(ops), diagnostic)


if __name__ == "__main__":
    unittest.main()
