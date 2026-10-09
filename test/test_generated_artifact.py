"""Every consumer classifies generated artifacts by one rule (lib/AtlasGeneratedArtifact.cpp).

Only the resource-contract-v4 marker with all five contracts and a timing state is a generated
artifact; a module without marker, contracts or tags is a hand-written stream; anything else
fails with one diagnostic.
"""

from __future__ import annotations

import unittest

from generated_fixture import INCOMPLETE, MARKER, UNMARKED, UNSUPPORTED
from test_virtual_dma import copy
from test_virtual_lowering import BIN, lower, run


CONTRACTS = ("atlas.virtual_dma_contract", "atlas.virtual_mxu_contract",
             "atlas.virtual_tile_contract", "atlas.virtual_cfg_contract",
             "atlas.virtual_source_memory_contract")
TIMED_BOUNDARIES = (("atlas-opt", ("--verify-atlas-generated-schedule",)),
                    ("atlas-emit", ()),
                    ("atlas-opt", ("--convert-atlas-to-llvm",)),
                    ("atlas-opt", ("--convert-atlas-to-llvm-calls",)))
UNTIMED_BOUNDARIES = (("atlas-opt", ("--insert-atlas-delays",)),
                      ("atlas-opt", ("--schedule-atlas-stream",)))
MARKERS = ('"resource-contract-v1"', '"resource-contract-v2"', '"resource-contract-v3"', '"dma-contract-v1"',
           '"resource-contract-v999"', '"resource-contract-v4 "', "1 : i32", "unit")


def drop_attribute(machine: str, name: str) -> str:
    """Remove one top-level module attribute, whose value may nest brackets."""
    header, rest = machine.split("\n", 1)
    start = header.index(name + " = ") if name + " = " in header else header.index(name)
    end, depth, quoted = start + len(name), 0, False
    if header.startswith(" = ", end):
        end += 3
        while quoted or depth or header[end] not in ",}":
            char = header[end]
            if char == '"' and header[end - 1] != "\\":
                quoted = not quoted
            elif not quoted and char in "[{<(":
                depth += 1
            elif not quoted and char in "]}>)":
                depth -= 1
            end += 1
    if header.startswith(", ", end):
        end += 2
    elif header[start - 2:start] == ", ":
        start -= 2
    return header[:start] + header[end:] + "\n" + rest


def remark(machine: str, marker: str) -> str:
    return machine.replace(MARKER, "atlas.generated_from_virtual" + ("" if marker == "unit" else " = " + marker), 1)


class GeneratedArtifactClassificationTest(unittest.TestCase):
    def setUp(self) -> None:
        for tool in ("atlas-opt", "atlas-emit"):
            self.assertTrue((BIN / tool).is_file(), f"build {tool} first")
        self.timed = lower(copy())
        self.untimed = lower(copy(), timed=False)
        self.assertIn(MARKER, self.timed)
        self.assertIn(MARKER, self.untimed)

    def rejected(self, machine: str, untimed: str | None, diagnostic: str) -> None:
        for source, boundaries in ((machine, TIMED_BOUNDARIES), (untimed, UNTIMED_BOUNDARIES)):
            if source is None:
                continue
            for tool, options in boundaries:
                with self.subTest(tool=tool, options=options):
                    result = run(tool, source, *options)
                    self.assertNotEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(diagnostic, result.stderr)

    def test_lowered_resource_contract_v4_is_accepted_everywhere(self) -> None:
        for source, boundaries in ((self.timed, TIMED_BOUNDARIES), (self.untimed, UNTIMED_BOUNDARIES)):
            for tool, options in boundaries:
                with self.subTest(tool=tool, options=options):
                    result = run(tool, source, *options)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertTrue(result.stdout)

    def test_legacy_unknown_and_malformed_markers_are_rejected(self) -> None:
        for marker in MARKERS:
            with self.subTest(marker=marker):
                changed = remark(self.timed, marker)
                self.assertNotEqual(changed, self.timed)
                self.rejected(changed, remark(self.untimed, marker), UNSUPPORTED)

    def test_generated_metadata_without_marker_is_rejected(self) -> None:
        unmarked = drop_attribute(self.timed, "atlas.generated_from_virtual")
        self.assertNotIn("atlas.generated_from_virtual", unmarked)
        self.rejected(unmarked, drop_attribute(self.untimed, "atlas.generated_from_virtual"), UNMARKED)
        # Tags alone, and each contract alone on an untagged stream, are metadata too.
        header, body = self.timed.split("\n", 1)
        self.rejected("module {\n" + body, None, UNMARKED)
        untagged = "\n".join(line.split(" {atlas.")[0] + " : (!atlas.state) -> !atlas.state" if '"atlas.' in line and " {atlas." in line else line
                             for line in body.splitlines())
        self.assertNotIn("atlas.virtual_", untagged)
        for contract in CONTRACTS:
            with self.subTest(contract=contract):
                self.rejected(f"module attributes {{{contract} = []}} {{\n" + untagged, None, UNMARKED)

    def test_marker_requires_every_contract_and_a_timing_state(self) -> None:
        for contract in CONTRACTS:
            with self.subTest(missing=contract):
                self.rejected(drop_attribute(self.timed, contract), drop_attribute(self.untimed, contract),
                              f"{INCOMPLETE} {contract}")
        untimed = drop_attribute(drop_attribute(self.timed, "atlas.timing_state"), "atlas.timing_provider")
        self.assertNotIn("atlas.timing_", untimed.split("\n", 1)[0])
        self.rejected(untimed, drop_attribute(self.untimed, "atlas.timing_state"), f"{INCOMPLETE} an explicit atlas.timing_state")

    def test_structured_handoff_reclassifies_before_finalization(self) -> None:
        structured = run("atlas-opt", self.timed, "--convert-atlas-to-llvm-calls")
        self.assertEqual(structured.returncode, 0, structured.stderr)
        for marker in MARKERS:
            with self.subTest(marker=marker):
                changed = remark(structured.stdout, marker)
                self.assertNotEqual(changed, structured.stdout)
                final = run("atlas-opt", changed, "--finalize-atlas-llvm-calls")
                self.assertNotEqual(final.returncode, 0, final.stdout)
                self.assertIn(UNSUPPORTED, final.stderr)

    def test_hand_written_stream_without_metadata_skips_generated_checks(self) -> None:
        source = "\n".join(['module {', '%s0 = "atlas.start"() : () -> !atlas.state',
                            '%s1 = "atlas.trap"(%s0) {kind = "ecall"} : (!atlas.state) -> !atlas.state', '}'])
        for tool, options in (("atlas-emit", ()), ("atlas-opt", ("--convert-atlas-to-llvm",)), ("atlas-opt", ("--insert-atlas-delays",))):
            with self.subTest(tool=tool, options=options):
                result = run(tool, source, *options)
                self.assertEqual(result.returncode, 0, result.stderr)
        result = run("atlas-opt", source, "--verify-atlas-generated-schedule")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected an Atlas virtual-to-machine artifact", result.stderr)


if __name__ == "__main__":
    unittest.main()
