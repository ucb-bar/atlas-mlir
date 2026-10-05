"""Bounded selected-core checks for all conditional scalar branch modes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

from test_scalar_alu_reference import MASK, _emitted, _object_words

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
PANELS = {
    "bne": ((-1, 1), (1, 1), (1, -1)),
    "bge": ((1, -1), (-1, 1), (1, 1)),
    "bltu": ((1, -1), (-1, 1), (1, 1)),
    "bgeu": ((-1, 1), (1, -1), (1, 1)),
}


def _signed(value: int) -> int:
    bits = value & MASK
    return bits - (1 << 32) if bits & (1 << 31) else bits


def _taken(kind: str, lhs: int, rhs: int) -> bool:
    if kind == "bne":
        return (lhs & MASK) != (rhs & MASK)
    if kind == "bge":
        return _signed(lhs) >= _signed(rhs)
    if kind == "bltu":
        return (lhs & MASK) < (rhs & MASK)
    if kind == "bgeu":
        return (lhs & MASK) >= (rhs & MASK)
    raise AssertionError(f"unexpected branch mode {kind}")


def _program(kind: str, lhs: int, rhs: int, *, offset_bytes: int = 6) -> tuple[str, str]:
    source = "\n".join((
        "module {",
        '  %s0 = "atlas.start"() : () -> !atlas.state',
        f'  %s1 = "atlas.alu_imm"(%s0) {{kind = "addi", dst = 10 : i32, src = 0 : i32, '
        f'immediate = {lhs} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s2 = "atlas.alu_imm"(%s1) {{kind = "addi", dst = 11 : i32, src = 0 : i32, '
        f'immediate = {rhs} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s3 = "atlas.branch"(%s2) {{kind = "{kind}", lhs = 10 : i32, rhs = 11 : i32, '
        f'offset_bytes = {offset_bytes} : i32}} : (!atlas.state) -> !atlas.state',
        '  %s4 = "atlas.alu_imm"(%s3) {kind = "addi", dst = 12 : i32, src = 0 : i32, '
        'immediate = 11 : i32} : (!atlas.state) -> !atlas.state',
        '  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 13 : i32, src = 0 : i32, '
        'immediate = 22 : i32} : (!atlas.state) -> !atlas.state',
        '  %s6 = "atlas.alu_imm"(%s5) {kind = "addi", dst = 14 : i32, src = 0 : i32, '
        'immediate = 33 : i32} : (!atlas.state) -> !atlas.state',
        '  %s7 = "atlas.trap"(%s6) {kind = "ecall"} : (!atlas.state) -> !atlas.state',
        "}", "",
    ))
    # The selected PC is a word index, but the B-immediate records bytes.
    # The assembler doubles this label displacement to offset_bytes = 6.
    transcription = "\n".join((
        f"ADDI x10, x0, {lhs}", f"ADDI x11, x0, {rhs}",
        f"{kind.upper()} x10, x11, target",
        "ADDI x12, x0, 11", "ADDI x13, x0, 22",
        "target:", "ADDI x14, x0, 33", "ECALL", "",
    ))
    return source, transcription


class BranchModeReferenceTest(unittest.TestCase):
    def test_invalid_offsets_are_rejected_for_each_mode(self) -> None:
        for kind in PANELS:
            for offset, diagnostic in ((5, "even"), (4096, "branch byte offset")):
                with self.subTest(kind=kind, offset=offset):
                    source = _program(kind, 1, -1, offset_bytes=offset)[0]
                    result = subprocess.run(
                        [str(BIN / "atlas-opt"), "-"], input=source,
                        text=True, capture_output=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)

    def test_typed_assembler_and_llvm_words_agree(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for the selected assembler")
        spec = importlib.util.spec_from_file_location(
            "atlas_selected_branch_assembler", pathlib.Path(assembler_root) / "assembler.py"
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "branch.mlir"
            for kind, panels in PANELS.items():
                for lhs, rhs in panels:
                    with self.subTest(kind=kind, lhs=lhs, rhs=rhs):
                        source, transcription = _program(kind, lhs, rhs)
                        path.write_text(source)
                        words = _emitted(path)
                        self.assertEqual(words, tuple(assembler.assemble(transcription)))
                        self.assertEqual(words, _object_words(path, len(words), directory))

    def test_selected_core_matches_signed_unsigned_branch_semantics(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        receipt_name = os.environ.get("ATLAS_BRANCH_RECEIPT")
        receipt = pathlib.Path(receipt_name) if receipt_name else None
        if receipt is not None and receipt.exists():
            raise AssertionError("branch receipt destination must be fresh")
        self.assertEqual(subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
                         "0079c0541111197741a231c002e3843fa6f545b2")
        manifest = state.read_text()
        for register in (10, 11, 12, 13, 14):
            self.assertIn(f"scalar/regfile/regs_{register}", manifest)
        old_cwd = pathlib.Path.cwd()
        cases: list[dict[str, object]] = []
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

            cosim_atlas.CosimCore = SelectedCore
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    path = pathlib.Path(temporary) / "branch.mlir"
                    for kind, panels in PANELS.items():
                        for lhs, rhs in panels:
                            with self.subTest(kind=kind, lhs=lhs, rhs=rhs):
                                path.write_text(_program(kind, lhs, rhs)[0])
                                observed: dict[str, int] = {}

                                def on_cycle(core: SelectedCore) -> None:
                                    for register in (12, 13, 14):
                                        observed[f"x{register}"] = core.peek(f"scalar/regfile/regs_{register}")

                                result = cosim_atlas.run_program(
                                    model, state, _emitted(path), max_cycles=80, on_cycle=on_cycle,
                                )
                                self.assertTrue(result.halted)
                                self.assertEqual(observed["x12"] & MASK, 11)
                                self.assertEqual(observed["x13"] & MASK, 0 if _taken(kind, lhs, rhs) else 22)
                                self.assertEqual(observed["x14"] & MASK, 33)
                                cases.append({
                                    "kind": kind, "lhs": lhs, "rhs": rhs,
                                    "taken": _taken(kind, lhs, rhs),
                                    "cycles": result.cycles,
                                    "x12": observed["x12"] & MASK,
                                    "x13": observed["x13"] & MASK,
                                    "x14": observed["x14"] & MASK,
                                })
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        if receipt is not None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "schema": "atlas.branch_mode_bounded_observation.v1",
                "scope": "selected standalone AtlasCore, fixed 6-byte offset and directed operands",
                "rtl_revision": "0079c0541111197741a231c002e3843fa6f545b2",
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "test_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "emitter_sha256": hashlib.sha256((BIN / "atlas-emit").read_bytes()).hexdigest(),
                "cases": cases,
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
