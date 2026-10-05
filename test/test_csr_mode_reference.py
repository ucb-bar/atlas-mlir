"""Bounded selected-core read/modify/write checks for all six CSR modes."""

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
MODES = ("rrw", "rrs", "rrc", "rrwi", "rrsi", "rrci")
PANELS = ((10, 12), (29, 6))
CSR_DBG0 = 0xC10


def _new_value(kind: str, previous: int, mask: int) -> int:
    if kind in ("rrw", "rrwi"):
        return mask
    if kind in ("rrs", "rrsi"):
        return previous | mask
    if kind in ("rrc", "rrci"):
        return previous & ~mask & MASK
    raise AssertionError(f"unknown CSR mode {kind}")


def _program(kind: str, previous: int, mask: int, *, address: int = CSR_DBG0,
             source: int | None = None) -> tuple[str, str]:
    operand = 11 if kind in ("rrw", "rrs", "rrc") else mask
    if source is not None:
        operand = source
    mlir = "\n".join((
        "module {",
        '  %s0 = "atlas.start"() : () -> !atlas.state',
        f'  %s1 = "atlas.alu_imm"(%s0) {{kind = "addi", dst = 10 : i32, src = 0 : i32, '
        f'immediate = {previous} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s2 = "atlas.alu_imm"(%s1) {{kind = "addi", dst = 11 : i32, src = 0 : i32, '
        f'immediate = {mask} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s3 = "atlas.csr"(%s2) {{kind = "rrw", dst = 0 : i32, source = 10 : i32, '
        f'address = {CSR_DBG0} : i32}} : (!atlas.state) -> !atlas.state',
        f'  %s4 = "atlas.csr"(%s3) {{kind = "{kind}", dst = 12 : i32, '
        f'source = {operand} : i32, address = {address} : i32}} : (!atlas.state) -> !atlas.state',
        '  %s5 = "atlas.alu_imm"(%s4) {kind = "addi", dst = 0 : i32, src = 0 : i32, '
        'immediate = 0 : i32} : (!atlas.state) -> !atlas.state',
        '  %s6 = "atlas.trap"(%s5) {kind = "ecall"} : (!atlas.state) -> !atlas.state',
        "}", "",
    ))
    mnemonic = "CS" + kind.upper()
    assembly = "\n".join((
        f"ADDI x10, x0, {previous}", f"ADDI x11, x0, {mask}",
        f"CSRRW x0, 0x{CSR_DBG0:x}, x10",
        (f"{mnemonic} x12, 0x{address:x}, x11" if kind in ("rrw", "rrs", "rrc")
         else f"{mnemonic} x12, 0x{address:x}, {operand}"),
        "ADDI x0, x0, 0", "ECALL", "",
    ))
    return mlir, assembly


class CSRModeReferenceTest(unittest.TestCase):
    def test_invalid_address_and_source_are_rejected(self) -> None:
        for kind in MODES:
            for changed, diagnostic in (
                ({"address": 0xC12}, "not implemented"),
                ({"source": 32}, "source/zimm"),
            ):
                with self.subTest(kind=kind, changed=changed):
                    source = _program(kind, 10, 12, **changed)[0]
                    result = subprocess.run(
                        [str(BIN / "atlas-opt"), "-"], input=source,
                        text=True, capture_output=True, check=False,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stderr)
        for kind in ("rrs", "rrc", "rrsi", "rrci"):
            with self.subTest(kind=kind, address="read-only"):
                source = _program(kind, 10, 12, address=0xC02)[0]
                result = subprocess.run(
                    [str(BIN / "atlas-opt"), "-"], input=source,
                    text=True, capture_output=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("read-only", result.stderr)

    def test_typed_assembler_and_llvm_words_agree(self) -> None:
        assembler_root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
        if not assembler_root:
            self.skipTest("set ATLAS_ASSEMBLER_ROOT for the selected assembler")
        spec = importlib.util.spec_from_file_location(
            "atlas_selected_csr_assembler", pathlib.Path(assembler_root) / "assembler.py",
        )
        assert spec is not None and spec.loader is not None
        assembler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(assembler)
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary)
            path = directory / "csr.mlir"
            for kind in MODES:
                for previous, mask in PANELS:
                    with self.subTest(kind=kind, previous=previous, mask=mask):
                        source, assembly = _program(kind, previous, mask)
                        path.write_text(source)
                        words = _emitted(path)
                        self.assertEqual(words, tuple(assembler.assemble(assembly)))
                        self.assertEqual(words, _object_words(path, len(words), directory))
                        if kind == "rrci":
                            self.assertEqual((words[3] >> 12) & 7, 7)

    def test_selected_core_matches_independent_read_modify_write(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected-source-linked ARC model, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True) for key in keys)
        self.assertEqual(subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"], text=True).strip(),
                         "0079c0541111197741a231c002e3843fa6f545b2")
        manifest = state.read_text()
        for signal in ("scalar/regfile/regs_12", "csrfile/reg_dbg0", "csrfile/reg_dbg1"):
            self.assertIn(signal, manifest)
        receipt_name = os.environ.get("ATLAS_CSR_RECEIPT")
        receipt = pathlib.Path(receipt_name) if receipt_name else None
        if receipt is not None and receipt.exists():
            raise AssertionError("CSR receipt destination must be fresh")
        cases: list[dict[str, object]] = []
        old_cwd = pathlib.Path.cwd()
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
                    path = pathlib.Path(temporary) / "csr.mlir"
                    for kind in MODES:
                        for previous, mask in PANELS:
                            with self.subTest(kind=kind, previous=previous, mask=mask):
                                path.write_text(_program(kind, previous, mask)[0])
                                observed: dict[str, int] = {}

                                def on_cycle(core: SelectedCore) -> None:
                                    observed["x12"] = core.peek("scalar/regfile/regs_12")
                                    observed["dbg0"] = core.peek("csrfile/reg_dbg0")
                                    observed["dbg1"] = core.peek("csrfile/reg_dbg1")

                                result = cosim_atlas.run_program(
                                    model, state, _emitted(path), max_cycles=80, on_cycle=on_cycle,
                                )
                                self.assertTrue(result.halted)
                                self.assertEqual(observed["x12"] & MASK, previous)
                                self.assertEqual(observed["dbg0"] & MASK, _new_value(kind, previous, mask))
                                self.assertEqual(observed["dbg1"] & MASK, 0)
                                cases.append({
                                    "kind": kind, "previous": previous, "mask": mask,
                                    "cycles": result.cycles, "old_value": observed["x12"] & MASK,
                                    "new_value": observed["dbg0"] & MASK,
                                    "untouched_dbg1": observed["dbg1"] & MASK,
                                })
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(old_cwd)
            sys.path.remove(str(modelir))
        if receipt is not None:
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps({
                "schema": "atlas.csr_mode_bounded_observation.v1",
                "scope": "selected standalone AtlasCore, writable debug CSR and directed masks",
                "rtl_revision": "0079c0541111197741a231c002e3843fa6f545b2",
                "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "test_source_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                "emitter_sha256": hashlib.sha256((BIN / "atlas-emit").read_bytes()).hexdigest(),
                "cases": cases,
            }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
