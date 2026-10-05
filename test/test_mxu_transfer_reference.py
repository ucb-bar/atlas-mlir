"""Bounded selected-core checks for the ten MXU push/pop decoder modes."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile
import unittest

from test_vpu_relu_reference import _emitted, _object_words


ROOT = pathlib.Path(__file__).resolve().parents[1]
BIN = pathlib.Path(os.environ.get("ATLAS_OOT_BIN_DIR", ROOT / "build/bin"))
BASE = ROOT / "test/examples/vli_all_pair.mlir"
RTL_REVISION = "0079c0541111197741a231c002e3843fa6f545b2"
MODEL_SHA256 = "196380fca0cb3416a188538b342f8940802ebcf4d42e8d05dfe351b00b2c46fc"
STATE_SHA256 = "db2d8ae3c8a4ce6a417b0c691be1d446f1c0be95efe5607a251a6946a80de9e3"
FP8_TO_BF16 = {0x00: 0x0000, 0x30: 0x3F00, 0x38: 0x3F80,
               0x40: 0x4000, 0xB0: 0xBF00, 0xB8: 0xBF80, 0xC0: 0xC000}


def _all_code_fp8_to_bf16(code: int) -> int:
    """Independent bit-field oracle for the selected E4M3 movement policy."""
    sign, exp, mantissa = code >> 7, (code >> 3) & 0xF, code & 0x7
    if exp == 0 or (exp == 15 and mantissa == 7):
        return sign << 15
    return (sign << 15) | ((exp + 120) << 7) | (mantissa << 4)


def _base_ops() -> list[tuple[str, dict[str, str | int]]]:
    """Read the authored staging fixture; this parser is only a test builder."""
    result = []
    for line in BASE.read_text().splitlines():
        match = re.fullmatch(
            r'\s*%s\d+ = "atlas\.([a-z_]+)"\(%s\d+\) \{(.*)\} '
            r': \(!atlas\.state\) -> !atlas\.state', line)
        if not match:
            continue
        fields: dict[str, str | int] = {}
        for field in match[2].split(", "):
            key, value = field.split(" = ", 1)
            fields[key] = value[1:-1] if value.startswith('"') else int(value.split(" : ")[0])
        result.append((match[1], fields))
    if len(result) != 36:
        raise AssertionError("staging fixture changed")
    return result


def _render(ops: list[tuple[str, dict[str, str | int]]]) -> str:
    lines = ["module {", '  %s0 = "atlas.start"() : () -> !atlas.state']
    for index, (name, fields) in enumerate(ops, start=1):
        attrs = ", ".join(f'{key} = "{value}"' if isinstance(value, str)
                          else f"{key} = {value} : i32" for key, value in fields.items())
        lines.append(f'  %s{index} = "atlas.{name}"(%s{index - 1}) '
                     f'{{{attrs}}} : (!atlas.state) -> !atlas.state')
    return "\n".join((*lines, "}", ""))


def _program(kind: str, unit: int, slot: int) -> tuple[str, tuple[int, ...]]:
    base = _base_ops()
    if kind == "weight_fp8":
        ops = base[:10] + [
            ("mxu_push", {"kind": kind, "unit": unit, "src": 0, "slot": slot}),
            ("delay", {"cycles": 64}),
            # On this RTL ECALL immediately after DELAY bypasses the stall.
            ("alu_imm", {"kind": "addi", "dst": 9, "src": 0, "immediate": 11}),
            ("trap", {"kind": "ecall"}),
        ]
        return _render(ops), (10,)
    if kind == "acc_fp8":
        ops = base[:10] + [
            ("mxu_push", {"kind": kind, "unit": unit, "src": 0, "slot": slot}),
            ("delay", {"cycles": 64}),
            ("scalar_load", {"kind": "seli", "dst": 3, "base": 0, "offset": 127}),
            ("mxu_pop", {"format": "fp8", "unit": unit, "dst": 4,
                         "slot": slot, "scale_reg": 3}),
            ("delay", {"cycles": 64}),
        ] + base[19:26] + base[-3:]
        return _render(ops), (10, 13)
    if kind == "acc_bf16":
        ops = []
        loads = stores = 0
        for name, original in base:
            fields = dict(original)
            if name == "vload":
                fields["dst"] = 4 + loads
                loads += 1
            elif name == "vstore":
                fields["src"] = 6 + stores
                stores += 1
            elif name == "vli":
                ops.extend([
                    ("mxu_push", {"kind": kind, "unit": unit, "src": 4, "slot": slot}),
                    ("delay", {"cycles": 64}),
                    ("mxu_pop", {"format": "bf16", "unit": unit, "dst": 6,
                                 "slot": slot, "scale_reg": 0}),
                ])
                continue
            ops.append((name, fields))
        if (loads, stores) != (2, 2):
            raise AssertionError("staging fixture transfer count changed")
        return _render(ops), (17, 19)
    raise ValueError(kind)


def _payload(kind: str, phase: int) -> tuple[bytes, bytes]:
    if kind == "acc_bf16":
        codes = (0x0000, 0x3F00, 0x3F80, 0x4000,
                 0xBF00, 0xBF80, 0x8000, 0x4040)
        values = [codes[(i * 7 + i // 13 + phase) % len(codes)] for i in range(1024)]
        data = b"".join(struct.pack("<H", code) for code in values)
        physical = b"".join(data[32 * row:32 * row + 32] +
                            data[1024 + 32 * row:1024 + 32 * row + 32]
                            for row in range(32))
        return data, physical
    if kind == "acc_fp8_all":
        data = bytes((i + 17 * phase) & 0xFF for i in range(1024))
        physical = b"".join(struct.pack("<H", _all_code_fp8_to_bf16(code))
                            for code in data)
        return data, physical
    if kind == "acc_fp8":
        codes = tuple(FP8_TO_BF16)
        data = bytes(codes[(i * 5 + i // 17 + phase) % len(codes)] for i in range(1024))
        physical = b"".join(struct.pack("<H", FP8_TO_BF16[code]) for code in data)
        return data, physical
    data = bytes((17 * i + 31 * phase) & 0xFF for i in range(1024))
    return data, data


def _assembler():
    root = os.environ.get("ATLAS_ASSEMBLER_ROOT")
    if not root:
        raise unittest.SkipTest("set ATLAS_ASSEMBLER_ROOT")
    spec = importlib.util.spec_from_file_location("selected_mxu_transfer_assembler",
                                                   pathlib.Path(root) / "assembler.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MxuTransferReferenceTest(unittest.TestCase):
    def test_exhaustive_fp8_dequantization_oracle_anchors(self) -> None:
        for code, expected in {**FP8_TO_BF16, 0x01: 0, 0x81: 0x8000,
                               0x7F: 0, 0xFF: 0x8000,
                               0x7E: 0x43E0, 0xFE: 0xC3E0}.items():
            with self.subTest(code=code):
                self.assertEqual(_all_code_fp8_to_bf16(code), expected)

    def test_selected_words_and_llvm_object_for_both_units_and_slots(self) -> None:
        assembler = _assembler()
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "transfer.mlir"
            for kind in ("weight_fp8", "acc_fp8", "acc_bf16"):
                for unit in (0, 1):
                    for slot in (0, 1):
                        with self.subTest(kind=kind, unit=unit, slot=slot):
                            source, indices = _program(kind, unit, slot)
                            path.write_text(source)
                            words = _emitted(path)
                            self.assertEqual(words, _object_words(path))
                            if kind == "weight_fp8":
                                expected = getattr(assembler, f"VMATPUSH_W_MXU{unit}")(slot, 0)
                                self.assertEqual(words[indices[0]], expected)
                            elif kind == "acc_fp8":
                                push = getattr(assembler, f"VMATPUSH_AFP8_MXU{unit}")(slot, 0)
                                pop = getattr(assembler, f"VMATPOP_FP8_MXU{unit}")(4, 3, slot)
                                self.assertEqual(tuple(words[i] for i in indices), (push, pop))
                            else:
                                push = getattr(assembler, f"VMATPUSH_ABF16_MXU{unit}")(slot, 4)
                                pop = getattr(assembler, f"VMATPOP_BF16_MXU{unit}")(6, slot)
                                self.assertEqual(tuple(words[i] for i in indices), (push, pop))
                            printed = subprocess.check_output([str(BIN / "atlas-opt"), str(path)],
                                                              text=True)
                            reparsed = subprocess.check_output([str(BIN / "atlas-opt"), "-"],
                                                               input=printed, text=True)
                            self.assertEqual(reparsed, printed)

    def test_selected_core_transfer_and_physical_storage(self) -> None:
        keys = ("ATLAS_ARC_MODEL", "ATLAS_ARC_STATE", "ATLAS_MODELIR_ROOT", "ATLAS_RTL_ROOT")
        if not all(os.environ.get(key) for key in keys):
            self.skipTest("set selected ARC, state, ModeLIR and RTL paths")
        model, state, modelir, rtl = (pathlib.Path(os.environ[key]).resolve(strict=True)
                                     for key in keys)
        self.assertEqual(subprocess.check_output(["git", "-C", str(rtl), "rev-parse", "HEAD"],
                                                 text=True).strip(), RTL_REVISION)
        for artifact, expected in ((model, MODEL_SHA256), (state, STATE_SHA256)):
            with artifact.open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), expected)
        previous_cwd = pathlib.Path.cwd()
        sys.path.insert(0, str(modelir))
        try:
            os.chdir(modelir)
            from mlc.backends import cosim_atlas

            original = cosim_atlas.CosimCore

            class SelectedCore(original):
                def peek(self, name: str) -> int:
                    return super().peek("scalar/halt_now" if name == "io_halted" else name)

                def poke(self, name: str, value: int) -> None:
                    if name in ("io_dmaTL_d_bits_opcode", "io_dmaTL_d_bits_size"):
                        if name in self._S:
                            raise AssertionError(f"unexpected live ARC input {name}")
                        return
                    super().poke(name, value)

            cosim_atlas.CosimCore = SelectedCore
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    path = pathlib.Path(temporary) / "transfer.mlir"
                    for kind in ("weight_fp8", "acc_fp8", "acc_fp8_all", "acc_bf16"):
                        for unit in (0, 1):
                            for slot in (0, 1):
                                with self.subTest(kind=kind, unit=unit, slot=slot):
                                    selected_kind = "acc_fp8" if kind == "acc_fp8_all" else kind
                                    source, _ = _program(selected_kind, unit, slot)
                                    path.write_text(source)
                                    words = _emitted(path)
                                    payload, physical = _payload(kind, unit + 2 * slot)
                                    observed: dict[str, bytes | int] = {}
                                    seen_running = False

                                    def on_cycle(core: SelectedCore) -> None:
                                        nonlocal seen_running
                                        halted = core.peek("scalar/halt_now") == 1
                                        if not halted:
                                            seen_running = True
                                        if not seen_running or not halted or observed:
                                            return
                                        observed["x9"] = core.peek("scalar/regfile/regs_9")
                                        if kind == "weight_fp8":
                                            observed["storage"] = bytes(
                                                core.peek(f"mxu{unit}/wbuf/weightSlot{slot}_{lane}_{col}")
                                                for lane in range(32) for col in range(32))
                                            observed["other"] = bytes(
                                                core.peek(f"mxu{unit}/wbuf/weightSlot{1 - slot}_{lane}_{col}")
                                                for lane in range(32) for col in range(32))
                                        else:
                                            observed["storage"] = b"".join(
                                                core.read_mem_row(f"mxu{unit}/accBuf/buffer{slot}_ext", row, 64)
                                                for row in range(32))

                                    preload = [(0x90000000, payload[:1024]),
                                               (0x90000400, payload[1024:]),
                                               (0x90000800, b"\xA5" * 1024),
                                               (0x90000C00, b"\xA5" * 1024),
                                               (0x90001000, b"\x5A" * 32)]
                                    result = cosim_atlas.run_program(
                                        model, state, words, preload=preload,
                                        max_cycles=8000, on_cycle=on_cycle)
                                    self.assertTrue(result.halted)
                                    self.assertEqual(result.halt_reason, 2)
                                    expected_beats = 64 if kind == "acc_bf16" else 32
                                    expected_writes = 0 if kind == "weight_fp8" else expected_beats
                                    self.assertEqual((result.reads, result.writes),
                                                     (expected_beats, expected_writes),
                                                     f"cycles={result.cycles} x9={observed.get('x9')}")
                                    self.assertTrue(observed["storage"] == physical,
                                                    f"cycles={result.cycles} x9={observed.get('x9')} "
                                                    f"actual={observed['storage'][:32].hex()} "
                                                    f"expected={physical[:32].hex()}")
                                    self.assertEqual(result.slave.captured(0x90000000, len(payload)), payload)
                                    self.assertEqual(result.slave.captured(0x90001000, 32), b"\x5A" * 32)
                                    if kind == "weight_fp8":
                                        self.assertEqual(observed["other"], bytes(1024))
                                        self.assertEqual(observed["x9"], 11)
                                    elif kind != "acc_fp8_all":
                                        self.assertEqual(result.slave.captured(0x90000800, len(payload)),
                                                         payload)
            finally:
                cosim_atlas.CosimCore = original
        finally:
            os.chdir(previous_cwd)
            sys.path.remove(str(modelir))


if __name__ == "__main__":
    unittest.main()
