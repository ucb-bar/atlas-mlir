#!/usr/bin/env python3
"""Synthetic SRAM-port projection checks, not measured hardware evidence."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parent.parent
SPEC = importlib.util.spec_from_file_location("sram", ROOT / "tools/check-atlascore-sram-observation.py")
SRAM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SRAM)
BASESPEC = importlib.util.spec_from_file_location("boundary_tests", SRAM.bootstrap(ROOT / "test/test_ee290_vls_observation.py"))
BASE = importlib.util.module_from_spec(BASESPEC)
BASESPEC.loader.exec_module(BASE)


def physical_fixture():
    samples = BASE.frames()
    boundary = SRAM.OBS.validate_edges(copy.deepcopy(samples), 1)
    selected = SRAM.projection(boundary)
    for sample in samples:
        sample["values"].update({key: 0 for key in selected if key not in sample["values"]})
    for command in boundary["panels"][0]["commands"]:
        store = command["op"] == 2
        for age in range(1, 35):
            values = samples[command["edge"] + age - 1]["values"]
            if (3 <= age <= 34) if store else (1 <= age <= 32):
                values["v0_en"] = 1
                values["v0_mode"] = int(store)
                values["v0_addr"] = command["line"] + age - (3 if store else 1)
                if store:
                    values["v0_write"], values["v0_mask"] = values["vw_data"], 0xffffffff
            if store and age <= 32:
                values["m4_ren"], values["m4_raddr"] = 1, age - 1
            if not store and age >= 3:
                values["m4_wen"], values["m4_waddr"], values["m4_write"] = 1, age - 3, values["mw_data"]
            if 2 <= age <= 33:
                values["m4_read" if store else "v0_read"] = values["mresp_data" if store else "vresp_data"]
    return samples, boundary


def trace(samples, selected):
    out = ["$timescale 1 ns $end", "$scope module TOP $end", "$scope module AtlasCore $end"]
    codes = {}
    for i, (key, (name, width)) in enumerate(selected.items()):
        codes[key] = "id" + str(i)
        parts = name.split(".")
        for part in parts[:-1]: out.append("$scope module " + part + " $end")
        out.append(f"$var wire {width} {codes[key]} {parts[-1]} $end")
        for _ in parts[:-1]: out.append("$upscope $end")
    out += ["$upscope $end", "$upscope $end", "$enddefinitions $end", "#0", "0" + codes["clock"]]
    for sample in samples:
        stamp = sample["time"]
        out.append("#" + str(stamp - 5))
        for key, value in sample["values"].items():
            if key != "clock": out.append("b" + ("x" if value is None else format(value, "b")) + " " + codes[key])
        out += ["#" + str(stamp), "1" + codes["clock"], "#" + str(stamp + 5), "0" + codes["clock"]]
    return "\n".join(out) + "\n"


class SRAMTests(unittest.TestCase):
    def test_selected_physical_streams(self):
        samples, boundary = physical_fixture()
        result = SRAM.validate_physical(samples, boundary)
        self.assertEqual([x["command_edge"] for x in result], [3, 38])
        self.assertEqual(result[1]["mreg_physical_bank"], 4)

    def test_private_projection_uses_same_pre_edge_sampler(self):
        samples, boundary = physical_fixture()
        count = len(SRAM.OBS.SIGNALS)
        decoded = list(SRAM.physical_edges(io.StringIO(trace(samples, SRAM.projection(boundary))), boundary))
        self.assertEqual(len(SRAM.validate_physical(decoded, boundary)), 2)
        self.assertEqual(len(SRAM.OBS.SIGNALS), count)

    def test_competition_address_mask_and_transport_mutations(self):
        for edge, key, value in [(10, "m7_wen", 1), (20, "v2_en", 1), (41, "v0_mask", 0),
                                 (41, "v0_addr", 100), (44, "m4_read", 99), (8, "m4_waddr", 31),
                                 (8, "m4_write", 99), (8, "m4_wen", None)]:
            with self.subTest(edge=edge, key=key):
                samples, boundary = physical_fixture()
                samples[edge - 1]["values"][key] = value
                with self.assertRaises(SRAM.OBS.ObservationError):
                    SRAM.validate_physical(samples, boundary)

    def test_unused_payload_unknown_allowed(self):
        samples, boundary = physical_fixture()
        samples[2]["values"]["v0_addr"] = None
        SRAM.validate_physical(samples, boundary)

    def test_wrong_compile_output_execution_binary_link_rejects_before_decoder(self):
        # Tiny receipts deliberately contain no executable or valid VCD. The
        # binary crosslink must reject before any decoder/status promotion.
        with tempfile.TemporaryDirectory(prefix="sram-link-test-", dir=SRAM.CHECK.allowed(ROOT / "build/rtl-timing")) as directory:
            base = Path(directory)
            def member(name, text):
                path = base / name
                path.write_text(text)
                return SRAM.CHECK.Checker().identity(path)
            compiler = member("compiler", "synthetic tool; never executed")
            compiled = member("compiled-model", "selected captured output")
            unrelated = member("unrelated-model", "different uncaptured executable")
            words = member("program.hex", "00000073\n")
            waveform = member("trace.vcd", "not parsed: crosslink must reject first")
            stdout = member("stdout.log", "not parsed: crosslink must reject first")
            compile_inputs = [{"role": "tool", "identity": compiler}]
            execution_inputs = [{"role": "tool", "identity": unrelated}, {"role": "program_words", "identity": words}]
            compilation = member("compile-phase.json", json.dumps({
                "schema": "atlas.ee290_captured_phase.v0", "kind": "atlascore_verilator_compile",
                "state": "phase_completed", "inputs_stable": True,
                "inputs_before": compile_inputs, "inputs_after": compile_inputs,
                "command": {"returncode": 0, "timed_out": False},
                "outputs": [{"files": [{"identity": compiled}]}]}))
            execution = member("execute-phase.json", json.dumps({
                "schema": "atlas.ee290_captured_phase.v0", "kind": "atlascore_vls_execute",
                "state": "phase_completed", "inputs_stable": True,
                "inputs_before": execution_inputs, "inputs_after": execution_inputs,
                "command": {"argv": [unrelated["path"], words["path"], waveform["path"], "200000"],
                            "returncode": 0, "timed_out": False, "stdout": stdout},
                "outputs": [{"files": [{"identity": waveform}]}]}))
            receipt = member("report.json", json.dumps({
                "schema": "atlas.selected_atlascore_replay.v0", "state": "numerical_and_boundary_replay_passed",
                "compile_phase": compilation, "execution_phase": execution, "program_words": words,
                "trace": waveform, "scope": "synthetic test fixture"}))
            with patch.object(SRAM.OBS, "validate_edges", side_effect=AssertionError("decoder reached")):
                with self.assertRaisesRegex(SRAM.OBS.ObservationError, "not a captured compilation output"):
                    SRAM.run(Path(receipt["path"]), receipt["sha256"], base / "output")
            self.assertFalse((base / "output").exists())


if __name__ == "__main__":
    unittest.main()
