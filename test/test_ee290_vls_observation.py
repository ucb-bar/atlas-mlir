#!/usr/bin/env python3
"""Synthetic rejection checks; these tests are not integrated timing evidence."""

import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parent.parent
SPEC = importlib.util.spec_from_file_location("observation", ROOT / "tools/observe-ee290-vls.py")
OBS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OBS)


def frames():
    result = [{"edge": edge, "time": edge * 20 + 10,
               "values": {key: 0 for key in OBS.SIGNALS}} for edge in range(1, 82)]
    result[0]["values"]["reset"] = 1
    for command, op, line in [(3, 1, 128), (38, 2, 384)]:
        values = result[command - 1]["values"]
        values.update(fire=1, launch=1, cmd=1, op=op, line=line, mreg=4)
        for age in range(1, 35):
            values = result[command + age - 1]["values"]
            values["load_busy" if op == 1 else "store_busy"] = 1
            source, response, dest = ("vr", "vresp", "mw") if op == 1 else ("mr", "mresp", "vw")
            if age <= 32:
                values[source] = 1
                if op == 1:
                    values.update(vr_bank=0, vr_addr=line + age - 1)
                else:
                    values.update(mr_id=4, mr_row=age - 1)
            if 2 <= age <= 33:
                values[response] = 1
                values[response + "_data"] = (1 << 255) + age - 2
            if 3 <= age <= 34:
                values[dest] = 1
                values[dest + "_data"] = (1 << 255) + age - 3
                if op == 1:
                    values.update(mw_id=4, mw_row=age - 3)
                else:
                    values.update(vw_bank=0, vw_addr=line + age - 3)
    result[73]["values"].update(fire=1, csr=1, csr_addr=0xC10, csr_op=1, csr_data=1)
    for sample in result[74:]:
        sample["values"]["marker"] = 1
    result[75]["values"]["ecall"] = 1
    for sample in result[75:]:
        sample["values"]["halt"] = 1
    return result


def vcd(samples, at_edge_mutation=False):
    out = ["$timescale 1 ns $end", "$scope module TestDriver $end", "$scope module atlas $end"]
    keys = list(OBS.SIGNALS)
    codes = {key: "id" + str(i) for i, key in enumerate(keys)}
    for instance in OBS.MODULES:
        out.append("$scope module " + instance + " $end")
        for key, (signal, width) in OBS.SIGNALS.items():
            inst, name = signal.split(".")
            if inst == instance:
                out.append(f"$var wire {width} {codes[key]} {name} $end")
        out.append("$upscope $end")
    out.extend(["$upscope $end", "$upscope $end", "$enddefinitions $end", "#0", "0" + codes["clock"]])
    for sample in samples:
        timestamp = sample["time"]
        out.append("#" + str(timestamp - 5))
        for key, value in sample["values"].items():
            if key != "clock":
                bits = "x" if value is None else format(value, "b")
                out.append("b" + bits + " " + codes[key])
        out.extend(["#" + str(timestamp), "1" + codes["clock"]])
        if at_edge_mutation:
            # Settled post-edge command inputs differ. Ordering within this
            # timestamp cannot change which pre-edge values are consumed.
            out.append("b0 " + codes["cmd"])
        out.extend(["#" + str(timestamp + 5), "0" + codes["clock"]])
    return "\n".join(out) + "\n"


class ObservationTests(unittest.TestCase):
    def reject(self, samples, text=None):
        context = self.assertRaisesRegex(OBS.ObservationError, text) if text else self.assertRaises(OBS.ObservationError)
        with context:
            OBS.validate_edges(samples, 1)

    def test_complete_rows_and_endpoints(self):
        report = OBS.validate_edges(frames(), 1)
        commands = report["panels"][0]["commands"]
        self.assertEqual([x["release_edge"] for x in commands], [38, 73])
        self.assertEqual(commands[0]["source_edges"], list(range(4, 36)))
        self.assertEqual(commands[1]["destination_edges"], list(range(41, 73)))
        self.assertEqual(report["panels"][0]["halt_edge"], 76)

    def test_pre_edge_sampling_ignores_post_edge_change(self):
        samples = list(OBS.vcd_edges(io.StringIO(vcd(frames(), True)), "TestDriver.atlas"))
        self.assertEqual(samples[2]["values"]["cmd"], 1)
        self.assertEqual(OBS.validate_edges(samples, 1)["sampled_edges"], 81)

    def test_missing_response_wrong_row_data_and_busy(self):
        for edge, key, value in [(5, "vresp", 0), (10, "mw_row", 21), (10, "mw_data", 77),
                                 (72, "store_busy", 0), (73, "store_busy", 1)]:
            with self.subTest(key=key, edge=edge):
                samples = frames()
                samples[edge - 1]["values"][key] = value
                self.reject(samples)

    def test_coherent_store_transport_cannot_hide_changed_mreg_data(self):
        samples = frames()
        samples[44]["values"]["mresp_data"] = 999
        samples[45]["values"]["vw_data"] = 999
        self.reject(samples, "differs from the loaded row")

    def test_issue_acceptance_and_address_rejections(self):
        for key, value in [("launch", 0), ("fire", 0), ("load_state", 1),
                           ("line", 129), ("line", 49152), ("op", 0), ("mreg", 64)]:
            with self.subTest(key=key, value=value):
                samples = frames()
                samples[2]["values"][key] = value
                self.reject(samples)

    def test_early_marker_halt_and_missing_tail_reject(self):
        samples = frames()
        samples[68]["values"].update(fire=1, csr=1, csr_addr=0xC10, csr_op=1, csr_data=1)
        self.reject(samples, "marker")
        samples = frames()
        samples[74]["values"]["marker"] = 0
        self.reject(samples, "publication")
        samples = frames()
        samples[70]["values"]["halt"] = 1
        self.reject(samples, "halt")
        self.reject(frames()[:72], "complete")
        self.reject(frames()[:75], "complete")

    def test_unknown_required_value_and_inactive_payload(self):
        samples = frames()
        samples[8]["values"]["vresp_data"] = None
        self.reject(samples, "unknown")

    def test_initial_host_loading_and_between_panel_quiescence(self):
        base = frames()
        idle = copy.deepcopy(base[1])
        idle["values"]["halt"] = 1
        samples = [copy.deepcopy(base[0])] + [copy.deepcopy(idle) for _ in range(7)] + copy.deepcopy(base[1:])
        # The host reads status/data and clears DBG0 while the completed core
        # remains halted, then starts the next panel without resetting engines.
        readback = copy.deepcopy(idle)
        readback["values"]["marker"] = 1
        for _ in range(2):
            samples += [copy.deepcopy(readback) for _ in range(4)]
            samples += [copy.deepcopy(idle) for _ in range(4)]
            samples += copy.deepcopy(base[1:])
        for i, sample in enumerate(samples, 1):
            sample.update(edge=i, time=i * 20 + 10)
        result = OBS.validate_edges(samples, 3)
        self.assertEqual(len(result["panels"]), 3)
        self.assertEqual(sum(x["kind"] == "halt_deasserted" for x in result["endpoints"]), 3)
        samples[3]["values"]["fire"] = 1
        self.reject(samples, "halt")
        samples = frames()
        samples[1]["values"]["vresp_data"] = None
        OBS.validate_edges(samples, 1)
        samples[1]["values"]["cmd"] = None
        self.reject(samples, "unknown")

    def test_vcd_missing_width_suppression_and_clock_ambiguity(self):
        trace = vcd(frames())
        broken = [trace.replace("$var wire 6 id6 io_cmd_bits_mregBank $end", ""),
                  trace.replace("$var wire 6 id6", "$var wire 7 id6"),
                  trace + "$dumpoff $end\n", trace.replace("#30\n1id0", "#30\n1id0\n0id0")]
        for text in broken:
            with self.subTest(text=text[-90:]), self.assertRaises(OBS.ObservationError):
                list(OBS.vcd_edges(io.StringIO(text), "TestDriver.atlas"))

    def test_vcd_unknown_payload_and_limits(self):
        samples = frames()
        samples[8]["values"]["vresp_data"] = None
        decoded = list(OBS.vcd_edges(io.StringIO(vcd(samples)), "TestDriver.atlas"))
        self.reject(decoded, "unknown")
        with self.assertRaisesRegex(OBS.ObservationError, "limit"):
            list(OBS.vcd_edges(io.StringIO(vcd(frames())), "TestDriver.atlas", 4))

    def test_illegal_and_other_csr_reject(self):
        samples = frames()
        samples[75]["values"]["illegal"] = 1
        self.reject(samples, "illegal")
        samples = frames()
        samples[73]["values"]["csr_addr"] = 1
        self.reject(samples, "CSR")
        samples = frames()
        samples[75]["values"]["fire"] = 1
        self.reject(samples, "ECALL")

    def test_path_restriction_precedes_target_access(self):
        # Never create/stat a prohibited directory; lexical rejection must
        # happen before the mocked target access can occur.
        with patch.object(Path, "lstat", side_effect=AssertionError("unexpected access")):
            with self.assertRaises(OBS.ObservationError):
                OBS.CHECK.allowed("/tmp/forbidden-vlsi-target/not-created")

    def test_cli_artifact_selection_and_no_qualification(self):
        scratch = OBS.CHECK.allowed(ROOT / "build/rtl-timing")
        with tempfile.TemporaryDirectory(prefix="observation-test-", dir=scratch) as directory:
            base = Path(directory)
            checker = OBS.CHECK.Checker()
            def identity(name, content):
                path = base / name
                path.write_text(content)
                return checker.identity(path)
            sources = []
            for instance, module in OBS.MODULES.items():
                names = [signal.split(".")[1] for signal, _ in OBS.SIGNALS.values() if signal.startswith(instance + ".")]
                sources.append(identity(module + ".sv", "module " + module + ";\nwire " + ",".join(names) + ";\nendmodule\n"))
            hardware = identity("hw.mlir", "hw")
            manifest = identity("manifest.json", json.dumps({"schema": "atlas.retained_hw_ir.v0", "state": "verified", "hardware_ir": hardware}))
            report = {"schema": "atlas.ee290_vls_witness.v0", "state": "integrated_execution_passed",
                      "integrated_execution_passed": True, "provenance": {
                          "manifest": manifest, "hardware_ir": hardware,
                          "simulator": identity("simulator", "executable"), "program": identity("program.mlir", "program"),
                          "simulator_sources": sources}}
            witness = identity("witness.json", json.dumps(report))
            plan_dir, result_dir = base / "plan", base / "result"
            plan = OBS.prepare(witness["path"], witness["sha256"], "TestDriver.atlas", plan_dir)
            self.assertFalse(plan["capture_command_validated"])
            trace = identity("trace.vcd", vcd(frames()))
            plan_id = checker.identity(plan_dir / "plan.json")
            result = OBS.decode(plan_id["path"], plan_id["sha256"], trace["path"], trace["sha256"], result_dir, 1)
            self.assertEqual(result["state"], "bounded_boundary_events_passed")
            self.assertFalse(result["scheduling_qualified"])
            self.assertFalse(result["capture_execution_link_verified"])
            with self.assertRaisesRegex(OBS.ObservationError, "digest"):
                OBS.decode(plan_id["path"], "0" * 64, trace["path"], trace["sha256"], base / "bad", 1)
            with self.assertRaisesRegex(OBS.ObservationError, "fresh"):
                OBS.decode(plan_id["path"], plan_id["sha256"], trace["path"], trace["sha256"], result_dir, 1)
            report["provenance"]["hardware_ir"] = identity("different.hw.mlir", "different hw")
            bad_witness = identity("mismatched-witness.json", json.dumps(report))
            with self.assertRaisesRegex(OBS.ObservationError, "hardware identity"):
                OBS.prepare(bad_witness["path"], bad_witness["sha256"], "TestDriver.atlas", base / "bad-plan")
            plan["signals"]["cmd"]["path"] = "TestDriver.atlas.wrong"
            changed = identity("changed-plan.json", json.dumps(plan))
            with self.assertRaisesRegex(OBS.ObservationError, "projection"):
                OBS.decode(changed["path"], changed["sha256"], trace["path"], trace["sha256"], base / "bad-projection", 1)


if __name__ == "__main__":
    unittest.main()
