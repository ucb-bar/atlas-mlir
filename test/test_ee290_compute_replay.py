"""Compute replay checks: synthetic VCD/checker tests always run; the Verilator run is env-gated.

The real run needs ATLAS_EE290_COMPUTE_MODEL (a built VAtlasCore), ATLAS_OOT_BIN_DIR (atlas-opt, atlas-emit)
and ATLAS_OP_TIMING (merlin.op_timing.v1 facts). It replays vmul and dma_xlu with both consumers, then
mutates the captured traces and exports to confirm each check bites.
"""
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay", ROOT / "tools/replay-ee290-compute.py")
REPLAY = importlib.util.module_from_spec(spec); spec.loader.exec_module(REPLAY)
BIND = REPLAY.BIND
MODEL, BIN, FACTS = (os.environ.get(k) for k in ("ATLAS_EE290_COMPUTE_MODEL", "ATLAS_OOT_BIN_DIR", "ATLAS_OP_TIMING"))


def vcd(cycles, drop=None):
    """VCD with every selected signal under TOP.AtlasCore; the clock rises at 10 + 20n and cycles[n] holds over [20n, 20n+20)."""
    ids = {key: f"s{n}" for n, key in enumerate(REPLAY.SIGNALS)}
    lines = ["$timescale 1ps $end", "$scope module TOP $end", "$scope module AtlasCore $end"]
    for key, (path, width) in REPLAY.SIGNALS.items():
        if key == drop: continue
        *scopes, leaf = path.split(".")
        lines += [f"$scope module {s} $end" for s in scopes] + [f"$var wire {width} {ids[key]} {leaf} $end"] + ["$upscope $end"] * len(scopes)
    lines += ["$upscope $end"] * 2 + ["$enddefinitions $end"]
    for n, values in enumerate(cycles):
        for edge in (0, 1):
            lines.append(f"#{20 * n + 10 * edge}")
            for key, value in {**values, "clock": edge}.items():
                lines.append(f"b{value:b} {ids[key]}" if REPLAY.SIGNALS[key][1] > 1 else f"{value}{ids[key]}")
    return io.StringIO("\n".join(lines) + "\n")


class PortableTest(unittest.TestCase):
    def test_bf16_power_product_reference(self):
        word = lambda codes: b"".join(x.to_bytes(2, "little") for x in codes * 4)
        self.assertEqual(REPLAY.bf16_power_product(word([0x3f80, 0x4000, 0xbf80, 0xc000]), word([0x4000, 0x3f00, 0xbf80, 0x4000])),
                         word([0x4000, 0x3f80, 0x3f80, 0xc080]))
        normal = (0x3f80).to_bytes(2, "little") * 16
        for code in (0, 0x7f80, 0x7fc0, 0x3f81):
            with self.subTest(code=code), self.assertRaises(REPLAY.ReplayError):
                REPLAY.bf16_power_product(code.to_bytes(2, "little") * 16, normal)

    def test_initial_memory_pairs_differ(self):
        memory = REPLAY.initial_memory("vmul")
        self.assertEqual(len(memory), 8192)
        self.assertNotEqual(memory[:2048], memory[2048:4096])

    def test_cases_are_delay_free_and_end_with_ecall(self):
        for name, text in REPLAY.cases().items():
            self.assertNotIn("atlas.delay", text, name)
            self.assertIn('kind = "ecall"', text.splitlines()[-2], name)

    def test_vcd_samples_values_before_the_rising_edge(self):
        cycles = [{"pc": 4, "word": 0xdead}, {"pc": 8, "word": 0xbeef}, {"pc": 12, "word": 0}]
        samples = list(REPLAY.vcd_edges(vcd(cycles), "TOP.AtlasCore"))
        self.assertEqual([s["edge"] for s in samples], [1, 2, 3])
        self.assertEqual([s["values"]["pc"] for s in samples], [4, 8, 12])
        self.assertEqual(samples[1]["values"]["word"], 0xbeef)

    def test_vcd_rejects_missing_signal_and_backward_time(self):
        with self.assertRaises(REPLAY.ReplayError): list(REPLAY.vcd_edges(vcd([{}, {}], drop="xcmd")))
        text = vcd([{}, {}]).getvalue().replace("#30", "#10", 1)
        with self.assertRaises(REPLAY.ReplayError): list(REPLAY.vcd_edges(io.StringIO(text)))

    def test_unknown_values_are_rejected_by_the_validator(self):
        samples = [{"edge": 1, "values": {key: 0 for key in REPLAY.SIGNALS} | {"pc": None}}]
        with self.assertRaises(REPLAY.ReplayError): REPLAY.validate_compute_edges(samples, [0x73], "vmul")

    def test_word_decoding_and_stream_expansion(self):
        mnemonic, operands = BIND.decoded_operands(0x06100257)
        self.assertEqual((mnemonic, operands["rd"], operands["rs1"], operands["rs2"]), ("vmul.bf16", 4, 0, 2))
        with self.assertRaises(BIND.BindingError): BIND.decoded_operands(0)
        stream = dict(resource="mreg", write=True, first=128, count=64, age=2, step=1, anywhere=False, at_completion=False)
        points = BIND.finite_stream(stream, 100)
        self.assertEqual((points[0], points[-1]), (("mreg", True, 128, 102), ("mreg", True, 191, 165)))
        for field, value in [("anywhere", True), ("age", None), ("count", 0), ("step", 0)]:
            with self.subTest(field=field), self.assertRaises((BIND.BindingError, TypeError)):
                BIND.finite_stream({**stream, field: value}, 100)


@unittest.skipUnless(MODEL and BIN and FACTS, "requires a Verilated AtlasCore, compiler binaries and facts")
class VerilatorReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.work = Path(cls.temporary.name)
        run = subprocess.run([sys.executable, str(ROOT / "tools/replay-ee290-compute.py"), "--facts", FACTS, "--bin", BIN, "--model", MODEL, "--work", str(cls.work)],
                             capture_output=True, text=True)
        if run.returncode: raise AssertionError(run.stdout + run.stderr)
        cls.digest = hashlib.sha256(Path(FACTS).read_bytes()).hexdigest()
        cls.cases = {}
        for name in ("vmul_delay", "vmul_schedule", "dma_xlu_delay", "dma_xlu_schedule"):
            directory = cls.work / name
            words = [int(w, 16) for w in (directory / "program.hex").read_text().split()]
            with (directory / "trace.vcd").open() as stream: samples = list(REPLAY.vcd_edges(stream))
            cls.cases[name] = (samples, words, json.loads((directory / "events.json").read_text()), json.loads((directory / "timing.json").read_text()))

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def validate(self, name, samples=None, words=None):
        original, original_words = self.cases[name][:2]
        return REPLAY.validate_compute_edges(original if samples is None else samples, original_words if words is None else words, name.rsplit("_", 1)[0])

    def mutate(self, name, key, predicate, value):
        samples = copy.deepcopy(self.cases[name][0])
        sample = next(s for s in samples if predicate(s["values"]))
        sample["values"][key] = value(sample["values"][key]) if callable(value) else value
        with self.assertRaises(REPLAY.ReplayError): self.validate(name, samples)

    def bind(self, name, export=None, events=None, words=None):
        _, original_words, original_events, original_export = self.cases[name]
        return BIND.bind_export(original_export if export is None else export, original_events if events is None else events, self.digest, original_words if words is None else words)

    def test_replays_pass_with_full_drain(self):
        for name in self.cases:
            with self.subTest(name=name):
                result = self.validate(name)
                self.assertEqual(result["full_memory_checked_bytes"], 8192)
                self.assertGreaterEqual(result["terminal_drain_edges"], 2)
                self.assertGreater(self.bind(name)["memory_access_elements_bound"], 0)

    def test_vmul_trace_mutations_are_rejected(self):
        for key, predicate, value in [("prid0", lambda v: v["pr0"] and v["prid0"] == 1, 0), ("prrow0", lambda v: v["pr0"], lambda x: x ^ 1),
                                      ("pdata1", lambda v: v["presp1"], lambda x: x ^ 1), ("pwid0", lambda v: v["pw0"] and v["pwid0"] == 5, 4),
                                      ("pwdata0", lambda v: v["pw0"], lambda x: x ^ 1), ("pw0", lambda v: v["pw0"], None), ("presp1", lambda v: v["presp1"], None)]:
            with self.subTest(key=key): self.mutate("vmul_delay", key, predicate, value)

    def test_dma_trace_mutations_are_rejected(self):
        for key, predicate, value in [("dma_aa", lambda v: v["dma_av"] and v["dma_ar"], 0x90001000), ("dma_dd", lambda v: v["dma_dv"] and v["dma_dr"], lambda x: x ^ 1),
                                      ("dma_ds", lambda v: v["dma_dv"] and v["dma_dr"], 63), ("dma_vwd", lambda v: v["dma_vw"], lambda x: x ^ 1),
                                      ("dma_vdata", lambda v: v["dma_vresp"], lambda x: x ^ 1), ("dma_busy0", lambda v: v["dma_busy0"], 0)]:
            with self.subTest(key=key): self.mutate("dma_xlu_delay", key, predicate, value)

    def test_wrong_words_and_truncated_drain_are_rejected(self):
        for name, (samples, words, events, _) in self.cases.items():
            changed = words.copy(); changed[0] ^= 1
            halt = next(e["edge"] for e in events["endpoints"] if e["kind"] == "halt")
            with self.subTest(name=name):
                with self.assertRaises(REPLAY.ReplayError): self.validate(name, words=changed)
                with self.assertRaises(REPLAY.ReplayError): self.validate(name, [s for s in samples if s["edge"] <= halt])

    def test_export_mutations_are_rejected(self):
        name = "vmul_schedule"; export = self.cases[name][3]
        index = next(i["word_index"] for i in export["instructions"] if i["mnemonic"] == "vmul.bf16")
        footprint = lambda e: e["instructions"][index]["footprint"]
        for position, mutate in enumerate([lambda e: e["instructions"][index]["operands"].__setitem__("rd", 6),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("first", 32),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("age", 1),
                                           lambda e: footprint(e).__setitem__("done_age", 66),
                                           lambda e: footprint(e).__setitem__("read_release", 63),
                                           lambda e: footprint(e)["mreg_reads"].pop(),
                                           lambda e: footprint(e)["holds"].pop(0),
                                           lambda e: e["instructions"][index].__setitem__("epoch_offset", e["instructions"][index]["epoch_offset"] + 1),
                                           lambda e: e["evidence"].__setitem__("op_timing_sha256", "0" * 64),
                                           lambda e: e["resolver"].__setitem__("id", "other")]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(position=position), self.assertRaises(BIND.BindingError): self.bind(name, export=changed)

    def test_dma_completion_stays_dynamic(self):
        name = "dma_xlu_schedule"; export, events = self.cases[name][3], self.cases[name][2]
        dma = next(i for i in export["instructions"] if i["mnemonic"].startswith("dma.load"))["word_index"]
        for mutate in (lambda e: e["instructions"][dma]["footprint"].__setitem__("done_age", 50),
                       lambda e: e["instructions"][dma]["footprint"]["accesses"][-1].__setitem__("at_completion", False),
                       lambda e: e["instructions"][next(i["word_index"] for i in e["instructions"] if i["event_kind"] == "matching_wait_acceptance")].__setitem__("issue_epoch", 0)):
            changed = copy.deepcopy(export); mutate(changed)
            with self.assertRaises(BIND.BindingError): self.bind(name, export=changed)
        changed = copy.deepcopy(events); command = next(c for c in changed["commands"] if c["engine"] == "dma")
        command["memory"][0]["edge"] = command["release_edge"]
        with self.assertRaises(BIND.BindingError): self.bind(name, events=changed)


if __name__ == "__main__":
    unittest.main()
