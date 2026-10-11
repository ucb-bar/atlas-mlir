"""Compute replay checks: synthetic VCD/checker tests always run; the Verilator run is env-gated.

The real run needs ATLAS_EE290_COMPUTE_MODEL (a VAtlasCore built with test/ee290-compute-replay.cpp),
ATLAS_OOT_BIN_DIR (atlas-opt, atlas-emit) and ATLAS_OP_TIMING (merlin.op_timing.v1 facts). It replays every
case in tools/ee290_replay_cases, then mutates the captured traces, events and exports to confirm each check bites.
"""
import copy
import dataclasses
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
BIND, CASES = REPLAY.BIND, REPLAY.CASES
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


def btype(offset_bytes, rs1=13, rs2=14, funct3=4):
    imm = offset_bytes & 0x1fff
    return (imm >> 12) << 31 | ((imm >> 5) & 63) << 25 | rs2 << 20 | rs1 << 15 | funct3 << 12 | ((imm >> 1) & 15) << 8 | ((imm >> 11) & 1) << 7 | 0x63


class PortableTest(unittest.TestCase):
    def test_bf16_power_product_reference(self):
        word = lambda codes: b"".join(x.to_bytes(2, "little") for x in codes * 4)
        self.assertEqual(REPLAY.bf16_power_product(word([0x3f80, 0x4000, 0xbf80, 0xc000]), word([0x4000, 0x3f00, 0xbf80, 0x4000])),
                         word([0x4000, 0x3f80, 0x3f80, 0xc080]))
        normal = (0x3f80).to_bytes(2, "little") * 16
        for code in (0, 0x7f80, 0x7fc0, 0x3f81):
            with self.subTest(code=code), self.assertRaises(REPLAY.ReplayError):
                REPLAY.bf16_power_product(code.to_bytes(2, "little") * 16, normal)

    def test_cases_are_closed_and_well_formed(self):
        cases = CASES.load()
        self.assertLessEqual({"xlu", "vmul", "dma_xlu", "loop", "diamond_taken", "overlap_vls"}, set(cases))
        for name, case in cases.items():
            with self.subTest(name=name):
                self.assertIn("expect_vmem", REPLAY.image(case))
                self.assertEqual(case.ops[-1], CASES.END[-1])
                self.assertEqual(case.final, case.consumers == ("final",))
                if not case.final: self.assertNotIn("delay", [op for op, _ in case.ops])
        # The VMUL case's closed-form products agree with the validator's dataflow reference.
        vm, _, expected, _ = REPLAY.memories(cases["vmul"])
        self.assertEqual([expected[128 + n] for n in range(64)], [REPLAY.bf16_power_product(vm[n], vm[64 + n]) for n in range(64)])
        with self.assertRaises(REPLAY.ReplayError):
            REPLAY.image(dataclasses.replace(cases["xlu"], expect=lambda vm, dram: CASES.put(vm, 256, bytes(32))))

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
        with self.assertRaises(REPLAY.ReplayError): REPLAY.validate_compute_edges(samples, [0x73], CASES.load()["vmul"])

    def test_word_decoding_targets_and_stream_expansion(self):
        mnemonic, operands = BIND.decoded_operands(0x06100257)
        self.assertEqual((mnemonic, operands["rd"], operands["rs1"], operands["rs2"]), ("vmul.bf16", 4, 0, 2))
        self.assertEqual(BIND.decoded_operands(btype(-6))[0], "blt")
        self.assertEqual(BIND.decoded_operands(0x80006f), ("jal", {"rd": 0, "rs1": 0, "rs2": 0, "immediate": 8, "release": False}))
        self.assertEqual((BIND.branch_target(10, btype(-6)), BIND.branch_target(9, 0x80006f)), (7, 13))
        # Leaders: entry, the target, after the delay slot and after ECALL.
        self.assertEqual(REPLAY.leaders([0x13, 0x13, 0x13, btype(-4), 0x13, 0x73, 0x13, 0x73]), {0, 1, 5, 6})
        for word in (0, 0x1ef):  # unsupported, linking JAL
            with self.assertRaises(BIND.BindingError): BIND.decoded_operands(word)
        stream = dict(resource="mreg", write=True, first=128, count=64, age=2, step=1, anywhere=False, at_completion=False)
        points = BIND.finite_stream(stream, 100)
        self.assertEqual((points[0], points[-1]), (("mreg", True, 128, 102), ("mreg", True, 191, 165)))
        for field, value in [("anywhere", True), ("age", None), ("count", 0), ("step", 0)]:
            with self.subTest(field=field), self.assertRaises((BIND.BindingError, TypeError)):
                BIND.finite_stream({**stream, field: value}, 100)


@unittest.skipUnless(MODEL and BIN and FACTS, "requires a Verilated AtlasCore, compiler binaries and facts")
class VerilatorReplayTest(unittest.TestCase):
    NAMES = ("vmul_delay", "vmul_schedule", "dma_xlu_delay", "dma_xlu_schedule", "loop_delay", "loop_schedule",
             "dma_loop_schedule", "diamond_taken_delay", "diamond_fall_schedule", "overlap_vls", "overlap_vpu")

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.work = Path(cls.temporary.name)
        run = subprocess.run([sys.executable, str(ROOT / "tools/replay-ee290-compute.py"), "--facts", FACTS, "--bin", BIN, "--model", MODEL, "--work", str(cls.work)],
                             capture_output=True, text=True)
        if run.returncode: raise AssertionError(run.stdout + run.stderr)
        cls.log = run.stdout
        cls.digest = hashlib.sha256(Path(FACTS).read_bytes()).hexdigest()
        cls.cases, available = {}, CASES.load()
        for name in cls.NAMES:
            directory = cls.work / name
            words = [int(w, 16) for w in (directory / "program.hex").read_text().split()]
            with (directory / "trace.vcd").open() as stream: samples = list(REPLAY.vcd_edges(stream))
            case = available[name if name in available else name.rsplit("_", 1)[0]]
            cls.cases[name] = (samples, words, json.loads((directory / "events.json").read_text()), json.loads((directory / "timing.json").read_text()), case)

    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()

    def validate(self, name, samples=None, words=None):
        original, original_words, _, _, case = self.cases[name]
        return REPLAY.validate_compute_edges(original if samples is None else samples, original_words if words is None else words, case)

    def mutate(self, name, predicate, change):
        samples = copy.deepcopy(self.cases[name][0])
        sample = next(s for s in samples if predicate(s["values"], s["edge"]))
        change(sample["values"])
        with self.assertRaises(REPLAY.ReplayError): self.validate(name, samples)

    def bind(self, name, export=None, events=None):
        _, words, original_events, original_export, case = self.cases[name]
        export, events = original_export if export is None else export, original_events if events is None else events
        return BIND.bind_footprints(export, events, self.digest) if case.final else BIND.bind_export(export, events, self.digest, words)

    def rejects(self, name, export=None, events=None):
        with self.assertRaises(BIND.BindingError): self.bind(name, export, events)

    def test_replays_pass_with_full_drain(self):
        self.assertEqual(self.log.count("PASS "), sum(len(c.consumers) for c in CASES.load().values()))
        for name in self.cases:
            with self.subTest(name=name):
                result = self.validate(name)
                self.assertGreaterEqual(result["full_memory_checked_bytes"], 8192)
                self.assertGreaterEqual(result["terminal_drain_edges"], 2)
                self.assertGreater(self.bind(name)["memory_access_elements_bound"], 0)

    def test_loop_binds_every_iteration_to_one_drained_block(self):
        events, export = self.cases["loop_delay"][2:4]
        instances = self.bind("loop_delay")["block_instances"]
        self.assertEqual([i["block"] for i in instances], [0, 1, 1, 1, 2])
        loop = export["blocks"][1]
        self.assertEqual(loop["successors"], [1, 2])
        for previous, current in zip(instances[1:], instances[2:]):
            self.assertEqual(current["entry_edge"] - previous["entry_edge"], loop["successor_issue_offset"])
        commands = events["commands"]
        self.assertEqual([c["engine"] for c in commands], ["vload", "vstore"] * 3)
        self.assertTrue(all(c["release_edge"] <= e["edge"] for c, e in zip(commands[1::2], events["block_entries"][2:])))
        # A DMA wait inside the loop opens a new epoch in every iteration.
        instances = self.bind("dma_loop_schedule")["block_instances"]
        self.assertEqual([len(i["epoch_origins"]) for i in instances if i["block"] == 1], [2, 2, 2])

    def test_diamond_executes_only_the_selected_arm(self):
        for name, arm in (("diamond_taken_delay", 2), ("diamond_fall_schedule", 1)):
            with self.subTest(name=name):
                self.assertEqual([i["block"] for i in self.bind(name)["block_instances"]], [0, arm, 3])

    def test_overlapping_engines_keep_their_isolated_footprints(self):
        for name, pair in (("overlap_vls", ("vload", "vstore")), ("overlap_vpu", ("vpu", "vload"))):
            with self.subTest(name=name):
                commands = self.cases[name][2]["commands"]
                (first, second), = self.cases[name][2]["concurrent_commands"]
                self.assertEqual((commands[first]["engine"], commands[second]["engine"]), pair)
                self.assertEqual(commands[second]["edge"] - commands[first]["edge"], 1)
                self.assertGreater(self.bind(name)["memory_access_elements_bound"], 0)

    def test_vmul_trace_mutations_are_rejected(self):
        for key, predicate, value in [("prid0", lambda v: v["pr0"] and v["prid0"] == 1, 0), ("prrow0", lambda v: v["pr0"], lambda x: x ^ 1),
                                      ("pdata1", lambda v: v["presp1"], lambda x: x ^ 1), ("pwid0", lambda v: v["pw0"] and v["pwid0"] == 5, 4),
                                      ("pwdata0", lambda v: v["pw0"], lambda x: x ^ 1), ("pw0", lambda v: v["pw0"], None), ("presp1", lambda v: v["presp1"], None)]:
            with self.subTest(key=key):
                self.mutate("vmul_delay", lambda v, _: predicate(v), lambda v: v.__setitem__(key, value(v[key]) if callable(value) else value))

    def test_dma_trace_mutations_are_rejected(self):
        for key, predicate, value in [("dma_aa", lambda v: v["dma_av"] and v["dma_ar"], 0x90001000), ("dma_dd", lambda v: v["dma_dv"] and v["dma_dr"], lambda x: x ^ 1),
                                      ("dma_ds", lambda v: v["dma_dv"] and v["dma_dr"], 63), ("dma_vwd", lambda v: v["dma_vw"], lambda x: x ^ 1),
                                      ("dma_vdata", lambda v: v["dma_vresp"], lambda x: x ^ 1), ("dma_busy0", lambda v: v["dma_busy0"], 0)]:
            with self.subTest(key=key):
                self.mutate("dma_xlu_delay", lambda v, _: predicate(v), lambda v: v.__setitem__(key, value(v[key]) if callable(value) else value))

    def test_missing_idle_at_block_entry_is_rejected(self):
        events = self.cases["loop_delay"][2]
        entry = events["block_entries"][2]["edge"]
        for key in ("mreg_write_busy", "store_busy", "dma_av"):
            with self.subTest(key=key): self.mutate("loop_delay", lambda v, edge: edge == entry, lambda v: v.__setitem__(key, 1))
        changed = copy.deepcopy(events); changed["block_entries"][2]["engine_state"]["vls_store_busy"] = 1
        self.rejects("loop_delay", events=changed)

    def test_wrong_block_binding_is_rejected(self):
        export, events = self.cases["loop_delay"][3], self.cases["loop_delay"][2]
        store = next(i["word_index"] for i in export["instructions"] if i["mnemonic"] == "vstore")
        for position, mutate in enumerate([lambda e: e["blocks"][1].__setitem__("successors", [2]),
                                           lambda e: e["blocks"][1].__setitem__("successor_issue_offset", e["blocks"][1]["successor_issue_offset"] + 1),
                                           lambda e: e["instructions"][store].__setitem__("block", 0),
                                           lambda e: e["instructions"][store].__setitem__("epoch_offset", e["instructions"][store]["epoch_offset"] + 1),
                                           lambda e: e["blocks"][0].__setitem__("word_count", e["blocks"][0]["word_count"] + 1)]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(export=position): self.rejects("loop_delay", export=changed)
        for position, mutate in enumerate([lambda o: o["block_entries"].pop(2),
                                           lambda o: o["block_entries"][3].__setitem__("edge", o["block_entries"][2]["edge"]),
                                           lambda o: o["instructions"].pop(next(n for n, i in enumerate(o["instructions"]) if i["word_index"] == store))]):
            changed = copy.deepcopy(events); mutate(changed)
            with self.subTest(events=position): self.rejects("loop_delay", events=changed)

    def test_event_attributed_to_wrong_engine_is_rejected(self):
        # While both LSU paths are live, a VSTORE read moved to the XLU port and a VLOAD write moved to the VPU port.
        def move(source, target, fields):
            def change(v):
                for a, b in fields: v[b] = v[a]
                v[source], v[target] = 0, 1
            return change
        concurrent = lambda key: lambda v, _: v[key] and v["load_busy"] and v["store_busy"]
        self.mutate("overlap_vls", concurrent("mr"), move("mr", "xr", [("mr_id", "xrid"), ("mr_row", "xrrow")]))
        self.mutate("overlap_vls", concurrent("mw"), move("mw", "pw0", [("mw_id", "pwid0"), ("mw_row", "pwrow0"), ("mw_data", "pwdata0")]))
        events = self.cases["overlap_vls"][2]
        pair = events["concurrent_commands"][0]
        for position, mutate in enumerate([lambda load, store: store.__setitem__("engine", "vload"),
                                           lambda load, store: load["writes"].append(store["writes"].pop()),
                                           lambda load, store: store["reads"][5].__setitem__("edge", store["reads"][5]["edge"] + 1),
                                           lambda load, store: load.__setitem__("release_edge", store["release_edge"])]):
            changed = copy.deepcopy(events); mutate(*(changed["commands"][n] for n in pair))
            with self.subTest(position=position): self.rejects("overlap_vls", events=changed)

    def test_shifted_event_is_rejected(self):
        events = self.cases["loop_schedule"][2]
        for field in ("reads", "writes"):
            changed = copy.deepcopy(events); changed["commands"][3][field][7]["edge"] += 1
            with self.subTest(field=field): self.rejects("loop_schedule", events=changed)
        self.mutate("loop_schedule", lambda v, _: v["vw"] and v["vw_addr"] == 40, lambda v: v.__setitem__("vw_addr", 41))

    def test_wrong_words_and_truncated_drain_are_rejected(self):
        for name, (samples, words, events, _, _) in self.cases.items():
            changed = words.copy(); changed[0] ^= 1
            halt = next(e["edge"] for e in events["endpoints"] if e["kind"] == "halt")
            with self.subTest(name=name):
                with self.assertRaises(REPLAY.ReplayError): self.validate(name, words=changed)
                with self.assertRaises(REPLAY.ReplayError): self.validate(name, [s for s in samples if s["edge"] <= halt])

    def test_export_holds_bind_own_release_and_bound_policy(self):
        name = "vmul_schedule"; export = self.cases[name][3]
        index = next(i["word_index"] for i in export["instructions"] if i["mnemonic"] == "vmul.bf16")
        load = next(i["word_index"] for i in export["instructions"] if i["mnemonic"] == "vload")
        footprint = lambda e, n=index: e["instructions"][n]["footprint"]
        hold = lambda e, unit, n=index: next(h for h in footprint(e, n)["holds"] if h["unit"] == unit)
        for position, mutate in enumerate([lambda e: e["instructions"][index]["operands"].__setitem__("rd", 6),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("first", 32),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("age", 1),
                                           lambda e: footprint(e).__setitem__("done_age", 64),
                                           lambda e: footprint(e).__setitem__("read_release", 63),
                                           lambda e: footprint(e)["mreg_reads"].pop(),
                                           lambda e: footprint(e)["holds"].pop(0),
                                           lambda e: hold(e, "VPU").__setitem__("to", 64),
                                           lambda e: hold(e, "VPU").__setitem__("to", 66),
                                           lambda e: hold(e, "VLOAD path").__setitem__("to", 64),
                                           lambda e: hold(e, "VMEM bank", load).__setitem__("from", 2),
                                           lambda e: hold(e, "VMEM bank", load).__setitem__("unit", "MXU port"),
                                           lambda e: e["instructions"][index].__setitem__("epoch_offset", e["instructions"][index]["epoch_offset"] + 1),
                                           lambda e: e["evidence"].__setitem__("op_timing_sha256", "0" * 64),
                                           lambda e: e["resolver"].__setitem__("id", "other")]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(position=position): self.rejects(name, export=changed)
        # A policy hold or lifetime may outlast the engine's own activity.
        for mutate in (lambda e: hold(e, "VLOAD path").__setitem__("to", 100), lambda e: hold(e, "VSTORE path", load).__setitem__("to", 90),
                       lambda e: footprint(e).__setitem__("done_age", 80)):
            changed = copy.deepcopy(export); mutate(changed)
            self.assertGreater(self.bind(name, export=changed)["memory_access_elements_bound"], 0)

    def test_dma_completion_stays_dynamic(self):
        name = "dma_xlu_schedule"; export, events = self.cases[name][3], self.cases[name][2]
        dma = next(i for i in export["instructions"] if i["mnemonic"].startswith("dma.load"))["word_index"]
        for mutate in (lambda e: e["instructions"][dma]["footprint"].__setitem__("done_age", 50),
                       lambda e: e["instructions"][dma]["footprint"]["accesses"][-1].__setitem__("at_completion", False),
                       lambda e: e["instructions"][next(i["word_index"] for i in e["instructions"] if i["event_kind"] == "matching_wait_acceptance")].__setitem__("issue_epoch", 0)):
            changed = copy.deepcopy(export); mutate(changed)
            self.rejects(name, export=changed)
        changed = copy.deepcopy(events); command = next(c for c in changed["commands"] if c["engine"] == "dma")
        command["memory"][0]["edge"] = command["release_edge"]
        self.rejects(name, events=changed)


if __name__ == "__main__":
    unittest.main()
