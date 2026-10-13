"""Compute replay checks: synthetic VCD/checker tests always run; the Verilator run is env-gated.

The real run needs ATLAS_EE290_COMPUTE_MODEL (a VAtlasCore built with test/ee290-compute-replay.cpp),
ATLAS_OOT_BIN_DIR (atlas-opt, atlas-emit) and ATLAS_OP_TIMING (merlin.op_timing.v1 facts); with ATLAS_PAIR_FACTS
(pair facts) the cases that need them run too. It replays every case in tools/ee290_replay_cases, then mutates the captured traces, events and exports to confirm each check bites.
"""
import copy
import dataclasses
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_rtl_timing import PROVIDER

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay", ROOT / "tools/replay-ee290-compute.py")
REPLAY = importlib.util.module_from_spec(spec); spec.loader.exec_module(REPLAY)
BIND, CASES = REPLAY.BIND, REPLAY.CASES
MODEL, BIN, FACTS, PAIRS = (os.environ.get(k) for k in ("ATLAS_EE290_COMPUTE_MODEL", "ATLAS_OOT_BIN_DIR", "ATLAS_OP_TIMING", "ATLAS_PAIR_FACTS"))


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
                self.assertEqual(case.final, case.consumers == ("final",))
                if case.source:
                    # A virtual program is lowered by its pipeline; the lowering emits the terminal.
                    self.assertTrue(case.pipeline and not case.ops and not case.final)
                    self.assertRegex(case.mlir(), "atlas.virtual_(output_bf16|dma_store_fp8)")
                    continue
                self.assertEqual(case.ops[-1], CASES.END[-1])
                if not case.final: self.assertNotIn("delay", [op for op, _ in case.ops])
        # The VMUL case's closed-form products agree with the validator's dataflow reference.
        vm, _, expected, _ = REPLAY.memories(cases["vmul"])
        self.assertEqual([expected[128 + n] for n in range(64)], [REPLAY.bf16_power_product(vm[n], vm[64 + n]) for n in range(64)])
        with self.assertRaises(REPLAY.ReplayError):
            REPLAY.image(dataclasses.replace(cases["xlu"], expect=lambda vm, dram: CASES.put(vm, 256, bytes(32))))

    def test_mxu_reference_is_exact_and_independent(self):
        from ee290_replay_cases import mxu
        result = mxu.reference(mxu.ACTIVATION, mxu.WEIGHTS)
        # C[0,0] = A[0,0] W[0,0] = 0.5 * 0.25; C[0,1] = A[0,5] W[1,5] = 1 * -0.25 (hand-computed).
        self.assertEqual([int.from_bytes(result[o:o + 2], "little") for o in (0, 2)], [0x3e00, 0xbe80])
        self.assertEqual(len({result[32 * r:32 * r + 32] + result[1024 + 32 * r:1056 + 32 * r] for r in range(32)}), 32)  # distinct rows
        transposed = bytes(mxu.WEIGHTS[32 * k + j] for j in range(32) for k in range(32))
        self.assertNotEqual(mxu.reference(mxu.ACTIVATION, transposed), result)
        with self.assertRaises(ValueError): mxu.exact_bf16(0x10001)  # needs rounding
        with self.assertRaises(ValueError): mxu.decode(0x7f)
        mnemonic, operands = BIND.decoded_operands(0x14000077 | 1 << 25 | 1 << 19 | 1 << 7)
        self.assertEqual((mnemonic, operands["rd"], operands["rs1"], operands["rs2"]), ("vmatmul.mxu1", 1, 0, 1))

    def test_vpu_lane_references_are_exact(self):
        from fractions import Fraction
        from ee290_replay_cases import vpu_ops as V
        self.assertEqual([V.relu(c) for c in (0x8000, 0x7fc0, 0xffc1, 0x0001, 0xbf80)], [0, 0x7fc0, 0, 0x0001, 0])
        self.assertEqual((V.ELEMENTWISE["vadd.bf16"](0x3f80, 0x3f80), V.ELEMENTWISE["vsub.bf16"](0x3f80, 0x4040)), (0x4000, 0xc000))
        self.assertEqual((V.vmax(0x8000, 0), V.vmin(0x8000, 0), V.vmax(0xbf80, 0xc000), V.vmin(0x7f80, 0x4000)), (0, 0x8000, 0xbf80, 0x4000))
        self.assertEqual([V.exp_special(c) for c in (0, 0x8000, 0xff80, 0x7f80, 0x4300)], [0x3f80, 0x3f80, 0, 0x7f80, 0x7f80])
        self.assertEqual([V.exp2_integer(c) for c in (0x4100, 0xc2c8, 0x42b2)], [0x4380, 0x0d80, 0x7f80])  # 8, -100, 89
        self.assertEqual((V.total([0x3f80] * 31 + [0x4000]), V.total([0] * 32)), (0x4204, 0))
        self.assertEqual(V.vli("row", 0x3f80, 32), V.pack([0x3f80] * 16))
        self.assertEqual(V.vli("one", 0x3f80, 1), bytes(32))
        self.assertEqual(V.fp8_pack(V.pack([0x3f80] * 16), V.pack([0xc000] * 16), 128), b"\x30" * 16 + b"\xb8" * 16)
        self.assertEqual(V.fp8_unpack(b"\x30" * 16 + b"\xb8" * 16, 1, 128), V.pack([0xc000] * 16))
        refused = [lambda: V.ELEMENTWISE["vadd.bf16"](0x3f80, 0x3bc0), lambda: V.ELEMENTWISE["vadd.bf16"](0x3f80, 0xbf80),
                   lambda: V.exp_special(0x3f80), lambda: V.exp2_integer(0x3f00), lambda: V.total([0x4b80, 0x3f80]), lambda: V.total([0x3f80, 0xbf80]),
                   lambda: V.ELEMENTWISE["vsquare.bf16"](0x3f88), lambda: V.fp8_from_bf16(0x3f81, 0), lambda: V.fp8_from_bf16(0x3f80, -7),
                   lambda: V.bf16_from_fp8(0x38, 200), lambda: V.column_reduce("vredsum.bf16", [bytes(32)] * 32), lambda: V.elementwise("vsin.bf16", bytes(32))]
        for position, function in enumerate(refused):
            with self.subTest(refused=position), self.assertRaises(ValueError): function()
        self.assertEqual(V.exact(Fraction(-3, 8)), 0xbec0)

    def test_mxu_accumulate_reference_is_exact(self):
        from ee290_replay_cases import mxu
        # Accumulating the products onto themselves doubles them: C[0,0] = 2 * 0.125, C[0,1] = 2 * -0.25.
        doubled = mxu.reference(mxu.ACTIVATION, mxu.WEIGHTS, mxu.PRODUCTS)
        self.assertEqual([int.from_bytes(doubled[o:o + 2], "little") for o in (0, 2)], [0x3e80, 0xbf00])
        self.assertEqual(mxu.accumulator_codes(mxu.panels(mxu.ACC_BF16)), mxu.ACC_BF16)
        self.assertEqual((mxu.dequantize(b"\x38\xb0"), mxu.quantize((0x3f80).to_bytes(2, "little") * 32, 126)), (b"\x80\x3f\x00\xbf", b"\x30" * 32))
        negated = [code ^ 0x8000 for code in mxu.PRODUCTS]
        for function in (lambda: mxu.reference(mxu.ACTIVATION, mxu.WEIGHTS, negated), lambda: mxu.reference(mxu.ACTIVATION, mxu.WEIGHTS, [0x4980] * 1024),
                         lambda: mxu.quantize((0x3f88).to_bytes(2, "little") * 32, 127), lambda: mxu.quantize((0x3f80).to_bytes(2, "little") * 32, 100)):
            with self.assertRaises(ValueError): function()

    def test_new_words_decode_to_the_selected_mnemonics(self):
        vli = 0x3f80 << 16 | 1 << 13 | 2 << 7 | 0x5f
        seld = 1024 << 20 | 6 << 15 | 6 << 12 | 3 << 7 | 0x03
        for word, mnemonic, fields in ((vli, "vli.row", {"rd": 2, "immediate": 0x3f80}), (seld, "seld", {"rd": 3, "rs1": 6, "immediate": 1024}),
                                       (0x48 << 25 | 4 << 13 | 2 << 7 | 0x57, "vrelu.bf16", {"rd": 2, "rs1": 4}),
                                       (0x44 << 25 | 6 << 19 | 3 << 13 | 9 << 7 | 0x57, "vpack.bf16.fp8", {"rd": 9, "rs1": 3, "rs2": 6}),
                                       (0x18000077 | 1 << 25 | 1 << 19 | 2 << 13 | 1 << 7, "vmatmul.acc.mxu1", {"rd": 1, "rs1": 2, "rs2": 1}),
                                       (4 << 20 | 10 << 15 | 2 << 12 | 16 << 7 | 0x03, "lw", {"rd": 16, "rs1": 10, "immediate": 4}),
                                       (127 << 20 | 7 << 12 | 3 << 7 | 0x03, "seli", {"rd": 3, "immediate": 127}),
                                       (17 << 20 | 12 << 15 | 2 << 12 | 20 << 7 | 0x23, "sw", {"rs1": 12, "rs2": 17, "immediate": 20})):
            with self.subTest(mnemonic=mnemonic):
                decoded, operands = BIND.decoded_operands(word)
                self.assertEqual((decoded, {k: operands[k] for k in fields}), (mnemonic, fields))
        for word in (0x50 << 25 | 0x57, 4 << 13 | 0x5f):
            with self.assertRaises(BIND.BindingError): BIND.decoded_operands(word)
        # A final stream's reference keeps one DMA pending: each wait follows its own launch.
        ops = CASES.load()["dma_pair_in_flight"].ops
        order = [op[0] + op[1].split("channel = ")[1][0] for op in REPLAY.serialized(ops) if op[0] in ("dma", "dma_wait")]
        self.assertEqual(order, ["dma0", "dma_wait0", "dma1", "dma_wait1"])

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
             "dma_loop_schedule", "diamond_taken_delay", "diamond_fall_schedule", "overlap_vls", "overlap_vpu",
             "mxu0_tile_delay", "mxu1_tile_schedule", "vpu_relu_delay", "vpu_binary_schedule", "vpu_unary_delay", "vpu_reduce_schedule",
             "vpu_vli_delay", "vpu_pack_schedule", "mxu0_acc_bf16_delay", "mxu1_acc_bf16_schedule", "mxu0_acc_fp8_delay",
             "mxu1_acc_fp8_schedule", "dma_pair_in_flight", "virtual_relu_delay", "virtual_relu_schedule",
             "mxu0_under_dma_load_schedule", "mxu1_under_dma_store_delay", "mxu0_beside_vector_delay", "mxu1_beside_vector_schedule",
             "virtual_dma_mxu_delay", "mxu_both_units_schedule",
             "virtual_fp8_mlp_delay", "virtual_fp8_mlp_scheduled_schedule",
             *(("slots_exp_relu_delay", "slots_chain_delay", "slots_chain_schedule") if PAIRS else ()))
    # Selected mnemonic -> facts block (vpu.<op>, mxu<u>.<op>).
    FACTS = {"vadd.bf16": "add", "vsub.bf16": "sub", "vmul.bf16": "mul", "vmaximum.bf16": "pairmax", "vminimum.bf16": "pairmin", "vmov": "mov",
             "vexp.bf16": "exp", "vexp2.bf16": "exp2", "vsquare.bf16": "square", "vcube.bf16": "cube", "vrelu.bf16": "relu",
             "vredsum.row.bf16": "rsum", "vredmax.row.bf16": "rmax", "vredmin.row.bf16": "rmin", "vredsum.bf16": "csum", "vredmax.bf16": "cmax",
             "vpack.bf16.fp8": "fp8pack", "vunpack.fp8.bf16": "fp8unpack", "vli.all": "vliAll", "vli.row": "vliRow", "vli.col": "vliCol",
             "vli.one": "vliOne", "vmatpush.weight": "push_weight", "vmatpush.acc.fp8": "push_acc_fp8", "vmatpush.acc.bf16": "push_acc_bf16",
             "vmatpop.fp8.acc": "pop_acc_fp8", "vmatpop.bf16.acc": "pop_acc_bf16", "vmatmul": "matmul", "vmatmul.acc": "matmul_acc"}

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(); cls.work = Path(cls.temporary.name)
        run = subprocess.run([sys.executable, str(ROOT / "tools/replay-ee290-compute.py"), "--facts", FACTS, "--bin", BIN, "--model", MODEL, "--work", str(cls.work),
                              *(("--pairs", PAIRS) if PAIRS else ())], capture_output=True, text=True)
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

    def test_virtual_program_is_lowered_untimed_then_timed_from_the_facts(self):
        # Verified, scheduled, allocated and lowered untimed; facts selected on the lowering; timed, emitted and run.
        emit = Path(BIN) / "atlas-emit"
        for name in ("virtual_relu_delay", "virtual_relu_schedule"):
            with self.subTest(name=name):
                directory = self.work / name
                lowered, final = ((directory / f).read_text() for f in ("lowered.mlir", "final.mlir"))
                self.assertIn('atlas.generated_from_virtual = "resource-contract-v5"', lowered)
                self.assertIn('atlas.timing_state = "untimed"', lowered)
                self.assertNotIn("atlas.rtl_evidence", lowered)
                self.assertIn('atlas.timing_state = "timed"', final)
                self.assertIn(f'atlas.timing_provider = "{PROVIDER}"', final)
                self.assertIn(f'op_timing_sha256 = "{self.digest}"', final)
                words = (directory / "program.hex").read_text().split()
                structured = subprocess.run([Path(BIN) / "atlas-opt", directory / "final.mlir", "--convert-atlas-to-llvm-calls", "--finalize-atlas-llvm-calls"],
                                            capture_output=True, text=True)
                self.assertEqual(structured.returncode, 0, structured.stderr)
                self.assertEqual([w[2:] for w in re.findall(r"\.word (0x[0-9a-f]{8})", structured.stdout)], words)
                export = self.cases[name][3]
                self.assertEqual(export["provider"]["id"], PROVIDER)
                self.assertEqual([i["mnemonic"] for i in export["instructions"] if i["mnemonic"].startswith(("dma.", "vload", "vstore", "vrelu"))].count("vrelu.bf16"), 1)
                self.assertTrue(self.bind(name)["dynamic_dma_intervals"])

    def test_replays_pass_with_full_drain(self):
        self.assertEqual(self.log.count("PASS "), sum(len(c.consumers) for c in CASES.load().values() if PAIRS or not c.pairs))
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

    def test_fp8_mlp_relayout_binds_each_iteration_to_its_entry_context(self):
        # The pack relayout loop moves its LW/SW addresses every iteration; each instance binds the context its registers hold.
        for name in ("virtual_fp8_mlp_delay", "virtual_fp8_mlp_scheduled_schedule"):
            with self.subTest(name=name):
                events, export = self.cases[name][2:4]
                loop = next(b for b in export["blocks"] if "entry_contexts" in b)
                instances = [i for i in self.bind(name)["block_instances"] if i["block"] == loop["index"]]
                self.assertEqual((len(instances), len(loop["entry_contexts"])), (32, 32))
                self.assertTrue(all(i["entry_context_groups"] == 1 for i in instances))
                self.assertEqual(sorted({a["address"] >> 5 for a in events["scalar_memory"] if a["kind"] == "sw"}), list(range(4128, 4160)))
                swapped = copy.deepcopy(export); contexts = next(b for b in swapped["blocks"] if "entry_contexts" in b)["entry_contexts"]
                contexts[0]["footprints"], contexts[1]["footprints"] = contexts[1]["footprints"], contexts[0]["footprints"]
                self.rejects(name, export=swapped)
                missing = copy.deepcopy(export); next(b for b in missing["blocks"] if "entry_contexts" in b)["entry_contexts"].pop(7)
                self.rejects(name, export=missing)
                late = copy.deepcopy(events); late["scalar_memory"][0]["write_age"] += 1
                self.rejects(name, events=late)
        # A store to another word, or with other data, than its decoded SW is rejected by the validator.
        self.mutate("virtual_fp8_mlp_delay", lambda v, e: v["scmd"] and v["scmd_store"], lambda v: v.update(scmd_addr=v["scmd_addr"] + 4))
        self.mutate("virtual_fp8_mlp_delay", lambda v, e: v["scmd"] and v["scmd_store"], lambda v: v.update(scmd_wdata=v["scmd_wdata"] ^ 1))
        self.mutate("virtual_fp8_mlp_delay", lambda v, e: v["xw_en"] and v["xw_addr"] == 16, lambda v: v.update(xw_data=v["xw_data"] ^ 1))

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

    @unittest.skipUnless(PAIRS, "requires pair facts (ATLAS_PAIR_FACTS)")
    def test_two_vpu_commands_run_in_their_own_slots(self):
        for name, slots in (("slots_exp_relu_delay", [0, 0, 1]), ("slots_chain_delay", [0, 0, 1, 0]), ("slots_chain_schedule", [0, 0, 1, 0])):
            with self.subTest(name=name):
                vpus = [c for c in self.cases[name][2]["commands"] if c["engine"] == "vpu"]
                self.assertEqual([c["slot"] for c in vpus], slots)
                # Every command after the first VPU one issues while the one before it is live, in the other slot.
                self.assertTrue(all(b["edge"] < a["release_edge"] and a["slot"] != b["slot"] for a, b in zip(vpus[1:], vpus[2:])))
                self.assertGreater(self.bind(name)["memory_access_elements_bound"], 0)
        first, second = [c for c in self.cases["slots_exp_relu_delay"][2]["commands"] if c["engine"] == "vpu"][1:]
        self.assertEqual(second["edge"] - first["edge"], 1)
        # A read of the second slot moved to the first slot's port, or the second slot's write dropped, is rejected.
        overlap = lambda key: lambda v, edge: v[key] and second["edge"] <= edge < first["release_edge"]
        self.mutate("slots_exp_relu_delay", overlap("pr1"), lambda v: v.update(pr0=1, pr1=0, prid0=v["prid1"], prrow0=v["prrow1"]))
        self.mutate("slots_exp_relu_delay", overlap("pw1"), lambda v: v.__setitem__("pw1", 0))
        changed = copy.deepcopy(self.cases["slots_exp_relu_delay"][2])
        command = next(c for c in changed["commands"] if c["edge"] == second["edge"])
        command["reads"][3]["edge"] += 1
        self.rejects("slots_exp_relu_delay", events=changed)

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
        # The VMUL's holds on other engines (they may start at next_issue_age, when the VPU admits a command).
        policy = lambda e: [h for h in footprint(e)["holds"] if h["unit"] in ("VLOAD path", "VSTORE path", "XLU")]
        for position, mutate in enumerate([lambda e: e["instructions"][index]["operands"].__setitem__("rd", 6),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("first", 32),
                                           lambda e: footprint(e)["accesses"][0].__setitem__("age", 1),
                                           lambda e: footprint(e).__setitem__("done_age", 64),
                                           lambda e: footprint(e).__setitem__("read_release", 63),
                                           lambda e: footprint(e)["mreg_reads"].pop(),
                                           lambda e: footprint(e)["holds"].pop(0),
                                           lambda e: [h.__setitem__("to", 63) for h in footprint(e)["holds"] if h["unit"] == "VPU"],
                                           lambda e: hold(e, "VPU").__setitem__("to", 66),
                                           lambda e: [h.__setitem__("to", min(h["to"], 64)) for h in policy(e)],
                                           lambda e: hold(e, "VMEM bank", load).__setitem__("from", 2),
                                           lambda e: hold(e, "VMEM bank", load).__setitem__("unit", "MXU port"),
                                           lambda e: e["instructions"][index].__setitem__("epoch_offset", e["instructions"][index]["epoch_offset"] + 1),
                                           lambda e: e["evidence"].__setitem__("op_timing_sha256", "0" * 64),
                                           lambda e: e["provider"].__setitem__("id", "other")]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(position=position): self.rejects(name, export=changed)
        # A policy hold or lifetime may outlast the engine's own activity.
        for mutate in (lambda e: policy(e)[-1].__setitem__("to", 100), lambda e: [h.__setitem__("to", 90) for h in footprint(e, load)["holds"] if h["unit"] in ("VPU", "VSTORE path")],
                       lambda e: footprint(e).__setitem__("done_age", 80)):
            changed = copy.deepcopy(export); mutate(changed)
            self.assertGreater(self.bind(name, export=changed)["memory_access_elements_bound"], 0)

    def test_mxu_overlaps_dma_and_vector_engines(self):
        # MXU commands were live together with a pending transfer, with VPU, XLU, VLOAD and VSTORE commands and with the other MXU.
        for name, unit, others in (("mxu0_under_dma_load_schedule", "mxu0", {"dma"}), ("mxu1_under_dma_store_delay", "mxu1", {"dma"}),
                                   ("virtual_dma_mxu_delay", "mxu0", {"dma"}), ("mxu0_beside_vector_delay", "mxu0", {"vpu", "xlu", "vload", "vstore"}),
                                   ("mxu1_beside_vector_schedule", "mxu1", {"vpu", "xlu", "vload", "vstore"}),
                                   ("mxu_both_units_schedule", "mxu0", {"mxu1"})):
            commands = self.cases[name][2]["commands"]
            pairs = {frozenset((commands[i]["engine"], commands[j]["engine"])) for i, j in self.cases[name][2]["concurrent_commands"]}
            with self.subTest(name=name):
                self.assertTrue({frozenset((unit, other)) for other in others} <= pairs, pairs)

    def test_seli_writes_its_scale_register_on_issue(self):
        # #15's virtual FP8 readout sets its scale with SELI; the write must carry the decoded register and immediate.
        name = "virtual_dma_mxu_delay"
        for key, value in (("sew_data", lambda x: x ^ 1), ("sew_idx", lambda x: x ^ 1), ("sew", lambda x: 0)):
            with self.subTest(key=key):
                self.mutate(name, lambda v, _: v["sew"], lambda v: v.__setitem__(key, value(v[key])))

    def test_mxu_holds_bind_its_own_unit(self):
        name = "mxu0_beside_vector_delay"; export = self.cases[name][3]
        index = next(i["word_index"] for i in export["instructions"] if i["mnemonic"] == "vmatmul.mxu0")
        holds = lambda e: e["instructions"][index]["footprint"]["holds"]
        self.assertEqual([(h["unit"], h["index"]) for h in holds(export)], [("MXU in-flight matmuls", 0)])
        for mutate in (lambda e: holds(e)[0].__setitem__("index", 1), lambda e: holds(e).append({**holds(e)[0], "index": 1}),
                       lambda e: holds(e).pop(), lambda e: holds(e)[0].__setitem__("to", holds(e)[0]["to"] - 1),
                       lambda e: holds(e).append({**holds(e)[0], "unit": "MXU port"})):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(mutation=len(holds(changed))): self.rejects(name, export=changed)
        # A policy hold on another engine may outlast the MXU's activity, as for the other engines.
        changed = copy.deepcopy(export); holds(changed).append({**holds(changed)[0], "unit": "VPU", "index": 0})
        self.assertGreater(self.bind(name, export=changed)["memory_access_elements_bound"], 0)

    def test_mxu_events_match_the_facts(self):
        facts = {b["name"]: b for b in json.loads(Path(FACTS).read_text())["op_timing"]}
        for name, unit in (("mxu0_tile_delay", 0), ("mxu1_tile_schedule", 1)):
            commands = [c for c in self.cases[name][2]["commands"] if c["engine"] == f"mxu{unit}"]
            self.assertEqual([c["mnemonic"] for c in commands], [f"vmatpush.weight.mxu{unit}", f"vmatmul.mxu{unit}", f"vmatpop.bf16.acc.mxu{unit}"])
            for command, operation, group, field, resource in ((commands[0], "push_weight", "weight_write", "writes", "weight"),
                                                               (commands[1], "matmul", "acc_write", "writes", "acc"),
                                                               (commands[1], "matmul", "compute", "compute", None),
                                                               (commands[2], "pop_acc_bf16", "acc_store", "reads", "acc")):
                with self.subTest(name=name, group=group):
                    event = facts[f"mxu{unit}.{operation}"]["events"][group]
                    ages = [e["age"] for e in command[field] if resource in (None, e.get("resource"))]
                    self.assertEqual((min(ages), max(ages), len(ages)), (event["first_age"], event["last_age"], event["count"]))

    def test_mxu_trace_mutations_are_rejected(self):
        mutations = [  # shifted accumulator write row, wrong slots, missing compute beat, corrupted accumulator data, early pop data
            ("mxu0_tile_delay", lambda v: v["m0_aw"] and v["m0_aw_row"] == 3, "m0_aw_row", 4),
            ("mxu1_tile_schedule", lambda v: v["m1_ww"] and v["m1_ww_lane"] == 7, "m1_ww_slot", 0),
            ("mxu1_tile_schedule", lambda v: v["m1_cv"], "m1_cslot", 0),
            ("mxu1_tile_schedule", lambda v: v["m1_as"] and v["m1_as_row"] == 9, "m1_as_acc", 0),
            ("mxu0_tile_delay", lambda v: v["m0_cv"], "m0_cv", 0),
            ("mxu0_tile_delay", lambda v: v["m0_aw"] and v["m0_aw_row"] == 11, "m0_awd3", lambda x: x ^ 1),
            ("mxu0_tile_delay", lambda v: v["m0_w1"] and v["m0_wrow1"] == 2, "m0_wdata1", lambda x: x ^ 1 << 40),
            ("mxu1_tile_schedule", lambda v: v["m1_cmd"] and v["m1_op"] == 5, "m1_slot", 0),
            ("mxu0_tile_delay", lambda v: v["m0_aw"] and v["m0_aw_row"] == 31, "m0_aw", 0)]
        for name, predicate, key, value in mutations:
            with self.subTest(name=name, key=key):
                self.mutate(name, lambda v, _: predicate(v), lambda v: v.__setitem__(key, value(v[key]) if callable(value) else value))

    def test_mxu_binding_mutations_are_rejected(self):
        name = "mxu0_tile_delay"; export, events = self.cases[name][3], self.cases[name][2]
        index = lambda e, stem: next(i["word_index"] for i in e["instructions"] if i["mnemonic"] == stem + ".mxu0")
        footprint = lambda e, stem: e["instructions"][index(e, stem)]["footprint"]
        acc = lambda e: next(a for a in footprint(e, "vmatmul")["accesses"] if a["resource"] == "acc")
        for position, mutate in enumerate([lambda e: acc(e).__setitem__("age", acc(e)["age"] + 1),            # accumulator write shifted
                                           lambda e: acc(e).__setitem__("first", acc(e)["first"] + 32),         # wrong accumulator slot
                                           lambda e: e["instructions"][index(e, "vmatmul")]["operands"].__setitem__("rs2", 1),
                                           lambda e: footprint(e, "vmatmul")["holds"].append({**footprint(e, "vmatmul")["holds"][0], "index": 1}),  # the other MXU
                                           lambda e: footprint(e, "vmatpop.bf16.acc")["holds"][0].__setitem__("to", 31),
                                           lambda e: next(a for a in footprint(e, "vmatmul")["accesses"] if a["resource"] == "weight" and a["age"]).__setitem__("age", 31),
                                           lambda e: footprint(e, "vmatpush.weight")["accesses"][1].__setitem__("resource", "acc")]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(export=position): self.rejects(name, export=changed)
        matmul = lambda o: next(c for c in o["commands"] if c["engine"] == "mxu0" and c["op"] == 5)
        for position, mutate in enumerate([lambda o: matmul(o)["writes"][0].__setitem__("edge", matmul(o)["writes"][0]["edge"] + 1),
                                           lambda o: matmul(o)["writes"][4].__setitem__("id", 1),
                                           lambda o: matmul(o).__setitem__("release_edge", matmul(o)["release_edge"] + 1),
                                           lambda o: matmul(o)["compute"].append({"edge": matmul(o)["release_edge"] + 1, "row": 32})]):
            changed = copy.deepcopy(events); mutate(changed)
            with self.subTest(events=position): self.rejects(name, events=changed)

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

    def test_new_op_events_match_the_facts(self):
        facts, seen = {b["name"]: b["events"] for b in json.loads(Path(FACTS).read_text())["op_timing"]}, set()
        group = lambda e, field: (f"{'write' if field == 'writes' else 'read'}{e['port']}" if "resource" not in e else "compute" if field == "compute" else
                                  {("reads", "mreg"): f"mreg_read{e['port']}", ("writes", "mreg"): f"mreg_write{e['port']}", ("writes", "weight"): "weight_write",
                                   ("reads", "acc"): "acc_read" if e["port"] == 2 else "acc_store", ("writes", "acc"): "acc_load" if e["port"] == 3 else "acc_write"}[field, e["resource"]])
        for name in self.NAMES:
            for command in (c for c in self.cases[name][2]["commands"] if c["engine"] == "vpu" or c["engine"].startswith("mxu")):
                stem, unit = command["mnemonic"].rsplit(".mxu", 1) if ".mxu" in command["mnemonic"] else (command["mnemonic"], None)
                block = facts[f"mxu{unit}.{self.FACTS[stem]}" if unit else "vpu." + self.FACTS[stem]]
                observed = {}
                for field in ("reads", "writes", "compute"):
                    for e in command.get(field, ()): observed.setdefault(group(e, field) if field != "compute" else "compute", []).append(e["age"])
                expected = {k: (g["first_age"], g["last_age"], g["count"]) for k, g in block.items() if g["count"] and not k.endswith("busy")}
                with self.subTest(name=name, mnemonic=command["mnemonic"]):
                    self.assertEqual({k: (min(a), max(a), len(a)) for k, a in observed.items()}, expected)
                seen.add(command["mnemonic"])
        self.assertLessEqual({"vrelu.bf16", "vredsum.bf16", "vli.one", "vpack.bf16.fp8", "vmatmul.acc.mxu0", "vmatpop.fp8.acc.mxu1", "vmatpush.acc.fp8.mxu1"}, seen)

    def test_new_decoder_trace_mutations_are_rejected(self):
        flip = lambda bit=0: lambda x: x ^ 1 << bit
        mutations = [  # (case, predicate, signal, value): shifted, wrong register/slot/channel and corrupted data
            ("vpu_relu_delay", lambda v: v["pw0"] and v["pwrow0"] == 5, "pwdata0", flip(15)),
            ("vpu_relu_delay", lambda v: v["presp0"], "pdata0", flip(15)),
            ("vpu_relu_delay", lambda v: v["pw0"] and v["pwrow0"] == 7, "pwrow0", 8),
            ("vpu_relu_delay", lambda v: v["pw0"] and v["pwid0"] == 3, "pwid0", 2),
            ("vpu_binary_schedule", lambda v: v["pcmd"] and v["pop"] == 2, "pop", 1),
            ("vpu_binary_schedule", lambda v: v["pr1"] and v["prid1"] == 3, "prid1", 1),
            ("vpu_unary_delay", lambda v: v["pcmd"] and v["pop"] == 10, "pimm", flip()),
            ("vpu_unary_delay", lambda v: v["pw0"] and v["pwid0"] == 6 and v["pwrow0"] == 3, "pwdata0", flip()),
            ("vpu_reduce_schedule", lambda v: v["pw1"] and v["pwrow1"] == 4, "pwid1", flip()),
            ("vpu_reduce_schedule", lambda v: v["pw1"] and v["pwrow1"] == 9, "pwdata1", flip(8)),
            ("vpu_reduce_schedule", lambda v: v["pr0"] and v["prid0"] == 1 and v["prrow0"] == 31, "pr0", 0),
            ("vpu_vli_delay", lambda v: v["pw0"] and v["pwid0"] == 2 and v["pwrow0"] == 0, "pwdata0", 0),
            ("vpu_vli_delay", lambda v: v["pcmd"] and v["pop"] == 28, "pdst", 5),
            ("vpu_pack_schedule", lambda v: v["pcmd"] and v["pop"] == 17, "pscale", 127),
            ("vpu_pack_schedule", lambda v: v["pw0"] and v["pwid0"] == 4, "pwdata0", flip()),
            ("vpu_pack_schedule", lambda v: v["scmd"], "scmd_addr", 4),
            ("vpu_pack_schedule", lambda v: v["sresp"], "sresp_data", flip()),
            ("vpu_pack_schedule", lambda v: v["sew"], "sew_idx", 4),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_al"] and v["m0_al_row"] == 3, "m0_ald5", flip()),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_al"] and v["m0_al_row"] == 3, "m0_al_row", 4),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_al"], "m0_al_acc", flip()),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_cv"] and v["m0_cacc"], "m0_psum7", flip()),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_ar"] and v["m0_ar_row"] == 6, "m0_ar_row", 7),
            ("mxu0_acc_bf16_delay", lambda v: v["m0_cv"], "m0_cacc", 0),
            ("mxu1_acc_fp8_schedule", lambda v: v["m1_cmd"] and v["m1_op"] == 3, "m1_scale", 127),
            ("mxu1_acc_fp8_schedule", lambda v: v["m1_w0"] and v["m1_wrow0"] == 2, "m1_wdata0", flip()),
            ("mxu1_acc_fp8_schedule", lambda v: v["m1_as"], "m1_as_acc", flip()),
            ("mxu1_acc_fp8_schedule", lambda v: v["m1_r1"] and v["m1_rrow1"] == 5, "m1_rid1", flip()),
            ("mxu1_acc_fp8_schedule", lambda v: v["m1_al"] and v["m1_al_row"] == 0, "m1_ald0", flip(3)),
            ("dma_pair_in_flight", lambda v: v["dma_launch"] and v["dma_launch_channel"] == 1, "dma_launch_channel", 0),
            ("dma_pair_in_flight", lambda v: v["dma_av"] and v["dma_ar"] and v["dma_ao"] == 0, "dma_ad", flip()),
            ("dma_pair_in_flight", lambda v: v["dma_dv"] and v["dma_dr"], "dma_ds", flip()),
            ("dma_pair_in_flight", lambda v: v["dma_busy0"] and v["dma_busy1"], "dma_busy1", 0),
            ("dma_pair_in_flight", lambda v: v["dma_vw"] and v["dma_vwg"], "dma_vwa", flip())]
        for name, predicate, key, value in mutations:
            with self.subTest(name=name, key=key):
                self.mutate(name, lambda v, _: predicate(v), lambda v: v.__setitem__(key, value(v[key]) if callable(value) else value))

    def test_new_decoder_binding_mutations_are_rejected(self):
        export, events = self.cases["vpu_relu_delay"][3], self.cases["vpu_relu_delay"][2]
        relu = next(i for i in export["instructions"] if i["mnemonic"] == "vrelu.bf16")["word_index"]
        write = lambda e: next(a for a in e["instructions"][relu]["footprint"]["accesses"] if a["write"])
        for position, mutate in enumerate([lambda e: write(e).__setitem__("age", write(e)["age"] + 1), lambda e: write(e).__setitem__("first", write(e)["first"] + 32)]):
            changed = copy.deepcopy(export); mutate(changed)
            with self.subTest(export=position): self.rejects("vpu_relu_delay", export=changed)
        command = lambda o: next(c for c in o["commands"] if c["engine"] == "vpu")
        for position, mutate in enumerate([lambda o: command(o)["writes"][9].__setitem__("port", 1), lambda o: command(o)["reads"][3].__setitem__("id", 2),
                                           lambda o: command(o)["responses"][4].__setitem__("edge", command(o)["responses"][4]["edge"] + 1)]):
            changed = copy.deepcopy(events); mutate(changed)
            with self.subTest(events=position): self.rejects("vpu_relu_delay", events=changed)
        # A SELD binds to its exported VMEM read and scale write ages.
        for key in ("command_age", "write_age"):
            changed = copy.deepcopy(self.cases["vpu_pack_schedule"][2]); changed["scale_loads"][0][key] += 1
            with self.subTest(scale=key): self.rejects("vpu_pack_schedule", events=changed)
        changed = copy.deepcopy(self.cases["vpu_pack_schedule"][2]); changed["scale_loads"].clear()
        self.rejects("vpu_pack_schedule", events=changed)
        # Concurrent DMA: each completion anchors to its own channel's wait.
        events = self.cases["dma_pair_in_flight"][2]
        dmas = lambda o: [c for c in o["commands"] if c["engine"] == "dma"]
        self.assertTrue(any(events["commands"][i]["engine"] == events["commands"][j]["engine"] == "dma" for i, j in events["concurrent_commands"]))
        for position, mutate in enumerate([lambda o: dmas(o)[1].__setitem__("release_edge", dmas(o)[0]["release_edge"]),
                                           lambda o: [c.__setitem__("channel", 1 - c["channel"]) for c in dmas(o)],
                                           lambda o: dmas(o)[0]["memory"][-1].__setitem__("edge", dmas(o)[0]["release_edge"])]):
            changed = copy.deepcopy(events); mutate(changed)
            with self.subTest(dma=position): self.rejects("dma_pair_in_flight", events=changed)



if __name__ == "__main__":
    unittest.main()
