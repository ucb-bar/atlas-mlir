"""Model selection scores verified programs and keeps the safe default."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter
from itertools import permutations
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_evidence import requires_model_evidence
from atlas_model_selection import Candidate, candidate_options, evidence_option, order_neighbors, select_program


def candidate_command(argv):
    return next((arg for arg in argv if arg.startswith("--atlas-compile-candidate=")), None)


class ModelSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / "source.mlir"
        self.source.write_text("module {}")
        self.invocations = []
        self.programs = {}

    def fake_run(self, argv, *, cwd, **kwargs):
        self.invocations.append(argv)
        if Path(argv[0]).name == "atlas-emit":
            return subprocess.CompletedProcess(argv, 0, json.dumps(self.programs[cwd]), "")
        if "lookahead=4" in " ".join(argv):
            return subprocess.CompletedProcess(argv, 1, "", "allocation does not fit")
        (cwd / argv[argv.index("-o") + 1]).write_text("module {}")
        if candidate_command(argv):
            (cwd / "allocation.json").write_text('{"fits": true}')
            lookahead = next((rank for rank in (1, 2, 8) if f"lookahead={rank}" in " ".join(argv)), 0)
            self.programs[cwd] = {"instructions": [{"word_u32": lookahead, "operation": "atlas.dma",
                                                    "fields": {"direction": "store"}}],
                                  "cycles": {0: 10, 1: 20, 2: 5, 8: 5}[lookahead]}
        return subprocess.CompletedProcess(argv, 0, "", "")

    def select(self, *, name="compiled", runner=None, **kwargs):
        kwargs.setdefault("scorer", lambda program, **_: program["cycles"])
        with patch("atlas_model_selection.subprocess.run", side_effect=runner or self.fake_run):
            return select_program(self.source, self.directory / name, self.directory, **kwargs)

    def write_graph(self, cwd, preds, initial=None, *, curve=False):
        path = cwd / "order.json"
        order = tuple(json.loads(path.read_text())[0]) if path.exists() else tuple(
            range(len(preds)) if initial is None else initial)
        graph = {"preds": preds, "order": order}
        (cwd / "graph.json").write_text(json.dumps([graph]))
        return order

    def graph_run(self, preds, initial=None, *, curve=False, configure=None):
        def run(argv, *, cwd, **kwargs):
            result = self.fake_run(argv, cwd=cwd, **kwargs)
            if result.returncode or not candidate_command(argv):
                return result
            order = self.write_graph(cwd, preds, initial, curve=curve)
            return (configure(argv, cwd, order) if configure else None) or result
        return run

    def test_guarded_pack_is_threaded_into_model_preparation(self):
        receipt = self.select(max_candidates=1, guarded_pack=True)
        self.assertTrue(receipt["guarded_pack"])
        preparations = [arg for argv in self.invocations for arg in argv if arg.startswith("--atlas-compile-prepare=")]
        self.assertEqual(preparations,
                         ["--atlas-compile-prepare=spill-mxu-state=true boundary=boundary guarded-pack=true"])

    def test_standalone_preserves_guarded_pack_for_model_selection(self):
        import atlas_model_selection

        argv = ["atlas_model_selection.py", str(self.source), "--output-dir", str(self.directory / "compiled"),
                "--atlas-bin-dir", str(self.directory), "--guarded-pack"]
        result = dict(selected_index=0, selected_class="default", default_cycles=1, selected_cycles=1, elapsed_seconds=0)
        with patch.object(sys, "argv", argv), patch.object(atlas_model_selection, "select_program", return_value=result) as select:
            self.assertEqual(atlas_model_selection.main(), 0)
        self.assertTrue(select.call_args.kwargs["guarded_pack"])

    def test_driver_preserves_guarded_pack_for_model_selection(self):
        import lower_atlas_program

        argv = ["lower_atlas_program.py", str(self.source), "--output-dir", str(self.directory / "compiled"),
                "--atlas-bin-dir", str(self.directory), "--provider", "model", "--guarded-pack"]
        with patch.object(sys, "argv", argv), patch.object(lower_atlas_program, "select_program") as select:
            self.assertEqual(lower_atlas_program.main(), 0)
        self.assertTrue(select.call_args.kwargs["guarded_pack"])


    def test_invalid_search_controls_reject_before_tools(self):
        for options, diagnostic in (
            ({"provider": "builtin", "search_evaluations": 1}, "search-evaluations"),
            ({"search_evaluations": -1}, "search-evaluations"),
            ({"boundary": "inputs"}, "boundary must be boundary or none"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, diagnostic):
                self.select(**options)
        self.assertEqual(self.invocations, [])

    def test_package_import_finds_scorer_without_pythonpath(self):
        script = """from pathlib import Path
from tempfile import TemporaryDirectory
from subprocess import CompletedProcess
from unittest.mock import patch
from tools.atlas_model_selection import select_program
with TemporaryDirectory() as directory:
    root = Path(directory)
    source = root / "source.mlir"
    source.write_text("module {}")
    with patch("tools.atlas_model_selection.subprocess.run", return_value=CompletedProcess([], 0, "", "")):
        try:
            select_program(source, root / "output", root)
        except ValueError as error:
            assert str(error) == "model scoring requires --npu-model", str(error)
        else:
            raise AssertionError("missing model was accepted")
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                env=dict(os.environ, PYTHONPATH=""), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cycles_choose_winner_and_ties_keep_earlier_program(self) -> None:
        receipt = self.select(max_candidates=5)
        self.assertEqual(receipt["selected_index"], 2)
        self.assertEqual(receipt["selected_cycles"], 5)
        self.assertEqual(receipt["default_cycles"], 10)
        self.assertEqual(receipt["candidates"][3]["status"], "FAIL")
        self.assertIn("does not fit", receipt["candidates"][3]["error"])
        self.assertEqual((self.directory / "compiled/program.hex").read_text(), "00000002\n")
        self.assertTrue((self.directory / "compiled/candidates/000-default/program.json").exists())
        self.assertEqual(len(receipt["candidates"]), 5)
        compiled = [candidate_command(argv) for argv in self.invocations if candidate_command(argv)]
        self.assertEqual(len(compiled), 5)
        for option in compiled:
            self.assertIn("provider=model", option)
            self.assertIn("spill=true", option)
            self.assertIn("allocation-report=allocation.json", option)
        preparations = [arg for argv in self.invocations for arg in argv if arg.startswith("--atlas-compile-prepare=")]
        self.assertEqual(preparations, ["--atlas-compile-prepare=spill-mxu-state=true boundary=boundary"])

    def test_builtin_does_not_score_or_generate_alternatives(self) -> None:
        receipt = self.select(provider="builtin", scorer=lambda *_, **__: self.fail("builtin was scored"))
        self.assertEqual(len(receipt["candidates"]), 1)
        self.assertEqual(receipt["selected_index"], 0)
        self.assertIsNone(receipt["selected_cycles"])

    def test_duplicate_words_are_scored_once(self) -> None:
        scores = []
        receipt = self.select(profile="fifo43", max_candidates=16,
                              scorer=lambda program, **_: scores.append(program["cycles"]) or 10)
        self.assertEqual(len(scores), 3)
        self.assertEqual(receipt["selected_index"], 0)
        self.assertIn("uniform-store-windows", {row["class"] for row in receipt["candidates"]})


    def test_parallel_completion_order_does_not_change_a_tied_winner(self) -> None:
        later_finished = threading.Event()

        def reordered(argv, **kwargs):
            if "lookahead=2" in " ".join(argv):
                self.assertTrue(later_finished.wait(timeout=2))
            result = self.fake_run(argv, **kwargs)
            if candidate_command(argv) and "random-seed=0" in " ".join(argv):
                self.programs[kwargs["cwd"]] = {"instructions": [{"word_u32": 99}], "cycles": 5}
            if Path(argv[0]).name == "atlas-emit" and self.programs[kwargs["cwd"]]["instructions"][0]["word_u32"] == 99:
                later_finished.set()
            return result

        receipt = self.select(runner=reordered, max_candidates=5, workers=4)
        self.assertEqual(receipt["selected_index"], 2)


    def test_default_failure_stops_without_trying_alternatives(self) -> None:
        original = self.fake_run

        def fail_default(argv, **kwargs):
            if candidate_command(argv):
                return subprocess.CompletedProcess(argv, 1, "", "invalid default")
            return original(argv, **kwargs)

        with patch("atlas_model_selection.subprocess.run", side_effect=fail_default):
            with self.assertRaisesRegex(ValueError, "default candidate failed: invalid default"):
                select_program(self.source, self.directory / "compiled", self.directory, scorer=lambda *args: 0)
        self.assertFalse((self.directory / "compiled/program.json").exists())


    def test_candidate_choices_only_use_exposed_existing_knobs(self) -> None:
        options = candidate_options(Candidate(store_windows=(7, 3, 0), wait_placement="source"), "model", True)
        self.assertEqual(len(options), 1)
        self.assertTrue(options[0].startswith("--atlas-compile-candidate="))
        self.assertIn("provider=model", options[0])
        self.assertIn("spill=true", options[0])
        self.assertIn("allocation-report=allocation.json", options[0])
        self.assertTrue(any("store-windows='7,3,0'" in option for option in options))
        self.assertTrue(any("wait-placement=source" in option for option in options))
        with self.assertRaisesRegex(ValueError, "explicit virtual orders"):
            Candidate(lookahead=1, virtual_orders=((0,),))
        for invalid in ({"lookahead": 3}, {"lookahead": 1, "random_seed": 0},
                        {"store_windows": (-1,)}, {"wait_placement": "other"}):
            with self.assertRaises(ValueError):
                Candidate(**invalid)

    def test_order_neighbors_preserve_all_edges_and_only_move_one_node(self) -> None:
        preds = [[], [], [0], [1, 2], []]
        for order in permutations(range(5)):
            positions = {node: i for i, node in enumerate(order)}
            if any(positions[pred] >= positions[node] for node in order for pred in preds[node]):
                continue
            expected = set()
            for origin in range(5):
                for target in range(5):
                    moved = list(order)
                    moved.insert(target, moved.pop(origin))
                    rank = {node: i for i, node in enumerate(moved)}
                    if moved != list(order) and all(rank[pred] < rank[node] for node in moved for pred in preds[node]):
                        expected.add((tuple(moved), (0,)))
            actual = set(order_neighbors([{"preds": preds}, {"preds": [[]]}], (order, (0,))))
            self.assertEqual(actual, expected)

    def test_local_search_uses_selected_order_budget_and_strict_improvement(self) -> None:
        initial = (1, 0, 2, 3)
        attempted = []

        def compile_order(argv, **kwargs):
            cwd = kwargs["cwd"]
            result = self.fake_run(argv, **kwargs)
            if result.returncode or not candidate_command(argv):
                return result
            if "order-file=order.json" in " ".join(argv):
                order = tuple(json.loads((cwd / "order.json").read_text())[0])
                attempted.append(order)
                if len(attempted) == 1:
                    return subprocess.CompletedProcess(argv, 1, "", "dependence rejected")
                cycles = 4
                word = 100 + sum((node + 1) * 10 ** i for i, node in enumerate(order))
                self.programs[cwd] = {"instructions": [{"word_u32": word}], "cycles": cycles}
            else:
                order = initial if "lookahead=2" in " ".join(argv) else (0, 1, 2, 3)
            (cwd / "graph.json").write_text(json.dumps([{"preds": [[], [], [], []], "order": order}]))
            return result

        receipt = self.select(runner=compile_order, max_candidates=3, search_evaluations=13, workers=1)
        self.assertEqual(receipt["search"]["start_index"], 2)
        self.assertEqual(receipt["search"]["evaluations"], 13)
        self.assertEqual(receipt["selected_cycles"], 4)
        self.assertEqual(receipt["search"]["accepted_indices"], [4])
        local = receipt["candidates"][3:]
        self.assertEqual(local[0]["status"], "FAIL")
        self.assertEqual(local[0]["parent_index"], 2)
        self.assertEqual(local[-1]["parent_index"], 4)
        self.assertEqual(tuple(local[0]["options"]["virtual_orders"][0]),
                         next(order_neighbors([{"preds": [[], [], [], []]}], (initial,)))[0])
        for row in local:
            self.assertEqual(row["options"]["lookahead"], 0)
            self.assertIsNone(row["options"]["random_seed"])
        replay = next(argv for argv in self.invocations if "order-file=order.json" in " ".join(argv))
        option = candidate_command(replay)
        self.assertIsNotNone(option)
        self.assertIn("provider=model", option)
        self.assertIn("spill=true", option)
        self.assertIn("order-file=order.json", option)
        self.assertIn("graph-output=graph.json", option)


    def test_exhausted_tied_neighborhood_is_deterministic_across_worker_counts(self) -> None:
        receipts = []
        for workers in (1, 4):
            receipt = self.select(name=str(workers), runner=self.graph_run([[], [], []]), max_candidates=1,
                                  search_evaluations=20, workers=workers, scorer=lambda program, **_: 10)
            self.assertEqual(receipt["selected_index"], 0)
            self.assertEqual(receipt["search"]["accepted_indices"], [])
            self.assertEqual(receipt["search"]["evaluations"], 4)
            receipts.append([row["options"] for row in receipt["candidates"]])
        self.assertEqual(*receipts)


    def test_model_evidence_requires_explicit_pins(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            evidence_option(self.source)
        with self.assertRaisesRegex(ValueError, "requires"):
            evidence_option(pairs=self.source, pairs_sha256="ab")
        self.assertEqual(evidence_option(), [])

    def test_composed_evidence_requires_pinned_base_and_side(self) -> None:
        for kwargs in ({"composed": self.source}, {"composed_sha256": "cd"}):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                evidence_option(self.source, "ab", self.source, "ef", **kwargs)
        with self.assertRaisesRegex(ValueError, "requires its pinned pairs"):
            evidence_option(self.source, "ab", composed=self.source, composed_sha256="cd")

    def test_composed_selection_retains_exact_base_evidence(self) -> None:
        receipt = self.select(op_timing=self.source, op_timing_sha256="ab", pairs=self.source,
                              pairs_sha256="ef", composed=self.source, composed_sha256="cd", max_candidates=1)
        self.assertNotIn("overlap_projection", receipt)
        option = next(arg for arg in self.invocations[0] if arg.startswith("--select-atlas-model-evidence="))
        self.assertIn("pairs-sha256=ef", option)
        self.assertIn("composed-sha256=cd", option)


from test_virtual_ssa import BIN, ROOT, run
from test_virtual_scheduling import S, function, start, inp, const, dma_load, dma_store, outp

EXAMPLE = ROOT / "test/examples/virtual_bf16_shared_relu.mlir"


class GuardedModelPreparationTest(unittest.TestCase):
    def test_guarded_model_preparation_converts_pack(self):
        from test_dialect import OPT
        from test_guarded_pack import PACK, source

        consumer = ('%result = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} '
                    ': (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            facts, source_file = root / "facts.json", root / "guarded.mlir"
            fixture = subprocess.run([str(OPT.parent / "atlas-model-timing-test"), "--fixture"],
                                     capture_output=True, check=True)
            facts.write_bytes(fixture.stdout)
            source_file.write_text(source(after=consumer, value="%result"))
            evidence = evidence_option(facts, hashlib.sha256(facts.read_bytes()).hexdigest())
            for guarded in (False, True):
                flag = "--atlas-compile-prepare" + ("=guarded-pack=true" if guarded else "")
                prepared = subprocess.run([str(OPT), str(source_file), *evidence, flag],
                                          capture_output=True, text=True)
                self.assertEqual(prepared.returncode, 0, prepared.stderr)
                self.assertEqual(PACK not in prepared.stdout, guarded)
                if guarded:
                    self.assertIn('"atlas.virtual_mxu_readout_fp8"', prepared.stdout)


class ModelSelectionKnobsTest(unittest.TestCase):
    def test_candidate_controls_require_npu_model(self):
        for option, diagnostic in (
            ("--schedule-atlas-virtual=lookahead=1", "lookahead requires the model"),
            ("--schedule-atlas-virtual=order-file=missing.json", "require the model"),
            ("--schedule-atlas-virtual=graph-output=unused.json", "require the model"),
            ("--lower-atlas-virtual-to-machine=store-windows=1", "store-windows requires the model"),
            ("--schedule-atlas-stream=wait-placement=source", "wait-placement=source requires the model"),
        ):
            with self.subTest(option=option):
                source = "module {}" if "wait-placement" in option else EXAMPLE.read_text()
                result = run("atlas-opt", source, option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_invalid_candidate_controls_fail(self):
        for option, diagnostic in (
            ("--schedule-atlas-virtual=lookahead=3", "lookahead must be"),
            ("--schedule-atlas-stream=wait-placement=unknown", "wait-placement must be"),
        ):
            with self.subTest(option=option):
                source = "module {}" if "wait-placement" in option else EXAMPLE.read_text()
                result = run("atlas-opt", source, option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)

    def test_explicit_defaults_preserve_builtin_ir(self):
        source = EXAMPLE.read_text()
        for ordinary, explicit in (
            ("--schedule-atlas-virtual", "--schedule-atlas-virtual=provider=builtin lookahead=0"),
            ("--lower-atlas-virtual-to-machine", "--lower-atlas-virtual-to-machine=provider=builtin"),
        ):
            with self.subTest(ordinary=ordinary):
                before = run("atlas-opt", source, ordinary)
                after = run("atlas-opt", source, explicit)
                self.assertEqual(before.returncode, 0, before.stderr)
                self.assertEqual(after.returncode, 0, after.stderr)
                self.assertEqual(before.stdout, after.stdout)


# External evidence is optional only when neither pin was configured.
@requires_model_evidence
class TiledSelectionTest(unittest.TestCase):
    def test_mlp16_default_needs_and_receives_command_slot_waits(self):
        source = ROOT / "test/examples/virtual_fp8_mlp_16_tiles.mlir"
        facts, pairs = Path(os.environ["ATLAS_TEST_OP_TIMING"]), Path(os.environ["ATLAS_TEST_PAIR_TIMING"])
        digests = {"op_timing_sha256": hashlib.sha256(facts.read_bytes()).hexdigest(),
                   "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            receipt = select_program(source, Path(directory) / "selected", BIN, max_candidates=1, op_timing=facts,
                                     pairs=pairs, scorer=lambda program, **_: len(program["instructions"]), **digests)
        self.assertEqual((receipt["candidates"][0]["status"], receipt["selected_index"]), ("PASS", 0))


class ModelOrderReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        fixture = subprocess.run([str(BIN / "atlas-model-timing-test"), "--fixture"], capture_output=True, check=True)
        facts = self.directory / "facts.json"
        facts.write_bytes(fixture.stdout)
        self.evidence = (f"--select-atlas-model-evidence=op-timing={facts} "
                         f"op-timing-sha256={hashlib.sha256(fixture.stdout).hexdigest()} dma=wait")
        self.source = function("alias", [
            start("s0"), inp("s0", "s1", "tile", 0), const("address", -2147481600), const("size", 2048),
            dma_store("s1", "s2", "tile", "address"),
            dma_load("s2", "s3", "loaded", "address", "size", "bf16"),
            outp("s3", "s4", "loaded", 0), f"return %s4 : {S}"])
        self.graph_path = self.directory / "graph.json"
        self.order_path = self.directory / "order.json"

    def schedule(self, source=None, **options):
        option = "--schedule-atlas-virtual=provider=model " + " ".join(
            f"{key.replace('_', '-')}={value}" for key, value in options.items())
        return run("atlas-opt", source or self.source, self.evidence, option, "--verify-atlas-virtual-stream")

    def graph(self):
        result = self.schedule(graph_output=self.graph_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.graph_path.read_text()), result.stdout

    def replay(self, orders):
        self.order_path.write_text(json.dumps(orders))
        return self.schedule(order_file=self.order_path)

    def test_replay_matches_scheduler_and_repairs_state_after_independent_load_swap(self):
        graphs, scheduled = self.graph()
        replay = self.replay([graph["order"] for graph in graphs])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(replay.stdout, scheduled)
        self.source = function("loads", [start(), const("address", -2147481600), const("size", 2048),
            dma_load("io0", "s1", "a", "address", "size", "bf16"),
            dma_load("s1", "s2", "b", "address", "size", "bf16"),
            outp("s2", "s3", "a", 0), outp("s3", "s4", "b", 1), f"return %s4 : {S}"])
        graphs, _ = self.graph()
        order = graphs[0]["order"]
        loads = [i for i, name in enumerate(graphs[0]["operations"]) if name == "atlas.virtual_dma_load_bf16"]
        left, right = (order.index(node) for node in loads)
        order[left], order[right] = order[right], order[left]
        replay = self.replay([order])
        self.assertEqual(replay.returncode, 0, replay.stderr)

    def test_each_graph_edge_including_alias_order_is_mandatory(self):
        graphs, _ = self.graph()
        graph = graphs[0]
        store = graph["operations"].index("atlas.virtual_dma_store_bf16")
        load = graph["operations"].index("atlas.virtual_dma_load_bf16")
        self.assertIn(store, graph["preds"][load])
        for node, preds in enumerate(graph["preds"]):
            for pred in preds:
                with self.subTest(pred=pred, node=node):
                    order = list(graph["order"])
                    order.remove(node)
                    order.insert(order.index(pred), node)
                    rejected = self.replay([order])
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn("violates a dependence", rejected.stderr)

    def test_malformed_orders_and_conflicting_modes_reject(self):
        graphs, _ = self.graph()
        order = graphs[0]["order"]
        for bad in (None, {}, [], [order, []], [order[:-1]], [[-1] + order[1:]],
                    [[order[1]] + order[1:]], [[len(order)] + order[1:]], [[True] + order[1:]]):
            with self.subTest(order=bad):
                self.assertNotEqual(self.replay(bad).returncode, 0)
        self.order_path.write_text(json.dumps([order]))
        for option in ({"lookahead": 1}, {"random_seed": 1}):
            result = self.schedule(order_file=self.order_path, **option)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("order-file excludes", result.stderr)


if __name__ == "__main__":
    unittest.main()
