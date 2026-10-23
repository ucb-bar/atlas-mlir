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
from atlas_model_selection import (Candidate, candidate_options, evidence_option, group_order_neighbors,
                                   order_neighbors, select_program)


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


    def test_mreg_pool_is_eight_seeds_at_every_cap(self):
        vectors = [tuple(random.Random(seed).choices(range(32), k=64)) for seed in range(8)]
        for cap, evaluated in ((32, 8), (64, 8), (25, 2)):
            with self.subTest(cap=cap), patch("atlas_model_selection.subprocess.run", side_effect=self.fake_run):
                receipt = select_program(self.source, self.directory / str(cap), self.directory,
                    max_candidates=cap, mreg_candidates=None,
                    store_window_candidates=((0, 1), (0, 2), (0, 3)), scorer=lambda program, **_: program["cycles"])
            ranks = [row["options"]["mreg_ranks"] for row in receipt["candidates"] if row["class"] == "mreg-ranks"]
            self.assertEqual((receipt["mreg_pool_size"], len(ranks)), (8, evaluated))
            self.assertEqual(ranks, vectors[:evaluated])


    def test_invalid_search_controls_reject_before_tools(self):
        for options, diagnostic in (
            ({"provider": "builtin", "search_group_evaluations": 1}, "search-group-evaluations"),
            ({"search_group_evaluations": -1}, "search-group-evaluations"),
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

    def test_register_candidates_follow_existing_bounded_search(self):
        receipt = self.select(max_candidates=22, mreg_candidates=((31, 0), (7, 3)))
        self.assertEqual([row["class"] for row in receipt["candidates"]][-2:], ["mreg-ranks", "mreg-ranks"])
        self.assertEqual(receipt["candidates"][-1]["options"]["mreg_ranks"], (7, 3))
        self.assertEqual(receipt["selected_index"], 2)

    def test_local_search_starts_from_best_bank_candidate(self):
        def compile_bank(argv, **kwargs):
            result = self.fake_run(argv, **kwargs)
            cwd = kwargs["cwd"]
            if result.returncode == 0 and candidate_command(argv):
                self.write_graph(cwd, [[], [], []])
                if "mreg-ranks=" in " ".join(argv):
                    local = "order-file=" in " ".join(argv)
                    self.programs[cwd] = {"instructions": [{"word_u32": 101 if local else 100}], "cycles": 2 if local else 3}
            return result

        receipt = self.select(runner=compile_bank, max_candidates=25, mreg_candidates=((31, 7),),
                              search_evaluations=4)
        start = receipt["candidates"][receipt["search"]["start_index"]]
        self.assertEqual(start["class"], "mreg-ranks")
        self.assertEqual(receipt["selected_cycles"], 2)
        self.assertEqual(receipt["search"]["evaluations"], 4)
        for row in receipt["candidates"]:
            if row["class"] == "local-order":
                self.assertEqual(row["options"]["mreg_ranks"], (31, 7))

    def test_register_search_preserves_the_knob_prefix_after_deduplication(self):
        common = dict(scorer=lambda program, **_: program["cycles"], store_window_candidates=((0,), (0,), (1,)))
        prefix = self.select(name="prefix", max_candidates=24, **common)
        extended = self.select(name="extended", max_candidates=32, mreg_candidates=None, **common)
        self.assertEqual([row["options"] for row in prefix["candidates"]],
                         [row["options"] for row in extended["candidates"][:len(prefix["candidates"])]])
        self.assertEqual(sum(row["class"] == "mreg-ranks" for row in extended["candidates"]), 8)

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
        options = candidate_options(Candidate(store_windows=(7, 3, 0), wait_placement="source", mreg_ranks=(31, 0, 7)), "model", True)
        self.assertEqual(len(options), 1)
        self.assertTrue(options[0].startswith("--atlas-compile-candidate="))
        self.assertIn("provider=model", options[0])
        self.assertIn("spill=true", options[0])
        self.assertIn("allocation-report=allocation.json", options[0])
        self.assertTrue(any("store-windows='7,3,0'" in option for option in options))
        self.assertTrue(any("wait-placement=source" in option for option in options))
        self.assertTrue(any("mreg-ranks='31,0,7'" in option for option in options))
        with self.assertRaisesRegex(ValueError, "explicit virtual orders"):
            Candidate(lookahead=1, virtual_orders=((0,),))
        for invalid in ({"lookahead": 3}, {"lookahead": 1, "random_seed": 0},
                        {"store_windows": (-1,)}, {"mreg_ranks": (-1,)}, {"wait_placement": "other"}):
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

    def test_group_neighbors_preserve_dependencies_and_other_blocks(self) -> None:
        preds = [[], [], [0], [1, 2], []]
        for order in permutations(range(5)):
            positions = {node: i for i, node in enumerate(order)}
            if any(positions[pred] >= positions[node] for node in order for pred in preds[node]):
                continue
            neighbors = list(group_order_neighbors([{"preds": preds}, {"preds": [[]]}], (order, (0,))))
            self.assertEqual(len(neighbors), len(set(neighbors)))
            for moved, other in neighbors:
                self.assertEqual(other, (0,))
                self.assertEqual(sorted(moved), list(range(5)))
                rank = {node: i for i, node in enumerate(moved)}
                self.assertTrue(all(rank[pred] < rank[node] for node in moved for pred in preds[node]))

    def test_group_neighbors_move_crossed_predecessors_and_successors(self) -> None:
        graph = {"preds": [[], [0], [1], [], []]}
        orders = ((0, 3, 1, 4, 2),)
        neighbors = set(group_order_neighbors([graph], orders))
        self.assertIn(((0, 1, 2, 3, 4),), neighbors)
        self.assertIn(((3, 4, 0, 1, 2),), neighbors)
        self.assertNotIn(((0, 1, 2, 3, 4),), set(order_neighbors([graph], orders)))
        self.assertEqual(list(group_order_neighbors([graph], orders, max_nodes=1)), [])
        chain = {"preds": [[], [0], [1], [2], [3], [], [], [], [], []]}
        interleaved = ((5, 0, 6, 1, 7, 2, 8, 3, 9, 4),)
        whole_chain = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),)
        self.assertIn(whole_chain, set(group_order_neighbors([chain], interleaved, max_nodes=5)))
        self.assertNotIn(whole_chain, set(group_order_neighbors([chain], interleaved, max_nodes=4)))

    def test_group_search_crosses_tied_orders_after_local_search(self) -> None:
        initial, winner = (0, 3, 1, 4, 2), (0, 1, 2, 3, 4)

        def compile_order(argv, cwd, order):
            word = sum((node + 1) * 10 ** i for i, node in enumerate(order))
            self.programs[cwd] = {"instructions": [{"word_u32": word}], "cycles": 4 if order == winner else 10}

        receipts = []
        for workers in (1, 4):
            runner = self.graph_run([[], [0], [1], [], []], initial, configure=compile_order)
            receipt = self.select(name=str(workers), runner=runner, max_candidates=1, search_evaluations=100,
                                  search_group_evaluations=100, workers=workers)
            self.assertEqual(receipt["search"]["accepted_indices"], [])
            self.assertEqual(receipt["group_search"]["start_index"], 0)
            self.assertEqual(receipt["selected_cycles"], 4)
            self.assertEqual(receipt["selected_class"], "group-order")
            self.assertLessEqual(receipt["group_search"]["evaluations"], 100)
            receipts.append([(row["options"], row["status"]) for row in receipt["candidates"]])
        self.assertEqual(*receipts)

    def test_group_search_counts_rejections_and_stops_on_exhausted_ties(self) -> None:
        def compile_order(argv, cwd, order):
            if order == (1, 2, 0):
                return subprocess.CompletedProcess(argv, 1, "", "DMA command slot is still pending")

        with patch("atlas_model_selection.group_order_neighbors", return_value=iter([((1, 2, 0),), ((2, 0, 1),)])):
            runner = self.graph_run([[], [], []], configure=compile_order)
            receipt = self.select(runner=runner, max_candidates=1, search_group_evaluations=8,
                                  scorer=lambda program, **_: 10)
        self.assertEqual(receipt["selected_index"], 0)
        self.assertEqual(receipt["group_search"]["evaluations"], 2)
        self.assertEqual(receipt["group_search"]["accepted_indices"], [])
        self.assertEqual(receipt["candidates"][1]["status"], "FAIL")

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

    def test_bank_candidates_are_opt_in_and_provider_gated(self):
        receipt = self.select()
        self.assertFalse(any(row["class"] == "mreg-ranks" for row in receipt["candidates"]))
        with self.assertRaisesRegex(ValueError, "MREG candidates require"):
            self.select(name="other", provider="builtin", mreg_candidates=None)

    def test_pack_reload_candidate_requires_provider_and_strict_model_improvement(self):
        with self.assertRaisesRegex(ValueError, "pack-reload-candidates requires"):
            self.select(provider="builtin", pack_reload_candidates=True)
        for cycles in (4, 5, 6):
            def compile_reload(argv, **kwargs):
                result = self.fake_run(argv, **kwargs)
                if candidate_command(argv) and "pack-reload-after-weight=true" in " ".join(argv):
                    self.programs[kwargs["cwd"]] = {"instructions": [{"word_u32": 99}], "cycles": cycles}
                return result
            with self.subTest(cycles=cycles):
                receipt = self.select(name=str(cycles), runner=compile_reload, max_candidates=3,
                                      pack_reload_candidates=True)
            candidate = receipt["candidates"][-1]
            self.assertEqual(candidate["class"], "pack-reload-order")
            self.assertEqual(candidate["options"]["lookahead"], 2)
            self.assertTrue(candidate["options"]["pack_reload_after_weight"])
            self.assertEqual(receipt["selected_index"], 3 if cycles < 5 else 2)

    def test_invalid_pack_reload_candidate_keeps_verified_incumbent(self):
        def reject_reload(argv, **kwargs):
            if "pack-reload-after-weight=true" in " ".join(argv):
                return subprocess.CompletedProcess(argv, 1, "", "reload timing rejected")
            return self.fake_run(argv, **kwargs)
        receipt = self.select(runner=reject_reload, max_candidates=3, pack_reload_candidates=True)
        self.assertEqual(receipt["selected_index"], 2)
        self.assertEqual(receipt["candidates"][-1]["status"], "FAIL")
        self.assertIn("reload timing rejected", receipt["candidates"][-1]["error"])


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
from test_virtual_scheduling import S, T, F, function, start, inp, const, dma_load, dma_store, outp

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
            ("--lower-atlas-virtual-to-machine=mreg-ranks=1", "mreg-ranks requires the model"),
            ("--lower-atlas-virtual-to-machine=pack-reload-after-weight=true", "pack-reload-after-weight requires the model"),
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

    def test_invalid_register_ranks_fail(self):
        for ranks in ("-1", "1,,2", "4294967296", "unknown"):
            with self.subTest(ranks=ranks):
                result = run("atlas-opt", EXAMPLE.read_text(),
                             "--lower-atlas-virtual-to-machine=provider=model mreg-ranks='" + ranks + "'")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("mreg-ranks must contain", result.stderr)

    def test_explicit_defaults_preserve_builtin_ir(self):
        source = EXAMPLE.read_text()
        for ordinary, explicit in (
            ("--schedule-atlas-virtual", "--schedule-atlas-virtual=provider=builtin lookahead=0"),
            ("--lower-atlas-virtual-to-machine", "--lower-atlas-virtual-to-machine=provider=builtin pack-reload-after-weight=false"),
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

    def test_pack_reload_moves_only_the_trailing_same_block_reload(self):
        pack = f'%packed = "atlas.virtual_pack_fp8"(%tile) {{scale_code = 127 : i32}} : ({T}) -> {F}'
        matmul = f'%result = "atlas.virtual_mxu_matmul"(%packed, %weight) {{unit = 1 : i32}} : ({F}, {F}) -> {T}'
        prefix = [start(), inp("io0", "s1", "tile", 0), inp("s1", "s2", "weight", 1, "fp8"), pack]
        tail = [matmul, outp("s2", "s3", "result", 0), f"return %s3 : {S}"]
        source = function("reload", prefix + tail)
        def lower(text, enabled):
            result = run("atlas-opt", text, self.evidence, "--lower-atlas-virtual-boundary-to-dma",
                         "--verify-atlas-virtual-stream",
                         "--lower-atlas-virtual-to-machine=provider=model pack-reload-after-weight=" + str(enabled).lower(),
                         "--insert-atlas-dma-waits", "--verify-atlas-machine-stream")
            self.assertEqual(result.returncode, 0, result.stderr)
            return result.stdout
        before, after = lower(source, False), lower(source, True)
        self.assertNotEqual(before, after)
        def operations(text):
            return [re.sub(r'%[0-9]+', '%state', line.strip()) for line in text.splitlines() if '"atlas.' in line]
        self.assertEqual(Counter(operations(before)), Counter(operations(after)))
        # An emitted operation between PACK and its consumer forbids suffix motion.
        intervening = outp("s2", "s4", "tile", 1)
        blocked = function("blocked", prefix + [intervening] + [line.replace("%s2", "%s4") for line in tail])
        self.assertEqual(lower(blocked, False), lower(blocked, True))

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
