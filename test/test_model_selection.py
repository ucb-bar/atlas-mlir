"""Model selection scores verified programs and keeps the safe default."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import random
from collections import Counter
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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
from atlas_search_policy import COMPILER_POLICY_SHA256
from atlas_model_selection import (Candidate, add_model_arguments, candidate_options, evidence_option,
                                   group_order_neighbors, model_arguments, select_program)


def candidate_command(argv):
    return next((arg for arg in argv if arg.startswith("--atlas-compile-candidate=")), None)


class SyntheticNativeScorer:
    def __init__(self, *, score=None, curve_path="/tmp/test-curve.json"):
        self.score = score
        self.curve_path = curve_path

    def __call__(self, program, profile):
        return self.score(program, profile) if self.score else program["cycles"]

    def profile_receipt(self, program, profile):
        return {"profile": profile, "sha256": profile + "-configuration-digest"}

    def runtime_receipt(self):
        return {"curve_path": self.curve_path, "configuration": {}, "python": "test"}


class SelectionFixture:
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
        seed = re.search(r"\brandom-seed=(\d+)\b", " ".join(argv))
        seed = int(seed.group(1)) if seed else None
        if seed == 2:
            return subprocess.CompletedProcess(argv, 1, "", "allocation does not fit")
        (cwd / argv[argv.index("-o") + 1]).write_text("module {}")
        if candidate_command(argv):
            (cwd / "allocation.json").write_text('{"fits": true}')
            word = {0: 1, 1: 2, 3: 8}.get(seed, 0)
            self.programs[cwd] = {
                "compiler_search_policy_sha256": COMPILER_POLICY_SHA256,
                "instructions": [{"word_u32": word, "operation": "atlas.dma", "fields": {"direction": "store"}}],
                "cycles": {0: 10, 1: 20, 2: 5, 8: 5}[word]}
        return subprocess.CompletedProcess(argv, 0, "", "")

    def select(self, *, name="compiled", runner=None, **kwargs):
        kwargs.setdefault("scorer", lambda program, **_: program["cycles"])
        with patch("atlas_model_selection.subprocess.run", side_effect=runner or self.fake_run):
            return select_program(self.source, self.directory / name, self.directory, **kwargs)

    @contextmanager
    def native_scoring(self, scorer, *, evidence=(), runner=None):
        with patch("atlas_model_scorer.ModelScorer", return_value=scorer), \
                patch("atlas_model_selection.ProcessPoolExecutor", side_effect=lambda max_workers, **_: ThreadPoolExecutor(max_workers=max_workers)), \
                patch("atlas_model_selection.evidence_option", side_effect=lambda *_: list(evidence)), \
                patch("atlas_model_selection.subprocess.run", side_effect=runner or self.fake_run):
            yield

    def override_run(self, option, word, cycles, *, policy=COMPILER_POLICY_SHA256):
        def run(argv, **kwargs):
            result = self.fake_run(argv, **kwargs)
            if candidate_command(argv) and option in " ".join(argv):
                self.programs[kwargs["cwd"]] = {
                    "compiler_search_policy_sha256": policy,
                    "instructions": [{"word_u32": word}], "cycles": cycles}
            return result
        return run

    def write_graph(self, cwd, preds, initial=None):
        path = cwd / "order.json"
        order = tuple(json.loads(path.read_text())[0]) if path.exists() else tuple(
            range(len(preds)) if initial is None else initial)
        graph = {"preds": preds, "order": order}
        (cwd / "graph.json").write_text(json.dumps([graph]))
        return order

    def graph_run(self, preds, initial=None, *, configure=None):
        def run(argv, *, cwd, **kwargs):
            result = self.fake_run(argv, cwd=cwd, **kwargs)
            if result.returncode or not candidate_command(argv):
                return result
            order = self.write_graph(cwd, preds, initial)
            return (configure(argv, cwd, order) if configure else None) or result
        return run

class ModelSelectionTest(SelectionFixture, unittest.TestCase):
    def test_explicit_overrides(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        for budget, expected in (("full", (32, 204, 96)), ("selection", (24, 0, 0))):
            values = model_arguments(parser.parse_args(["--provider", "model", "--search-budget", budget]))
            self.assertEqual(tuple(values[key] for key in
                                   ("max_candidates", "search_group_evaluations", "search_patience")), expected)
            self.assertEqual(values["profile"], "curve")
            self.assertEqual(values["mreg_candidates"], None if budget == "full" else ())
        values = model_arguments(parser.parse_args(["--provider", "model", "--workers", "4",
            "--search-group-evaluations", "72", "--profile", "fifo43", "--no-bank-candidates",
            "--no-pack-reload-candidates", "--no-pack-preload-candidates"]))
        self.assertEqual((values["search_group_evaluations"], values["profile"], values["mreg_candidates"]), (72, "fifo43", ()))
        self.assertFalse(values["pack_reload_candidates"] or values["pack_preload_candidates"])
        builtin = model_arguments(parser.parse_args([]))
        self.assertEqual((builtin["profile"], builtin["search_group_evaluations"],
                          builtin["mreg_candidates"]), ("fifo43", 204, None))

    def test_guarded_pack_reaches_preparation_and_entry_points(self):
        import atlas_model_selection
        import lower_atlas_program
        receipt = self.select(max_candidates=1, guarded_pack=True)
        self.assertTrue(receipt["guarded_pack"])
        self.assertEqual([arg for argv in self.invocations for arg in argv
                          if arg.startswith("--atlas-compile-prepare=")],
                         ["--atlas-compile-prepare=spill-mxu-state=true boundary=boundary guarded-pack=true"])
        result = dict(selected_index=0, selected_class="default", default_cycles=1,
                      selected_cycles=1, elapsed_seconds=0)
        for module in (atlas_model_selection, lower_atlas_program):
            argv = [module.__name__, str(self.source), "--output-dir", str(self.directory / "entry"),
                    "--atlas-bin-dir", str(self.directory), "--provider", "model", "--guarded-pack"]
            with self.subTest(entry=module.__name__), patch.object(sys, "argv", argv), \
                    patch.object(module, "select_program", return_value=result) as select:
                self.assertEqual(module.main(), 0)
                self.assertTrue(select.call_args.kwargs["guarded_pack"])

    def test_candidate_options_and_validation(self):
        self.assertIn("lookahead=4", candidate_options(Candidate(lookahead=4), "model", True)[0])
        self.assertEqual(candidate_options(Candidate(store_windows=(7, 3, 0), mreg_ranks=(31, 0, 7),
                                                      wait_placement="source"), "model", True),
                         ["--atlas-compile-candidate=provider=model spill=true allocation-report=allocation.json "
                          "pack-relayout-rows=2 wait-reused-dma-slots=true "
                          "store-windows='7,3,0' mreg-ranks='31,0,7' wait-placement=source"])
        self.assertNotIn("wait-placement", candidate_options(Candidate(), "model", True)[0])
        for invalid in ({"lookahead": 3}, {"lookahead": 1, "random_seed": 0},
                        {"lookahead": 1, "virtual_orders": ((0,),)}, {"store_windows": (-1,)},
                        {"mreg_ranks": (-1,)}, {"wait_placement": "other"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                Candidate(**invalid)

    def test_initial_candidates_use_only_the_seed_pool(self):
        receipt = self.select(max_candidates=9)
        rows = receipt["candidates"]
        self.assertEqual([row["class"] for row in rows], ["default"] + ["random-seed"] * 8)
        self.assertEqual([row["options"]["random_seed"] for row in rows[1:]], list(range(8)))
        self.assertTrue(all(row["options"]["lookahead"] == 0 for row in rows))
        self.assertEqual(rows[3]["status"], "FAIL")

    def test_seeded_register_pool_preserves_store_prefix_and_budget(self):
        vectors = [tuple(random.Random(seed).choices(range(32), k=64)) for seed in range(8)]
        receipts = []
        for cap, evaluated in ((22, 2), (32, 8)):
            receipt = self.select(name=str(cap), max_candidates=cap, mreg_candidates=None,
                                  store_window_candidates=((0, 1), (0, 2), (0, 3)))
            ranks = [row["options"]["mreg_ranks"] for row in receipt["candidates"] if row["class"] == "mreg-ranks"]
            self.assertEqual((receipt["mreg_pool_size"], ranks), (8, vectors[:evaluated]))
            self.assertTrue({"explicit-store-windows", "uniform-store-windows"} <=
                            {row["class"] for row in receipt["candidates"]})
            receipts.append(receipt["candidates"])
        self.assertEqual([row["options"] for row in receipts[0]],
                         [row["options"] for row in receipts[1][:len(receipts[0])]])
        self.assertFalse(any(row["class"] == "mreg-ranks" for row in self.select()["candidates"]))
        builtin = self.select(name="builtin-registers", provider="builtin", max_candidates=32,
                              mreg_candidates=None, store_window_candidates=((0, 1), (0, 2), (0, 3)))
        self.assertEqual([row["options"]["mreg_ranks"] for row in builtin["candidates"]
                          if row["class"] == "mreg-ranks"], vectors)

    def test_register_search_preserves_the_knob_prefix_after_deduplication(self):
        common = dict(scorer=lambda program, **_: program["cycles"], store_window_candidates=((0,), (0,), (1,)))
        prefix = self.select(name="prefix", max_candidates=24, **common)
        extended = self.select(name="extended", max_candidates=32, mreg_candidates=None, **common)
        self.assertEqual([row["options"] for row in prefix["candidates"]],
                         [row["options"] for row in extended["candidates"][:len(prefix["candidates"])]])
        self.assertEqual(sum(row["class"] == "mreg-ranks" for row in extended["candidates"]), 8)

    def test_programming_errors_during_candidate_scoring_propagate(self) -> None:
        for index, error in enumerate((TypeError("scorer bug"), KeyError("scorer bug"))):
            def scorer(program, **_):
                raise error
            with self.subTest(error=error), self.assertRaises(type(error)) as raised:
                self.select(name="unexpected-" + str(index), max_candidates=1, scorer=scorer)
            self.assertIs(raised.exception, error)

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
        self.assertEqual(list(group_order_neighbors([graph], orders, max_nodes=1)), [])
        chain = {"preds": [[], [0], [1], [2], [3], [], [], [], [], []]}
        interleaved = ((5, 0, 6, 1, 7, 2, 8, 3, 9, 4),)
        whole_chain = ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9),)
        self.assertIn(whole_chain, set(group_order_neighbors([chain], interleaved, max_nodes=5)))
        self.assertNotIn(whole_chain, set(group_order_neighbors([chain], interleaved, max_nodes=4)))


    def test_patience_stops_at_batch_boundary_and_still_checks_pack_alternatives(self):
        receipt = self.select(runner=self.graph_run([[], [0], *[[] for _ in range(14)]]), max_candidates=1,
                              search_group_evaluations=100, search_patience=12, workers=4,
                              pack_reload_candidates=True, pack_preload_candidates=True)
        self.assertEqual(receipt["group_search"]["evaluations"], 12)
        self.assertTrue(receipt["group_search"]["stopped_for_patience"])
        self.assertEqual(receipt["group_search"]["accepted_indices"], [])
        self.assertEqual([row["class"] for row in receipt["candidates"]][-2:],
                         ["pack-reload-order", "pack-preload-waits"])
        self.assertEqual(receipt["selected_index"], 0)
        self.assertEqual(receipt["score_cache_entries"], 1)
        self.assertEqual(receipt["score_cache_hits"], 14)

    def test_invalid_search_controls_reject_before_tools(self):
        for options, diagnostic in (
            ({"search_patience": -1}, "search-patience"),
            ({"search_group_evaluations": -1}, "search-group-evaluations"),
            ({"boundary": "inputs"}, "boundary must be boundary or none"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, diagnostic):
                self.select(**options)
        self.assertEqual(self.invocations, [])

    def test_builtin_scores_and_selects_alternatives(self) -> None:
        receipt = self.select(provider="builtin", max_candidates=5)
        self.assertEqual(len(receipt["candidates"]), 5)
        self.assertEqual((receipt["selected_index"], receipt["selected_cycles"]), (2, 5))
        self.assertEqual(receipt["profile"], "fifo43")
        self.assert_objective(receipt, scoring="injected-scorer")

    def test_builtin_accepts_symlinked_source_and_binary_directory(self):
        linked_source = self.directory / "linked.mlir"
        linked_source.symlink_to(self.source)
        linked_binaries = self.directory / "linked-bin"
        linked_binaries.symlink_to(self.directory, target_is_directory=True)
        with patch("atlas_model_selection.subprocess.run", side_effect=self.fake_run):
            receipt = select_program(linked_source, self.directory / "builtin-linked", linked_binaries,
                                     provider="builtin", max_candidates=1, scorer=lambda program, **_: program["cycles"])
        self.assertEqual(receipt["selected_index"], 0)
        receipt = self.select(name="model-linked", provider="model", max_candidates=1)
        self.assertEqual(receipt["selected_index"], 0)

    def assert_objective(self, receipt, *, scoring, configuration=None):
        objective = receipt["objective"]
        expected = {"schema": "atlas.model_selection_objective.v1",
                    "policy": "selected-profile-cycles-then-index-v1",
                    "scoring": scoring, "profile": receipt["profile"],
                    "minimize": ["cycles", "index"],
                    "tie_rule": "canonical-candidate-index-v1",
                    "scoring_configuration_sha256": configuration}
        self.assertEqual({key: value for key, value in objective.items() if key != "sha256"}, expected)
        digest = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(objective["sha256"], digest)
        self.assertNotIn("fifo43_tiebreak_cache_entries", receipt)

    def test_curve_exact_ties_use_index_cache_words_and_copy_the_winner(self):
        calls = Counter()

        def scorer(program, profile):
            self.assertEqual(profile, "curve")
            word = program["instructions"][0]["word_u32"]
            calls[profile, word] += 1
            return program["cycles"]

        receipt = self.select(runner=self.override_run("random-seed=4", 8, 5), profile="curve",
                              max_candidates=16, workers=4, scorer=scorer)
        self.assertEqual((receipt["selected_index"], receipt["selected_cycles"]), (2, 5))
        self.assertEqual(receipt["candidates"][3]["status"], "FAIL")
        self.assertEqual(calls, Counter({("curve", word): 1 for word in (0, 1, 2, 8)}))
        self.assertEqual(receipt["score_cache_entries"], 4)
        self.assertGreater(receipt["score_cache_hits"], 0)
        self.assertFalse(any("fifo43_cycles" in row or "fifo43_score_cache_hit" in row
                             for row in receipt["candidates"]))
        output = self.directory / "compiled"
        winner = output / receipt["candidates"][2]["directory"]
        for artifact in ("program.json", "program.hex", "final.mlir", "allocation.json"):
            self.assertEqual((output / artifact).read_bytes(), (winner / artifact).read_bytes())
        self.assertEqual((output / "program.hex").read_text(), "00000002\n")
        self.assertEqual(json.loads((output / "selection.json").read_text()), json.loads(json.dumps(receipt)))
        self.assert_objective(receipt, scoring="injected-scorer")

    def test_native_profile_receipts_bind_only_the_selected_profile(self):
        for provider, profile, cap, expected in (("model", "curve", 3, {"curve"}), ("model", "curve", 5, {"curve"}),
                                               ("model", "fifo43", 5, {"fifo43"}), ("builtin", "curve", 5, {"curve"}),
                                               ("builtin", "fifo43", 5, {"fifo43"})):
            with self.subTest(provider=provider, profile=profile, cap=cap), \
                    self.native_scoring(SyntheticNativeScorer(), runner=self.override_run("random-seed=4", 8, 5),
                        evidence=("--select-atlas-model-evidence=facts=synthetic-test",)):
                output = self.directory / f"native-{provider}-{profile}-{cap}"
                receipt = select_program(self.source, output, self.directory, model=self.directory,
                                         provider=provider, profile=profile, model_facts=self.directory,
                                         max_candidates=cap, workers=2)
            self.assert_objective(receipt, scoring="native-model", configuration=profile + "-configuration-digest")
            self.assertEqual(set(receipt["scoring_configurations"]), expected)
            self.assertEqual(json.loads((output / "selection.json").read_text())["scoring_configurations"],
                             {name: {"profile": name, "sha256": name + "-configuration-digest"} for name in expected})
        self.assertNotIn("scoring_configurations", self.select(name="custom", max_candidates=1))


    def test_parallel_completion_order_does_not_change_a_tied_winner(self) -> None:
        later_finished = threading.Event()
        runner = self.override_run("random-seed=3", 99, 5)

        def reordered(argv, **kwargs):
            if "random-seed=1" in " ".join(argv):
                self.assertTrue(later_finished.wait(timeout=2))
            result = runner(argv, **kwargs)
            if Path(argv[0]).name == "atlas-emit" and self.programs[kwargs["cwd"]]["instructions"][0]["word_u32"] == 99:
                later_finished.set()
            return result

        receipt = self.select(runner=reordered, max_candidates=5, workers=4)
        self.assertEqual(receipt["selected_index"], 2)

    def refuse(self, refused):
        def run(argv, **kwargs):
            if (option := candidate_command(argv)) and refused(option):
                self.invocations.append(argv)
                return subprocess.CompletedProcess(argv, 1, "", "error: model DMA command slot is still pending")
            return self.fake_run(argv, **kwargs)
        return run

    def test_default_compile_refusal_continues_and_records_failure(self) -> None:
        default = candidate_options(Candidate(), "model", True)[0]
        receipt = self.select(runner=self.refuse(lambda option: option == default), max_candidates=5)
        failed = receipt["candidates"][0]
        self.assertEqual((failed["status"], failed["error_stage"]), ("FAIL", "compile"))
        self.assertIsNone(receipt["default_cycles"])
        self.assertEqual((receipt["selected_index"], receipt["selected_cycles"]), (2, 5))
        self.assertTrue((self.directory / "compiled/program.json").exists())

    def test_default_score_failure_remains_fatal(self) -> None:
        def scorer(program, **_):
            if program["cycles"] == fatal:
                raise RuntimeError("bank conflict")
            return program["cycles"]

        for fatal, index in ((10, 0), (20, 1)):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, f"candidate {index} failed model scoring: bank conflict"):
                    self.select(name=str(index), max_candidates=5, scorer=scorer)
            self.assertFalse((self.directory / str(index) / "program.json").exists())
            receipt = json.loads((self.directory / str(index) / "selection.json").read_text())
            self.assertEqual(receipt["candidates"][index]["error_stage"], "score")

    def test_all_refusals_raise_no_legal_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "no legal candidate"):
            self.select(runner=self.refuse(lambda option: True), max_candidates=5)
        self.assertFalse((self.directory / "compiled/program.json").exists())
        receipt = json.loads((self.directory / "compiled/selection.json").read_text())
        self.assertEqual({row["status"] for row in receipt["candidates"]}, {"FAIL"})

    def test_builtin_refusals_count_toward_the_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "no legal candidate"):
            self.select(runner=self.refuse(lambda option: True), provider="builtin", max_candidates=5)
        self.assertEqual(sum(bool(candidate_command(argv)) for argv in self.invocations), 5)

    def test_group_search_crosses_tied_orders(self) -> None:
        initial, winner = (0, 3, 1, 4, 2), (0, 1, 2, 3, 4)

        def compile_order(argv, cwd, order):
            word = sum((node + 1) * 10 ** i for i, node in enumerate(order))
            self.programs[cwd] = {"compiler_search_policy_sha256": COMPILER_POLICY_SHA256, "instructions": [{"word_u32": word}], "cycles": 4 if order == winner else 10}

        receipts = []
        for workers in (1, 4):
            runner = self.graph_run([[], [0], [1], [], []], initial, configure=compile_order)
            receipt = self.select(name=str(workers), runner=runner, max_candidates=1,
                                  search_group_evaluations=100, workers=workers)
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
        self.assertEqual(receipt["group_search"]["evaluations"], 4)
        self.assertEqual(receipt["group_search"]["accepted_indices"], [])
        self.assertEqual(receipt["candidates"][1]["status"], "FAIL")

    def test_group_wait_siblings_share_the_native_budget(self):
        orders = [((1, 2, 0),), ((2, 0, 1),)]

        def compile_order(argv, cwd, order):
            if order == (1, 2, 0):
                return subprocess.CompletedProcess(argv, 1, "", "DMA command slot is still pending")

        for profile, placements, statuses in (
                ("curve", ["default", "source", "default"], ["FAIL", "FAIL", "PASS"]),
                ("fifo43", ["default", "default"], ["FAIL", "PASS"])):
            with self.subTest(profile=profile), patch(
                    "atlas_model_selection.group_order_neighbors", return_value=iter(orders)):
                receipt = self.select(name=profile, profile=profile, max_candidates=1,
                                      search_group_evaluations=3, scorer=lambda *args, **kwargs: 10,
                                      runner=self.graph_run([[], [], []], configure=compile_order))
            rows = receipt["candidates"][1:]
            self.assertEqual([row["options"]["wait_placement"] for row in rows], placements)
            self.assertEqual([row["status"] for row in rows], statuses)
            self.assertEqual(receipt["group_search"]["evaluations"], len(rows))
            self.assertEqual(receipt["selected_index"], 0)
            self.assertTrue(all(row["options"]["lookahead"] == 0 for row in receipt["candidates"]))

    def test_curve_group_siblings_clear_weight_preloading(self):
        from atlas_model_selection import Selection
        orders = [((1, 0),)]
        base = Candidate(virtual_orders=((0, 1),), wait_placement="preloaded", pack_weight_before_pack=True)
        curve = list(Selection.group_candidates(base, orders, "curve"))
        self.assertEqual([candidate.wait_placement for candidate in curve], ["default", "source"])
        self.assertTrue(all(not candidate.pack_weight_before_pack for candidate in curve))
        fifo = list(Selection.group_candidates(base, orders, "fifo43"))
        self.assertEqual(len(fifo), 1)
        self.assertEqual(fifo[0].wait_placement, "preloaded")
        self.assertTrue(fifo[0].pack_weight_before_pack)

    def test_pack_reload_candidate_requires_strict_improvement(self):
        for cycles in (4, 5, 6):
            with self.subTest(cycles=cycles):
                receipt = self.select(name=str(cycles), runner=self.override_run("pack-reload-after-weight=true", 99, cycles), max_candidates=3,
                                      provider="builtin", pack_reload_candidates=True)
            candidate = receipt["candidates"][-1]
            self.assertEqual(candidate["class"], "pack-reload-order")
            self.assertEqual(candidate["options"]["random_seed"], 1)
            self.assertTrue(candidate["options"]["pack_reload_after_weight"])
            self.assertEqual(receipt["selected_index"], 3 if cycles < 5 else 2)

    def test_preload_candidates_replace_reload_and_compare_wait_placements(self):
        for options in ({"pack_reload_after_weight": True, "pack_weight_before_pack": True},
                        {"wait_placement": "preloaded"}):
            with self.assertRaises(ValueError):
                Candidate(**options)
        def compile_placement(argv, **kwargs):
            result = self.fake_run(argv, **kwargs)
            if candidate_command(argv):
                flags = " ".join(argv)
                cycles = (2 if "wait-placement=preloaded" in flags else
                          3 if "pack-weight-before-pack=true" in flags else
                          4 if "pack-reload-after-weight=true" in flags else None)
                if cycles is not None:
                    self.programs[kwargs["cwd"]] = {"instructions": [{"word_u32": 100 + cycles}], "cycles": cycles}
            return result
        receipt = self.select(runner=compile_placement, provider="builtin", max_candidates=3, pack_reload_candidates=True,
                              pack_preload_candidates=True)
        self.assertEqual(receipt["selected_index"], 4)
        self.assertEqual(receipt["selected_cycles"], 2)
        self.assertEqual(receipt["candidates"][-1]["class"], "pack-preload-waits")
        for row in receipt["candidates"][-1:]:
            self.assertEqual(row["options"]["random_seed"], 1)
            self.assertFalse(row["options"]["pack_reload_after_weight"])
            self.assertTrue(row["options"]["pack_weight_before_pack"])
        self.assertEqual(receipt["candidates"][-1]["options"]["wait_placement"], "preloaded")

    def bundle(self):
        bundle = self.directory / "facts"
        bundle.mkdir(exist_ok=True)
        for name in ("facts.lock.json", "model-timing.json"):
            (bundle / name).write_text("{}")
        return bundle

    def test_model_evidence_requires_complete_bundle(self) -> None:
        with self.assertRaisesRegex(ValueError, "bundle directory"):
            evidence_option(self.source)
        self.assertEqual(evidence_option(), [])
        bundle = self.bundle()
        option = evidence_option(bundle)[0]
        self.assertIn("facts=", option)
        self.assertNotIn("sha256", option)
        (bundle / "model-timing.json").unlink()
        with self.assertRaisesRegex(ValueError, "model-timing.json"):
            evidence_option(bundle)

    def test_model_selection_regenerates_bundle_by_default(self) -> None:
        def run(argv, **kwargs):
            if len(argv) > 2 and argv[1] == "-B" and "atlas_model_facts.py" in argv[2]:
                bundle = Path(argv[argv.index("--out") + 1]); bundle.mkdir()
                for name in ("facts.lock.json", "model-timing.json"):
                    (bundle / name).write_text("{}")
                self.invocations.append(argv)
                return subprocess.CompletedProcess(argv, 0, "", "")
            return self.fake_run(argv, **kwargs)
        self.select(model=self.directory, runner=run, max_candidates=1)
        self.assertEqual(self.invocations[0][2], str(ROOT / "tools/atlas_model_facts.py"))
        self.assertIn("facts=", self.invocations[1][2])


from test_virtual_ssa import BIN, ROOT, run
from test_virtual_scheduling import S, T, F, function, start, inp, const, dma_load, dma_store, outp

EXAMPLE = ROOT / "test/examples/virtual_bf16_shared_relu.mlir"


@requires_model_evidence
class TiledSelectionTest(unittest.TestCase):
    def test_one_candidate_real_emit_and_policy_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selected"
            receipt = select_program(ROOT / "test/examples/virtual_fp8_mlp_16_tiles.mlir", output, BIN,
                                     max_candidates=1, model_facts=Path(os.environ["ATLAS_TEST_MODEL_FACTS"]),
                                     scorer=lambda program, **_: len(program["instructions"]))
            self.assertEqual((receipt["candidates"][0]["status"], receipt["selected_index"]), ("PASS", 0))
            self.assertEqual(json.loads((output / "program.json").read_text())["compiler_search_policy_sha256"],
                             COMPILER_POLICY_SHA256)


@requires_model_evidence
class ModelFactsBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        original = Path(os.environ["ATLAS_TEST_MODEL_FACTS"])
        for name in ("model-timing.json", "facts.lock.json"):
            (self.bundle / name).write_bytes((original / name).read_bytes())
        self.lock = json.loads((self.bundle / "facts.lock.json").read_text())

    def write_lock(self):
        (self.bundle / "facts.lock.json").write_text(json.dumps(self.lock))

    def select(self):
        return run("atlas-opt", "module {}", *evidence_option(self.bundle))

    def test_bundle_selects_tables(self):
        selected = self.select()
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIn(self.lock["op_timing_sha256"], selected.stdout)

    def test_unqualified_or_mismatched_lock_refuses(self):
        for key, value in (("op_timing_sha256", "0" * 64), ("op_timing_sha256", None)):
            old = dict(self.lock)
            if value is None:
                self.lock.pop(key)
            else:
                self.lock[key] = value
            self.write_lock()
            selected = self.select()
            self.assertNotEqual(selected.returncode, 0, selected.stdout)
            self.lock = old

    def test_changed_timing_file_refuses(self):
        for name in ("model-timing.json",):
            path = self.bundle / name
            original = path.read_bytes()
            path.write_bytes(original + b" ")
            selected = self.select()
            self.assertNotEqual(selected.returncode, 0, selected.stdout)
            self.assertIn("hash", selected.stderr)
            path.write_bytes(original)


@requires_model_evidence
class GuardedModelPreparationTest(unittest.TestCase):
    def test_guarded_model_preparation_converts_pack(self):
        from test_dialect import OPT
        from test_guarded_pack import PACK, source

        consumer = ('%result = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} '
                    ': (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "guarded.mlir"
            source_file.write_text(source(after=consumer, value="%result"))
            evidence = evidence_option(Path(os.environ["ATLAS_TEST_MODEL_FACTS"]))
            for guarded in (False, True):
                flag = "--atlas-compile-prepare" + ("=guarded-pack=true" if guarded else "")
                prepared = subprocess.run([str(OPT), str(source_file), *evidence, flag],
                                          capture_output=True, text=True)
                self.assertEqual(prepared.returncode, 0, prepared.stderr)
                self.assertEqual(PACK not in prepared.stdout, guarded)
                if guarded:
                    self.assertIn('"atlas.virtual_mxu_readout_fp8"', prepared.stdout)


class ModelSelectionKnobsTest(unittest.TestCase):
    def test_candidate_controls_accept_builtin(self):
        from test_delay_insertion import nop, program

        for option in (
            "--schedule-atlas-virtual=provider=builtin lookahead=1",
            "--lower-atlas-virtual-to-machine=provider=builtin store-windows=1",
            "--lower-atlas-virtual-to-machine=provider=builtin mreg-ranks=1",
            "--lower-atlas-virtual-to-machine=provider=builtin pack-reload-after-weight=true",
            "--lower-atlas-virtual-to-machine=provider=builtin pack-weight-before-pack=true",
            "--schedule-atlas-stream=provider=builtin wait-placement=source",
            "--schedule-atlas-stream=provider=builtin wait-placement=preloaded",
        ):
            with self.subTest(option=option):
                source = program([nop()]) if "wait-placement" in option else EXAMPLE.read_text()
                if "wait-placement=preloaded" in option:
                    source = source.replace("module {", "module attributes {atlas.model_preloaded_weights = array<i32>} {")
                result = run("atlas-opt", source, option)
                self.assertEqual(result.returncode, 0, result.stderr)


    def test_missing_order_file_refuses(self):
        result = run("atlas-opt", EXAMPLE.read_text(), "--schedule-atlas-virtual=provider=builtin order-file=missing.json")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot read order-file", result.stderr)

    def test_invalid_pack_rows_fail(self):
        for option in ("--schedule-atlas-virtual=pack-relayout-rows=3",
                       "--lower-atlas-virtual-to-machine=pack-relayout-rows=3"):
            with self.subTest(option=option):
                result = run("atlas-opt", EXAMPLE.read_text(), option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("pack-relayout-rows must be", result.stderr)

    def test_invalid_candidate_controls_fail(self):
        for option, diagnostic in (
            ("--schedule-atlas-virtual=lookahead=3", "lookahead must be"),
            ("--schedule-atlas-stream=wait-placement=unknown", "wait-placement must be"),
            ("--lower-atlas-virtual-to-machine=pack-reload-after-weight=true pack-weight-before-pack=true", "mutually exclusive"),
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
            ("--lower-atlas-virtual-to-machine", "--lower-atlas-virtual-to-machine=provider=builtin "
             "pack-reload-after-weight=false pack-weight-before-pack=false"),
        ):
            with self.subTest(ordinary=ordinary):
                before = run("atlas-opt", source, ordinary)
                after = run("atlas-opt", source, explicit)
                self.assertEqual(before.returncode, 0, before.stderr)
                self.assertEqual(after.returncode, 0, after.stderr)
                self.assertEqual(before.stdout, after.stdout)


@requires_model_evidence
class ModelOrderReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        from test_dma_ring_waits import fixture_evidence
        self.evidence = fixture_evidence(self.directory)
        self.source = function("alias", [
            start("s0"), inp("s0", "s1", "tile", 0), const("address", -2147481600), const("size", 2048),
            dma_store("s1", "s2", "tile", "address"),
            dma_load("s2", "s3", "loaded", "address", "size", "bf16"),
            outp("s3", "s4", "loaded", 0), f"return %s4 : {S}"])
        self.graph_path = self.directory / "graph.json"
        self.order_path = self.directory / "order.json"

    def schedule(self, source=None, provider="model", **options):
        option = f"--schedule-atlas-virtual=provider={provider} " + " ".join(
            f"{key.replace('_', '-')}={value}" for key, value in options.items())
        return run("atlas-opt", source or self.source, self.evidence, option, "--verify-atlas-virtual-stream")

    def test_builtin_graph_metadata_preserves_the_selected_schedule(self):
        ordinary = self.schedule(provider="builtin")
        exported = self.schedule(provider="builtin", graph_output=self.graph_path)
        self.assertEqual(ordinary.returncode, 0, ordinary.stderr)
        self.assertEqual(exported.returncode, 0, exported.stderr)
        self.assertEqual(ordinary.stdout, exported.stdout)
        graphs = json.loads(self.graph_path.read_text())
        self.assertTrue(graphs)
        self.assertEqual(set(graphs[0]), {"preds", "operations", "order"})
        self.order_path.write_text(json.dumps([graph["order"] for graph in graphs]))
        replay = self.schedule(provider="builtin", order_file=self.order_path)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(ordinary.stdout, replay.stdout)

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

    def test_preloaded_wait_ids_are_checked_and_empty_selection_preserves_default(self):
        source = function("copy", [start(), inp("io0", "s1", "tile", 0),
                                  outp("s1", "s2", "tile", 0), f"return %s2 : {S}"])
        lowered = run("atlas-opt", source, self.evidence, "--lower-atlas-virtual-boundary-to-dma",
                      "--lower-atlas-virtual-to-machine", "--insert-atlas-dma-waits")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        transfer = re.findall(r'"atlas.dma_wait"[^\n]*atlas.virtual_dma_transfer = (\d+)', lowered.stdout)[0]
        def annotated(value):
            return lowered.stdout if value is None else lowered.stdout.replace(
                "module attributes {", "module attributes {atlas.model_preloaded_weights = " + value + ", ", 1)
        for value, diagnostic in ((None, "requires preloaded weight transfer IDs"),
                                  ("42 : i32", "requires preloaded weight transfer IDs"),
                                  ("array<i32: -1>", "unique and nonnegative"),
                                  (f"array<i32: {transfer}, {transfer}>", "unique and nonnegative"),
                                  ("array<i32: 999>", "has no DMA wait")):
            with self.subTest(value=value):
                rejected = run("atlas-opt", annotated(value),
                               "--schedule-atlas-stream=provider=model wait-placement=preloaded")
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn(diagnostic, rejected.stderr)
        empty = annotated("array<i32>")
        default = run("atlas-opt", empty, "--schedule-atlas-stream=provider=model")
        selected = run("atlas-opt", empty, "--schedule-atlas-stream=provider=model wait-placement=preloaded")
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(default.stdout, selected.stdout)
        selected = run("atlas-opt", annotated(f"array<i32: {transfer}>"),
                       "--schedule-atlas-stream=provider=model wait-placement=preloaded",
                       "--verify-atlas-machine-stream", "--verify-atlas-timing", "--verify-atlas-generated-schedule")
        self.assertEqual(selected.returncode, 0, selected.stderr)

    def graph(self):
        result = self.schedule(graph_output=self.graph_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.graph_path.read_text()), result.stdout

    def replay(self, orders):
        self.order_path.write_text(json.dumps(orders))
        return self.schedule(order_file=self.order_path)


    def test_replay_matches_scheduler_and_repairs_state_after_independent_load_swap(self):
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
