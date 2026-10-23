"""Model selection scores verified programs and keeps the safe default."""

from __future__ import annotations

import argparse
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
from atlas_model_selection import (Candidate, add_model_arguments, candidate_options, evidence_option,
                                   group_order_neighbors, model_arguments, order_neighbors, select_program,
                                   order_candidates, proposal_estimate, rank_order_neighbors, validate_proposal_graphs)


def candidate_command(argv):
    return next((arg for arg in argv if arg.startswith("--atlas-compile-candidate=")), None)


PROPOSAL_ID = "a" * 64


def proposal_graph(preds, order, latency=100):
    costs = [{"unit": 1, "busy": latency, "latency": latency, "frontend": 1,
              "queued_dma": True, "drains": False}]
    costs += [{"unit": 0, "busy": 20 if node == 2 else 1, "latency": 20 if node == 2 else 1,
               "frontend": 20 if node == 2 else 1, "queued_dma": False, "drains": False}
              for node in range(1, len(preds))]
    graph = {"preds": preds, "order": list(order), "operations": ["test.work"] * len(preds),
             "proposal": {"schema": "atlas.virtual_proposal.v1", "facts_sha256": PROPOSAL_ID,
                          "units": 2, "dma_capacity": 8, "costs": costs, "estimate": {}}}
    makespan, waiting = proposal_estimate(graph, tuple(order))
    graph["proposal"]["estimate"] = {"makespan": makespan, "exposed_waiting": waiting}
    return graph


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
        kwargs.setdefault("curve_candidates", kwargs.get("profile") == "curve")
        kwargs.setdefault("scorer", lambda program, **_: program["cycles"])
        with patch("atlas_model_selection.subprocess.run", side_effect=runner or self.fake_run):
            return select_program(self.source, self.directory / name, self.directory, **kwargs)

    def write_graph(self, cwd, preds, initial=None, *, curve=False):
        path = cwd / "order.json"
        order = tuple(json.loads(path.read_text())[0]) if path.exists() else tuple(
            range(len(preds)) if initial is None else initial)
        graph = proposal_graph(preds, order) if curve else {"preds": preds, "order": order}
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

    def test_search_presets_and_explicit_overrides(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        full = model_arguments(parser.parse_args(["--provider", "model", "--workers", "36"]))
        self.assertEqual((full["max_candidates"], full["search_evaluations"], full["search_group_evaluations"],
                          full["search_patience"]), (32, 96, 204, 96))
        self.assertIsNone(full["mreg_candidates"])
        self.assertEqual(full["profile"], "curve")
        selection = model_arguments(parser.parse_args(["--provider", "model", "--search-budget", "selection"]))
        self.assertEqual((selection["max_candidates"], selection["search_evaluations"],
                          selection["search_group_evaluations"], selection["search_patience"]), (24, 0, 0, 0))
        values = model_arguments(parser.parse_args(["--provider", "model", "--workers", "4",
            "--search-group-evaluations", "72", "--profile", "fifo43", "--no-bank-candidates",
            "--no-pack-reload-candidates", "--no-pack-preload-candidates"]))
        self.assertEqual((values["search_group_evaluations"], values["profile"], values["mreg_candidates"]), (72, "fifo43", ()))
        self.assertFalse(values["pack_reload_candidates"] or values["pack_preload_candidates"])
        builtin = model_arguments(parser.parse_args([]))
        self.assertEqual((builtin["profile"], builtin["search_evaluations"], builtin["search_group_evaluations"],
                          builtin["mreg_candidates"]), ("fifo43", 0, 0, ()))

    def test_fixed_batches_keep_all_compiles_canonical(self):
        with patch("atlas_model_selection.subprocess.run", side_effect=self.search_run):
            receipt = select_program(self.source, self.directory / "compiled", self.directory,
                curve_candidates=False, max_candidates=1, search_evaluations=13, workers=4,
                scorer=lambda program, **_: program["cycles"])
        self.assertEqual(len(receipt["candidates"]), 14)
        self.assertEqual(receipt["search"]["evaluations"], 13)
        self.assertEqual([row["parent_index"] for row in receipt["candidates"][1:13]], [0] * 12)
        self.assertEqual(len([argv for argv in self.invocations if candidate_command(argv)]), 14)
        self.assertFalse((self.directory / "compiled/prefetch").exists())

    def test_patience_stops_at_batch_boundary_and_still_checks_pack_alternatives(self):
        receipt = self.select(runner=self.graph_run([[] for _ in range(8)]), max_candidates=1,
                              search_evaluations=100, search_patience=12, workers=4,
                              pack_reload_candidates=True, pack_preload_candidates=True)
        self.assertEqual(receipt["search"]["evaluations"], 12)
        self.assertTrue(receipt["search"]["stopped_for_patience"])
        self.assertEqual([row["class"] for row in receipt["candidates"]][-2:],
                         ["pack-reload-order", "pack-preload-waits"])
        self.assertEqual(receipt["selected_index"], 0)
        self.assertEqual(receipt["score_cache_entries"], 1)
        self.assertEqual(receipt["score_cache_hits"], 14)

    def search_run(self, argv, **kwargs):
        """Give every option set and order distinct words and cycles on a graph with dependences."""
        result = self.fake_run(argv, **kwargs)
        option = candidate_command(argv)
        if result.returncode or not option:
            return result
        cwd = kwargs["cwd"]
        order = self.write_graph(cwd, [[], [0], [], [2], [], [], [5], []])
        # Provider spelling is metadata, not part of the synthetic instruction stream.
        word_options = " ".join(token for token in option.split() if not token.startswith("--atlas-compile-candidate=provider="))
        # Fixed salt exercises accepted moves in both bounded search phases.
        key = int(hashlib.sha256(repr((10, word_options, order)).encode()).hexdigest()[:8], 16)
        if key % 11 == 0 and "order-file" in option:
            return subprocess.CompletedProcess(argv, 1, "", "refused")
        # Order replays score below every knob candidate, so both search phases accept moves.
        cycles = 1000 + key % 97 + (0 if "order-file" in option else 100)
        self.programs[cwd] = {"instructions": [{"word_u32": key}], "cycles": cycles}
        return result

    def test_full_preset_trajectory_is_deterministic_and_bounded(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        expected = None
        for workers in (1, 4):
            arguments = model_arguments(parser.parse_args(["--provider", "model", "--workers", str(workers), "--profile", "fifo43"]))
            with patch("atlas_model_selection.subprocess.run", side_effect=self.search_run):
                receipt = select_program(self.source, self.directory / str(workers), self.directory,
                                         scorer=lambda program, **_: program["cycles"], **arguments)
            rows = receipt["candidates"]
            phases = [{key: value for key, value in receipt[field].items() if key != "elapsed_seconds"}
                      for field in ("search", "group_search")]
            trajectory = ([(row["class"], row["options"], row["status"], row.get("cycles"), row.get("parent_index"))
                           for row in rows], phases, receipt["selected_index"])
            if expected is None:
                expected = trajectory
            self.assertEqual(trajectory, expected)
            self.assertEqual(hashlib.sha256(json.dumps(trajectory, sort_keys=True).encode()).hexdigest(),
                             "3b889839358bc91f33015945f3b4639613f84c10dd064113cf75641df95b88fc")
            self.assertEqual([row["class"] for row in rows[:12]], ["default"] + ["lookahead"] * 3 + ["random-seed"] * 8)
            self.assertEqual([row["options"]["lookahead"] for row in rows[1:4]], [1, 2, 4])
            self.assertEqual(sum(row["class"] == "mreg-ranks" for row in rows), 8)
            self.assertFalse(any(row["class"] in ("coordinate-store-windows", "pack-preload-order") for row in rows))
            for field, budget in (("search", 96), ("group_search", 204)):
                phase = receipt[field]
                self.assertGreater(phase["evaluations"], 0)
                self.assertLessEqual(phase["evaluations"], budget)
                self.assertTrue(phase["accepted_indices"])
                for index in phase["accepted_indices"]:
                    self.assertLess(rows[index]["cycles"], rows[rows[index]["parent_index"]]["cycles"])
            winner = min((row for row in rows if row["status"] == "PASS"), key=lambda row: (row["cycles"], row["index"]))
            self.assertEqual(receipt["selected_index"], winner["index"])
            self.assertTrue(any(row["status"] == "FAIL" for row in rows))
            self.assertEqual([row["class"] for row in rows[-2:]], ["pack-reload-order", "pack-preload-waits"])

    def test_mreg_pool_is_eight_seeds_at_every_cap(self):
        vectors = [tuple(random.Random(seed).choices(range(32), k=64)) for seed in range(8)]
        for cap, evaluated in ((32, 8), (64, 8), (25, 2)):
            with self.subTest(cap=cap), patch("atlas_model_selection.subprocess.run", side_effect=self.fake_run):
                receipt = select_program(self.source, self.directory / str(cap), self.directory,
                    curve_candidates=False, max_candidates=cap, mreg_candidates=None,
                    store_window_candidates=((0, 1), (0, 2), (0, 3)), scorer=lambda program, **_: program["cycles"])
            ranks = [row["options"]["mreg_ranks"] for row in receipt["candidates"] if row["class"] == "mreg-ranks"]
            self.assertEqual((receipt["mreg_pool_size"], len(ranks)), (8, evaluated))
            self.assertEqual(ranks, vectors[:evaluated])

    def test_removed_search_controls_are_rejected(self):
        with self.assertRaises(TypeError):
            self.select(search_prefetch_batches=1)
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        from contextlib import redirect_stderr
        from io import StringIO
        for options in (("--search-budget", "max"), ("--search-budget", "wide"), ("--search-prefetch-batches", "3")):
            with self.subTest(options=options), redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(options)

    def test_invalid_search_controls_reject_before_tools(self):
        for options, diagnostic in (
            ({"search_patience": -1}, "search-patience"),
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
        self.assertEqual(receipt["profile"], "curve")
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
        self.assertEqual(receipt["profile"], "fifo43")

    def test_curve_ties_score_fifo43_lazily_and_cache_by_profile_and_words(self):
        calls = Counter()

        def scorer(program, profile):
            word = program["instructions"][0]["word_u32"]
            calls[profile, word] += 1
            return program["cycles"] if profile == "curve" else {0: 1, 1: 0, 2: 9, 8: 8}[word]

        def tied_run(argv, **kwargs):
            result = self.fake_run(argv, **kwargs)
            if candidate_command(argv) and "random-seed=0" in " ".join(argv):
                self.programs[kwargs["cwd"]] = {"instructions": [{"word_u32": 8}], "cycles": 5}
            return result

        with patch("atlas_model_selection.subprocess.run", side_effect=tied_run):
            receipt = select_program(self.source, self.directory / "compiled", self.directory,
                                     curve_candidates=False, max_candidates=16, scorer=scorer)
        self.assertEqual(receipt["selected_index"], 4)
        self.assertEqual(receipt["selected_cycles"], 5)
        self.assertEqual(calls, Counter({("curve", word): 1 for word in (0, 1, 2, 8)}) +
                         Counter({("fifo43", 2): 1, ("fifo43", 8): 1}))
        tied = [row for row in receipt["candidates"] if "fifo43_cycles" in row]
        self.assertTrue(all(row["cycles"] == 5 for row in tied))
        self.assertEqual([row["fifo43_cycles"] for row in tied[:2]], [9, 8])
        self.assertTrue(any(row["fifo43_score_cache_hit"] for row in tied))
        self.assertEqual(receipt["fifo43_tiebreak_cache_entries"], 2)
        self.assertEqual(json.loads((self.directory / "compiled/selection.json").read_text())["candidates"][4]["fifo43_cycles"], 8)

    def test_unique_curve_minimum_never_scores_fifo43(self):
        def scorer(program, profile):
            self.assertEqual(profile, "curve")
            return program["cycles"]

        with patch("atlas_model_selection.subprocess.run", side_effect=self.fake_run):
            receipt = select_program(self.source, self.directory / "compiled", self.directory,
                                     curve_candidates=False, max_candidates=3, scorer=scorer)
        self.assertEqual(receipt["selected_index"], 2)
        self.assertFalse(any("fifo43_cycles" in row for row in receipt["candidates"]))

    def test_curve_secondary_gains_drive_search_restarts_and_patience(self):
        trajectories = []
        for profile in ("curve", "fifo43"):
            def scorer(program, profile):
                return 10 if profile == "curve" else program["cycles"]

            with patch("atlas_model_selection.subprocess.run", side_effect=self.search_run):
                receipt = select_program(self.source, self.directory / profile, self.directory,
                                         profile=profile, curve_candidates=False, max_candidates=32,
                                         search_evaluations=48, search_group_evaluations=48, search_patience=12,
                                         workers=4, pack_reload_candidates=True,
                                         pack_preload_candidates=True, scorer=scorer)
            self.assertTrue(receipt["search"]["accepted_indices"])
            self.assertTrue(receipt["group_search"]["accepted_indices"])
            phases = [{key: value for key, value in receipt[field].items() if key != "elapsed_seconds"}
                      for field in ("search", "group_search")]
            trajectories.append(([(row["options"], row["status"], row.get("parent_index"))
                                  for row in receipt["candidates"]], phases, receipt["selected_index"]))
            if profile == "fifo43":
                self.assertFalse(any("fifo43_cycles" in row for row in receipt["candidates"]))
        self.assertEqual(*trajectories)

    def test_duplicate_words_are_scored_once(self) -> None:
        scores = []
        receipt = self.select(profile="fifo43", max_candidates=16,
                              scorer=lambda program, **_: scores.append(program["cycles"]) or 10)
        self.assertEqual(len(scores), 3)
        self.assertEqual(receipt["score_cache_entries"], 3)
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
        common = dict(curve_candidates=False, scorer=lambda program, **_: program["cycles"], store_window_candidates=((0,), (0,), (1,)))
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

    def refuse(self, refused):
        """Refuse every candidate compile whose option string satisfies refused."""
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

    def test_builtin_failure_still_stops_after_one_compile(self) -> None:
        with self.assertRaisesRegex(ValueError, "default candidate failed"):
            self.select(runner=self.refuse(lambda option: True), provider="builtin")
        self.assertEqual(sum(bool(candidate_command(argv)) for argv in self.invocations), 1)

    def test_candidate_choices_only_use_exposed_existing_knobs(self) -> None:
        options = candidate_options(Candidate(store_windows=(7, 3, 0), wait_placement="source", mreg_ranks=(31, 0, 7)),
                                    "model", True)
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

    def test_preload_candidates_replace_reload_and_compare_wait_placements(self):
        with self.assertRaisesRegex(ValueError, "pack-preload-candidates requires"):
            self.select(provider="builtin", pack_preload_candidates=True)
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
        receipt = self.select(runner=compile_placement, max_candidates=3, pack_reload_candidates=True,
                              pack_preload_candidates=True)
        self.assertEqual(receipt["selected_index"], 4)
        self.assertEqual(receipt["selected_cycles"], 2)
        self.assertEqual(receipt["candidates"][-1]["class"], "pack-preload-waits")
        for row in receipt["candidates"][-1:]:
            self.assertEqual(row["options"]["lookahead"], 2)
            self.assertFalse(row["options"]["pack_reload_after_weight"])
            self.assertTrue(row["options"]["pack_weight_before_pack"])
        self.assertEqual(receipt["candidates"][-1]["options"]["wait_placement"], "preloaded")

    def test_rejected_preload_alternatives_keep_reload_winner(self):
        def reject_preload(argv, **kwargs):
            if "pack-weight-before-pack=true" in " ".join(argv):
                return subprocess.CompletedProcess(argv, 1, "", "preload resource conflict")
            result = self.fake_run(argv, **kwargs)
            if candidate_command(argv) and "pack-reload-after-weight=true" in " ".join(argv):
                self.programs[kwargs["cwd"]] = {"instructions": [{"word_u32": 99}], "cycles": 4}
            return result
        receipt = self.select(runner=reject_preload, max_candidates=3, pack_reload_candidates=True,
                              pack_preload_candidates=True)
        self.assertEqual(receipt["selected_index"], 3)
        self.assertEqual(receipt["candidates"][-1]["status"], "FAIL")

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

    def test_curve_priorities_preserve_all_neighbors_and_change_with_pacing(self):
        initial = (0, 1, 2)
        current_first = (0, 2, 1)
        slow = proposal_graph([[], [0], []], initial)
        fast = proposal_graph([[], [0], []], initial, latency=1)
        program = {"model_evidence": {"op_timing_sha256": PROPOSAL_ID}}
        for graph in (slow, fast):
            validate_proposal_graphs([graph], program)
        orders = [(initial,), (current_first,)]
        self.assertEqual(next(rank_order_neighbors([slow], orders)), (current_first,))
        self.assertEqual(next(rank_order_neighbors([fast], orders)), (initial,))
        preds = [[], [], [0], [1, 2], []]
        graph = proposal_graph(preds, tuple(range(5)))
        for neighbors in (order_neighbors, group_order_neighbors):
            ordinary = list(neighbors([graph], (tuple(range(5)),)))
            ranked = list(rank_order_neighbors([graph], ordinary))
            self.assertEqual(set(ordinary), set(ranked))
            self.assertEqual(len(set(ordinary)), len(ranked))
            self.assertEqual([proposal_estimate(graph, order[0]) for order in ranked],
                             sorted(proposal_estimate(graph, order[0]) for order in set(ordinary)))
            self.assertEqual(ranked, list(rank_order_neighbors([graph], ordinary)))

    def test_proposal_active_transfer_heap_matches_full_history_replay(self):
        def reference(graph, order):
            proposal = graph["proposal"]
            free, finished, transfers = {}, [0] * len(order), []
            frontend = makespan = waiting = 0
            for node in order:
                cost = proposal["costs"][node]
                unit = cost["unit"]
                cycle = frontend if cost["queued_dma"] else max(frontend, free.get(unit, 0))
                if cost["drains"]:
                    cycle = max([cycle, *transfers])
                cycle = max([cycle, *(finished[pred] for pred in graph["preds"][node])])
                if cost["queued_dma"]:
                    busy = sorted(end for end in transfers if end > cycle)
                    capacity = proposal["dma_capacity"]
                    if len(busy) >= capacity:
                        cycle = busy[len(busy) - capacity]
                work = max(cycle + cost["frontend"] - 1, free.get(unit, 0)) if cost["queued_dma"] else cycle
                finished[node] = work + cost["latency"]
                free[unit] = work + cost["busy"]
                if cost["queued_dma"]:
                    transfers.append(work + cost["busy"])
                waiting += cycle - frontend
                frontend = cycle + cost["frontend"]
                makespan = max(makespan, finished[node])
            return makespan, waiting

        rng = random.Random(71)
        for _ in range(100):
            count = rng.randrange(1, 60)
            order = list(range(count))
            rng.shuffle(order)
            preds = [[] for _ in order]
            costs = []
            for position, node in enumerate(order):
                preds[node] = [earlier for earlier in order[:position] if rng.random() < .08]
            for node in range(count):
                queued = rng.choice((False, True))
                costs.append(dict(unit=rng.randrange(4), busy=rng.randrange(100),
                                  latency=rng.randrange(100), frontend=rng.randrange(1 if queued else 0, 10),
                                  queued_dma=queued, drains=rng.choice((False, True))))
            graph = dict(preds=preds, proposal=dict(costs=costs, dma_capacity=rng.randrange(1, 9)))
            self.assertEqual(proposal_estimate(graph, tuple(order)), reference(graph, order))

    def test_curve_bounded_ranking_preserves_exhaustive_attempt_prefix(self):
        initial = tuple(range(6))
        graph = proposal_graph([[], [], [0], [], [1], []], initial)
        base = Candidate(virtual_orders=(initial,))
        for neighbors in (order_neighbors, group_order_neighbors):
            ordinary = [(initial,), *neighbors([graph], (initial,))]
            exhaustive = sorted(ordinary, key=lambda orders: proposal_estimate(graph, orders[0]))
            all_candidates = list(dict.fromkeys(order_candidates(base, exhaustive, True)))
            # Seen candidates can include one sibling, both siblings, or neither.
            for stride in (2, 3, 5):
                seen = set(all_candidates[::stride])
                expected = [candidate for candidate in all_candidates if candidate not in seen]
                for remaining in (1, 2, 7, len(expected) + 1):
                    eligible = lambda order: any(candidate not in seen for candidate in
                                                 order_candidates(base, (order,), True))
                    ranked = rank_order_neighbors([graph], iter(ordinary), remaining, eligible)
                    actual = [candidate for candidate in order_candidates(base, ranked, True)
                              if candidate not in seen]
                    self.assertEqual(actual[:remaining], expected[:remaining])

    def test_curve_ranking_deduplicates_and_filters_before_replay(self):
        graph = proposal_graph([[], [], []], (0, 1, 2))
        first, second = ((1, 0, 2),), ((2, 0, 1),)
        original = proposal_estimate
        with patch("atlas_model_selection.proposal_estimate", wraps=original) as replay, \
             patch("atlas_model_selection.heapq.nsmallest", wraps=__import__("heapq").nsmallest) as smallest:
            ranked = list(rank_order_neighbors([graph], iter([first, second, first, second]), 1,
                                               eligible=lambda orders: orders != second))
        self.assertEqual(ranked, [first])
        self.assertEqual(replay.call_count, 2)  # One parent and one eligible unique order.
        self.assertEqual(smallest.call_args.args[0], 1)
        self.assertEqual(list(rank_order_neighbors([graph], iter([first]), 0)), [])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            rank_order_neighbors([graph], iter([first]), -1)

    def test_curve_metadata_requires_schema_identity_costs_and_native_parity(self):
        from copy import deepcopy
        graph = proposal_graph([[], [0], []], (0, 1, 2))
        program = {"model_evidence": {"op_timing_sha256": PROPOSAL_ID}}
        mutations = (
            lambda g: g.pop("proposal"),
            lambda g: g["proposal"].update(schema="future"),
            lambda g: g["proposal"].update(facts_sha256="b" * 64),
            lambda g: g["proposal"]["estimate"].update(makespan=0),
            lambda g: g["proposal"]["costs"][0].update(busy=-1),
            lambda g: g["proposal"]["costs"][0].update(queued_dma=1),
            lambda g: g["proposal"]["costs"][0].update(frontend=0),
            lambda g: g["proposal"]["costs"].pop(),
            lambda g: g.update(order=[1, 0, 2]),
        )
        for mutate in mutations:
            broken = deepcopy(graph)
            mutate(broken)
            with self.assertRaisesRegex(ValueError, "curve proposal metadata"):
                validate_proposal_graphs([broken], program)
        with self.assertRaisesRegex(ValueError, "selected op_timing_sha256"):
            validate_proposal_graphs([graph], {})

    def test_curve_wait_siblings_reset_preloaded_options(self):
        base = Candidate(pack_weight_before_pack=True, wait_placement="preloaded")
        siblings = list(order_candidates(base, [((0, 1),)], True))
        self.assertEqual([candidate.wait_placement for candidate in siblings], ["default", "source"])
        self.assertTrue(all(not candidate.pack_weight_before_pack for candidate in siblings))
        self.assertEqual(list(order_candidates(base, [((0, 1),)], False)),
                         [Candidate(pack_weight_before_pack=True, wait_placement="preloaded", virtual_orders=((0, 1),))])

    def test_curve_search_bounds_wait_siblings_and_uses_native_scores(self):
        initial = tuple(range(8))
        preds = [[], [0], [], [], [], [], [], []]
        graph = proposal_graph(preds, initial)
        ranked = list(rank_order_neighbors([graph], order_neighbors([graph], (initial,))))
        target = ranked[1][0]
        self.assertGreaterEqual(proposal_estimate(graph, target), proposal_estimate(graph, ranked[0][0]))

        def compile_order(argv, cwd, order):
            flags = " ".join(argv)
            source_waits = "wait-placement=source" in flags
            word = sum((node + 1) * 9 ** i for i, node in enumerate(order)) * 2 + int(source_waits)
            self.programs[cwd] = {"instructions": [{"word_u32": word}],
                                  "model_evidence": {"op_timing_sha256": PROPOSAL_ID},
                                  "cycles": 5 if order == target and source_waits else 10}
            if order == ranked[0][0] and not source_waits:
                return subprocess.CompletedProcess(argv, 1, "", "dependence refused")

        expected = None
        for workers in (1, 4):
            runner = self.graph_run(preds, initial, curve=True, configure=compile_order)
            receipt = self.select(name=f"curve-{workers}", runner=runner, profile="curve",
                                  max_candidates=1, search_evaluations=13,
                                  workers=workers)
            self.assertTrue(receipt["curve_candidates"])
            self.assertEqual(receipt["search"]["evaluations"], 13)
            self.assertEqual(receipt["selected_cycles"], 5)
            self.assertEqual(receipt["selected_class"], "local-order")
            self.assertEqual(receipt["search"]["proposal_facts_sha256"], PROPOSAL_ID)
            local = receipt["candidates"][1:]
            self.assertEqual(local[0]["status"], "FAIL")
            self.assertEqual([row["options"]["wait_placement"] for row in local[:4]],
                             ["default", "source", "default", "source"])
            self.assertEqual(tuple(local[2]["options"]["virtual_orders"][0]), target)
            self.assertEqual(receipt["candidates"][receipt["selected_index"]]["options"]["wait_placement"], "source")
            canonical = [(row["options"], row["status"], row.get("cycles"), row.get("parent_index"))
                         for row in receipt["candidates"]]
            if expected is None:
                expected = canonical
            self.assertEqual(canonical, expected)

    def test_curve_search_tests_unseen_wait_sibling_after_bank_choice(self):
        initial = (0, 2, 1)
        preds = [[], [0], []]
        for parent_source in (False, True):
            def compile_order(argv, cwd, order):
                flags = " ".join(argv)
                bank = "mreg-ranks=" in flags
                source = "wait-placement=source" in flags
                word = 100 + 2 * int(bank) + int(source)
                cycles = (9 if source == parent_source else 5) if bank else (10 if source == parent_source else 11)
                self.programs[cwd] = {"instructions": [{"word_u32": word}],
                                      "model_evidence": {"op_timing_sha256": PROPOSAL_ID}, "cycles": cycles}
            runner = self.graph_run(preds, initial, curve=True, configure=compile_order)
            receipt = self.select(name=f"curve-bank-sibling-{parent_source}", runner=runner, profile="curve",
                                  max_candidates=32, random_seeds=(), mreg_candidates=((1,),),
                                  search_evaluations=1, workers=1)
            parent = receipt["candidates"][receipt["search"]["start_index"]]
            self.assertEqual(parent["class"], "mreg-ranks")
            self.assertEqual(parent["options"]["wait_placement"], "source" if parent_source else "default")
            self.assertEqual(receipt["search"]["evaluations"], 1)
            winner = receipt["candidates"][receipt["selected_index"]]
            self.assertEqual(winner["cycles"], 5)
            self.assertEqual(winner["options"]["wait_placement"], "default" if parent_source else "source")
            self.assertEqual(tuple(winner["options"]["virtual_orders"][0]), initial)
            self.assertEqual(tuple(winner["options"]["mreg_ranks"]), (1,))

    def test_curve_group_search_charges_odd_budget_per_wait_sibling(self):
        initial = tuple(range(8))
        preds = [[], [0], [], [], [], [], [], []]
        def compile_order(argv, cwd, order):
            word = sum((node + 1) * 9 ** i for i, node in enumerate(order))
            self.programs[cwd] = {"instructions": [{"word_u32": word}], "cycles": 10,
                                  "model_evidence": {"op_timing_sha256": PROPOSAL_ID}}
        runner = self.graph_run(preds, initial, curve=True, configure=compile_order)
        receipt = self.select(name="curve-group", runner=runner, profile="curve", max_candidates=1,
                              search_group_evaluations=5)
        local = receipt["candidates"][1:]
        self.assertEqual(receipt["group_search"]["evaluations"], 5)
        self.assertEqual([row["options"]["wait_placement"] for row in local],
                         ["source", "default", "source", "default", "source"])
        self.assertEqual(tuple(local[0]["options"]["virtual_orders"][0]), initial)
        self.assertEqual(len({Candidate(**{**row["options"], "virtual_orders": tuple(
            tuple(order) for order in row["options"]["virtual_orders"])}) for row in local}), 5)
        self.assertEqual(receipt["selected_index"], 0)

    def test_curve_search_malformed_metadata_is_fatal_and_baseline_switch_bypasses_ranking(self):
        def compile_order(argv, **kwargs):
            result = self.fake_run(argv, **kwargs)
            if not result.returncode and candidate_command(argv):
                (kwargs["cwd"] / "graph.json").write_text(json.dumps([{"preds": [[], []], "order": [0, 1]}]))
            return result
        with self.assertRaisesRegex(ValueError, "curve proposal metadata"):
            self.select(name="curve-invalid", runner=compile_order, profile="curve", max_candidates=1,
                        search_evaluations=1)
        receipt = self.select(name="curve-disabled", runner=compile_order, profile="curve",
                              curve_candidates=False, max_candidates=1, search_evaluations=1)
        self.assertFalse(receipt["curve_candidates"])
        self.assertEqual(receipt["search"]["evaluations"], 1)

    def test_curve_candidate_switch_is_forwarded_and_rejects_other_profiles(self):
        parser = argparse.ArgumentParser()
        add_model_arguments(parser)
        self.assertIsNone(model_arguments(parser.parse_args([]))["curve_candidates"])
        self.assertFalse(model_arguments(parser.parse_args(["--no-curve-candidates"]))["curve_candidates"])
        self.assertTrue(model_arguments(parser.parse_args(["--curve-candidates"]))["curve_candidates"])
        for kwargs in ({"provider": "builtin", "profile": "curve"}, {"profile": "fifo43"}):
            with self.assertRaisesRegex(ValueError, "curve-candidates requires"):
                self.select(curve_candidates=True, **kwargs)


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
            ("--lower-atlas-virtual-to-machine=pack-weight-before-pack=true", "pack-weight-before-pack requires the model"),
            ("--schedule-atlas-stream=wait-placement=source", "wait-placement=source requires the model"),
            ("--schedule-atlas-stream=wait-placement=preloaded", "wait-placement=preloaded requires the model"),
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


# External evidence is optional only when neither pin was configured.
@requires_model_evidence
class TiledSelectionTest(unittest.TestCase):
    def test_mlp16_default_is_ring_legal_after_wait_insertion(self):
        source = ROOT / "test/examples/virtual_fp8_mlp_16_tiles.mlir"
        facts, pairs = Path(os.environ["ATLAS_TEST_OP_TIMING"]), Path(os.environ["ATLAS_TEST_PAIR_TIMING"])
        digests = {"op_timing_sha256": hashlib.sha256(facts.read_bytes()).hexdigest(),
                   "pairs_sha256": hashlib.sha256(pairs.read_bytes()).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            receipt = select_program(source, Path(directory) / "selected", BIN, max_candidates=1, op_timing=facts,
                                     pairs=pairs, scorer=lambda program, **_: len(program["instructions"]), **digests)
        self.assertEqual((receipt["candidates"][0]["status"], receipt["selected_index"]), ("PASS", 0))
        # Derived costs may change whether this default needs an extra ring wait.
        # Explicit ninth-launch negative controls are covered by test_dma_ring_waits.
        evidence = evidence_option(facts, digests["op_timing_sha256"], pairs, digests["pairs_sha256"])
        lowered = run("atlas-opt", source.read_text(), *evidence, "--atlas-compile-prepare",
                      "--schedule-atlas-virtual=provider=model", "--atlas-compile-spill",
                      "--atlas-compile-lower=provider=model")
        self.assertEqual(lowered.returncode, 0, lowered.stderr)
        timed = run("atlas-opt", lowered.stdout, "--insert-atlas-dma-waits=provider=model",
                    "--schedule-atlas-stream=provider=model", "--verify-atlas-timing",
                    "--verify-atlas-generated-schedule")
        self.assertEqual(timed.returncode, 0, timed.stderr)


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

    def test_native_proposal_metadata_matches_explicit_order_replay(self):
        graphs, _ = self.graph()
        identity = re.search(r'op-timing-sha256=([0-9a-f]{64})', self.evidence).group(1)
        program = {"model_evidence": {"op_timing_sha256": identity}}
        validate_proposal_graphs(graphs, program)
        self.order_path.write_text(json.dumps([graph["order"] for graph in graphs]))
        replay_path = self.directory / "replayed-graph.json"
        replay = self.schedule(order_file=self.order_path, graph_output=replay_path)
        self.assertEqual(replay.returncode, 0, replay.stderr)
        replayed = json.loads(replay_path.read_text())
        validate_proposal_graphs(replayed, program)
        self.assertEqual(graphs, replayed)
        for graph in graphs:
            self.assertEqual(len(graph["proposal"]["costs"]), len(graph["operations"]))
            for name, cost in zip(graph["operations"], graph["proposal"]["costs"]):
                if name in ("atlas.virtual_input_bf16", "atlas.virtual_output_bf16"):
                    self.assertTrue(cost["drains"])
                    self.assertEqual(cost["frontend"], cost["latency"])
                if name in ("atlas.virtual_dma_load_bf16", "atlas.virtual_dma_store_bf16"):
                    self.assertTrue(cost["queued_dma"])

    def test_native_pack_proposal_is_frontend_blocking_beside_queued_load(self):
        from test_dma_ring_waits import fixture_evidence
        self.evidence = fixture_evidence(self.directory)
        self.source = function("pack", [start(), inp("io0", "s1", "tile", 0),
            const("address", -2147481600), const("size", 1024),
            dma_load("s1", "s2", "weight", "address", "size", "fp8"),
            f'%packed = "atlas.virtual_pack_fp8"(%tile) {{scale_code = 127 : i32}} : ({T}) -> {F}',
            outp("s2", "s3", "tile", 0), f"return %s3 : {S}"])
        graphs, _ = self.graph()
        graph = graphs[0]
        pack = graph["operations"].index("atlas.virtual_pack_fp8")
        load = graph["operations"].index("atlas.virtual_dma_load_fp8")
        costs = graph["proposal"]["costs"]
        self.assertEqual(costs[pack]["frontend"], costs[pack]["latency"])
        self.assertGreater(costs[pack]["frontend"], 1)
        self.assertFalse(costs[pack]["drains"])
        self.assertTrue(costs[load]["queued_dma"])
        self.assertEqual(costs[load]["frontend"], 1)
        validate_proposal_graphs(graphs, {"model_evidence": {
            "op_timing_sha256": re.search(r'op-timing-sha256=([0-9a-f]{64})', self.evidence).group(1)}})

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
