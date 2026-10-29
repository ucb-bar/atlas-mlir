"""Selection keeps the lowest-cycle legal program in candidate order."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from itertools import permutations
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_evidence import requires_model_evidence
from atlas_model_selection import (Candidate, SelectionConfig, candidate_options, evidence_option,
                                   group_order_neighbors, select_program)
from test_virtual_ssa import BIN, run
from test_virtual_scheduling import S, function, start, const, dma_load, outp


def candidate_command(argv):
    return next((arg for arg in argv if arg.startswith("--atlas-compile-candidate=")), None)


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
            self.programs[cwd] = {"instructions": [{"word_u32": word}],
                                  "cycles": {0: 10, 1: 20, 2: 5, 8: 5}[word]}
        return subprocess.CompletedProcess(argv, 0, "", "")

    def select(self, *, name="compiled", runner=None, **kwargs):
        kwargs.setdefault("scorer", lambda program, **_: program["cycles"])
        with patch("atlas_model_selection.subprocess.run", side_effect=runner or self.fake_run):
            scorer = kwargs.pop("scorer")
            return select_program(self.source, self.directory / name, self.directory,
                                  SelectionConfig(**kwargs), scorer=scorer)

    def graph_run(self, preds, initial=None, *, configure=None):
        def run(argv, *, cwd, **kwargs):
            result = self.fake_run(argv, cwd=cwd, **kwargs)
            if result.returncode or not candidate_command(argv):
                return result
            path = cwd / "order.json"
            order = tuple(json.loads(path.read_text())[0]) if path.exists() else tuple(
                range(len(preds)) if initial is None else initial)
            (cwd / "graph.json").write_text(json.dumps([{"preds": preds, "order": order}]))
            return (configure(argv, cwd, order) if configure else None) or result
        return run


class ModelSelectionTest(SelectionFixture, unittest.TestCase):
    def test_candidate_options(self):
        self.assertEqual(candidate_options(Candidate(mreg_ranks=(31, 0, 7),
                                                      wait_placement="source"), "model", True),
                         ["--atlas-compile-candidate=provider=model spill=true allocation-report=allocation.json "
                          "pack-relayout-rows=2 wait-reused-dma-slots=true "
                          "wait-placement=source mreg-ranks='31,0,7'"])
        self.assertNotIn("wait-placement", candidate_options(Candidate(), "model", True)[0])
        self.assertIn("order-file=order.json", candidate_options(Candidate(order_file=((0,),)), "model", True)[0])

    def test_seed_pool_and_budget_prefix(self):
        vectors = [tuple(random.Random(seed).choices(range(32), k=64)) for seed in range(8)]
        results = []
        for cap, evaluated in ((12, 2), (18, 8)):
            result = self.select(name=str(cap), max_candidates=cap, mreg_candidates=None)
            ranks = [row["options"]["mreg_ranks"] for row in result["candidates"] if row["class"] == "mreg-ranks"]
            self.assertEqual(ranks, vectors[:evaluated])
            results.append(result["candidates"])
        self.assertEqual([row["options"] for row in results[0]],
                         [row["options"] for row in results[1][:len(results[0])]])
        self.assertEqual([row["options"]["random_seed"] for row in results[0][1:9]], list(range(8)))

    def test_neighbors_preserve_dependencies_and_blocks(self) -> None:
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

    def test_neighbors_move_crossed_dependencies(self) -> None:
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

    def test_invalid_search_controls_reject_before_tools(self):
        with self.assertRaisesRegex(ValueError, "search-group-evaluations"):
            self.select(search_group_evaluations=-1)
        self.assertEqual(self.invocations, [])

    def test_builtin_scores_and_selects_alternatives(self) -> None:
        result = self.select(provider="builtin", max_candidates=5)
        self.assertEqual(len(result["candidates"]), 5)
        self.assertEqual((result["selected_index"], result["selected_cycles"]), (2, 5))
        self.assertEqual(result["profile"], "fifo43")

    def test_parallel_tie_uses_candidate_index(self) -> None:
        later_finished = threading.Event()

        def reordered(argv, **kwargs):
            if "random-seed=1" in " ".join(argv):
                self.assertTrue(later_finished.wait(timeout=2))
            result = self.fake_run(argv, **kwargs)
            if Path(argv[0]).name == "atlas-emit" and self.programs[kwargs["cwd"]]["cycles"] == 5:
                later_finished.set()
            return result

        result = self.select(runner=reordered, max_candidates=5, workers=4)
        self.assertEqual(result["selected_index"], 2)

    def refuse(self, refused):
        def run(argv, **kwargs):
            if (option := candidate_command(argv)) and refused(option):
                self.invocations.append(argv)
                return subprocess.CompletedProcess(argv, 1, "", "error: model DMA command slot is still pending")
            return self.fake_run(argv, **kwargs)
        return run

    def test_default_refusal_continues(self) -> None:
        default = candidate_options(Candidate(), "model", True)[0]
        result = self.select(runner=self.refuse(lambda option: option == default), max_candidates=5)
        failed = result["candidates"][0]
        self.assertEqual((failed["status"], failed["error_stage"]), ("FAIL", "compile"))
        self.assertIsNone(result["default_cycles"])
        self.assertEqual((result["selected_index"], result["selected_cycles"]), (2, 5))
        self.assertTrue((self.directory / "compiled/program.json").exists())

    def test_all_refusals_fail_and_count_toward_budget(self):
        for provider in ("model", "builtin"):
            self.invocations.clear()
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, "no legal candidate"):
                self.select(name=provider, runner=self.refuse(lambda option: True),
                            provider=provider, max_candidates=5)
            self.assertEqual(sum(bool(candidate_command(argv)) for argv in self.invocations), 5)
            self.assertFalse((self.directory / provider / "program.json").exists())
            result = json.loads((self.directory / provider / "selection.json").read_text())
            self.assertEqual({row["status"] for row in result["candidates"]}, {"FAIL"})

    def test_group_siblings_share_budget(self):
        orders = [((1, 2, 0),), ((2, 0, 1),)]

        def compile_order(argv, cwd, order):
            if order == (1, 2, 0):
                return subprocess.CompletedProcess(argv, 1, "", "DMA command slot is still pending")

        for profile, placements, statuses in (
                ("curve", ["default", "source", "default"], ["FAIL", "FAIL", "PASS"]),
                ("fifo43", ["default", "default"], ["FAIL", "PASS"])):
            with self.subTest(profile=profile), patch(
                    "atlas_model_selection.group_order_neighbors", return_value=iter(orders)):
                result = self.select(name=profile, profile=profile, max_candidates=1,
                                      search_group_evaluations=3, scorer=lambda *args, **kwargs: 10,
                                      runner=self.graph_run([[], [], []], configure=compile_order))
            rows = result["candidates"][1:]
            self.assertEqual([row["options"]["wait_placement"] for row in rows], placements)
            self.assertEqual([row["status"] for row in rows], statuses)
            self.assertEqual(result["group_search"]["evaluations"], len(rows))
            self.assertEqual(result["selected_index"], 0)


class CandidateControlsTest(unittest.TestCase):
    def test_invalid_controls_refuse(self):
        source = (ROOT / "test/examples/virtual_bf16_shared_relu.mlir").read_text()
        cases = [("--schedule-atlas-virtual=order-file=missing.json", "cannot read order-file"),
                 ("--schedule-atlas-stream=wait-placement=unknown", "wait-placement must be")]
        cases += [(f"--{stage}=pack-relayout-rows=3", "pack-relayout-rows must be")
                  for stage in ("schedule-atlas-virtual", "lower-atlas-virtual-to-machine")]
        cases += [(f"--lower-atlas-virtual-to-machine=provider=model mreg-ranks='{ranks}'",
                   "mreg-ranks must contain") for ranks in ("-1", "1,,2", "4294967296", "unknown")]
        for option, diagnostic in cases:
            with self.subTest(option=option):
                result = run("atlas-opt", source, option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)


@requires_model_evidence
class ModelFactsBundleTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        self.facts = self.bundle / "model-timing.json"
        self.facts.write_bytes((Path(os.environ["ATLAS_TEST_MODEL_FACTS"]) /
                               "model-timing.json").read_bytes())

    def test_selection_binds_current_facts(self):
        # Whitespace is valid JSON and belongs to the selected file's hash.
        self.facts.write_bytes(self.facts.read_bytes() + b" ")
        selected = run("atlas-opt", "module {}", *evidence_option(self.bundle))
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertIn(hashlib.sha256(self.facts.read_bytes()).hexdigest(), selected.stdout)

    def test_malformed_facts_refuse(self):
        self.facts.write_text("{")
        selected = run("atlas-opt", "module {}", *evidence_option(self.bundle))
        self.assertNotEqual(selected.returncode, 0, selected.stdout)


@requires_model_evidence
class TiledSelectionTest(unittest.TestCase):
    def test_one_candidate_real_emit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selected"
            result = select_program(ROOT / "test/examples/virtual_fp8_mlp_16_tiles.mlir", output, BIN,
                                     SelectionConfig(max_candidates=1,
                                         model_facts=Path(os.environ["ATLAS_TEST_MODEL_FACTS"])),
                                     scorer=lambda program, **_: len(program["instructions"]))
            self.assertEqual((result["candidates"][0]["status"], result["selected_index"]), ("PASS", 0))


@requires_model_evidence
class GuardedModelPreparationTest(unittest.TestCase):
    def test_guarded_model_preparation_converts_pack(self):
        from test_guarded_pack import PACK, source

        consumer = ('%result = "atlas.virtual_mxu_matmul"(%p, %p) {unit = 0 : i32} '
                    ': (!atlas.virtual_fp8, !atlas.virtual_fp8) -> !atlas.virtual_bf16')
        evidence = evidence_option(Path(os.environ["ATLAS_TEST_MODEL_FACTS"]))
        for guarded in (False, True):
            flag = "--atlas-compile-prepare" + ("=guarded-pack=true" if guarded else "")
            prepared = run("atlas-opt", source(after=consumer, value="%result"), *evidence, flag)
            self.assertEqual(prepared.returncode, 0, prepared.stderr)
            self.assertEqual(PACK not in prepared.stdout, guarded)
            if guarded:
                self.assertIn('"atlas.virtual_mxu_readout_fp8"', prepared.stdout)


@requires_model_evidence
class ModelOrderReplayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        from test_dma_ring_waits import fixture_evidence
        self.evidence = fixture_evidence(self.directory)
        self.graph_path = self.directory / "graph.json"
        self.order_path = self.directory / "order.json"

    def schedule(self, source=None, provider="model", **options):
        option = f"--schedule-atlas-virtual=provider={provider} " + " ".join(
            f"{key.replace('_', '-')}={value}" for key, value in options.items())
        return run("atlas-opt", source or self.source, self.evidence, option, "--verify-atlas-virtual-stream")

    def graph(self):
        result = self.schedule(graph_output=self.graph_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(self.graph_path.read_text()), result.stdout

    def replay(self, orders):
        self.order_path.write_text(json.dumps(orders))
        return self.schedule(order_file=self.order_path)

    def test_replay_matches_scheduler_and_repairs_state(self):
        self.source = function("loads", [start(), const("address", -2147481600), const("size", 2048),
            dma_load("io0", "s1", "a", "address", "size", "bf16"),
            dma_load("s1", "s2", "b", "address", "size", "bf16"),
            outp("s2", "s3", "a", 0), outp("s3", "s4", "b", 1), f"return %s4 : {S}"])
        graphs, ordinary = self.graph()
        replay = self.replay([graph["order"] for graph in graphs])
        self.assertEqual(replay.returncode, 0, replay.stderr)
        self.assertEqual(replay.stdout, ordinary)
        order = graphs[0]["order"]
        loads = [i for i, name in enumerate(graphs[0]["operations"]) if name == "atlas.virtual_dma_load_bf16"]
        left, right = (order.index(node) for node in loads)
        order[left], order[right] = order[right], order[left]
        replay = self.replay([order])
        self.assertEqual(replay.returncode, 0, replay.stderr)

        node, preds = next((node, preds) for node, preds in enumerate(graphs[0]["preds"]) if preds)
        order.remove(node)
        order.insert(order.index(preds[0]), node)
        rejected = self.replay([order])
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("violates a dependence", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
