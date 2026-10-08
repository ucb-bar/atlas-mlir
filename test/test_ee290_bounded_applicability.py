#!/usr/bin/env python3
"""Focused rejection tests for bounded evidence selection and resolver binding."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("bounded", Path(__file__).parent.parent / "tools/check-ee290-bounded-applicability.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.words = [0x10000313, 0x00030207]
        self.evidence = {"evidence_sha256": "1" * 64, "manifest_sha256": "2" * 64, "hardware_ir_sha256": "3" * 64}
        self.export = {"schema": "atlas.resolved_rtl_timing.v0", "target_config": "EE290SimConfig", "qualification": "conditional", "scheduling_qualified": False,
                       "resolver": {"id": "atlas.vls.conservative.v1", "version": 1}, "evidence": self.evidence.copy(),
                       "program": {"words": self.words.copy(), "word_count": 2, "words_sha256": hashlib.sha256(b"".join(struct.pack("<I", x) for x in self.words)).hexdigest()},
                       "instructions": [{"word_index": i, "word_u32": word, "logical_issue_cycle": i, "operands": {}, "footprint": {}} for i, word in enumerate(self.words)],
                       "applicability": {"scope": "finite", "supported_operations": ["vld"], "operand_domain": {}, "environment_assumptions": ["no competition"], "unsupported": ["fullsystem"]}}

    def test_accepts_bound_conditional_export(self):
        self.assertIs(B.validate_export(self.export, self.evidence, self.words), self.export)

    def test_rejects_evidence_substitution(self):
        self.export["evidence"]["hardware_ir_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "evidence mismatch"):
            B.validate_export(self.export, self.evidence, self.words)

    def test_rejects_program_substitution(self):
        self.export["program"]["words"][0] += 1
        with self.assertRaisesRegex(ValueError, "words mismatch"):
            B.validate_export(self.export, self.evidence, self.words)

    def test_rejects_word_digest_and_instruction_substitution(self):
        for field in ("digest", "instruction"):
            export = copy.deepcopy(self.export)
            if field == "digest": export["program"]["words_sha256"] = "0" * 64
            else: export["instructions"][1]["word_u32"] = 0
            with self.assertRaises(ValueError): B.validate_export(export, self.evidence, self.words)

    def test_rejects_qualification_promotion(self):
        for key, value in (("qualification", "qualified"), ("scheduling_qualified", True), ("target_config", "OtherConfig")):
            export = copy.deepcopy(self.export); export[key] = value
            with self.assertRaises(ValueError): B.validate_export(export, self.evidence, self.words)

    def test_rejects_missing_applicability(self):
        del self.export["applicability"]["environment_assumptions"]
        with self.assertRaisesRegex(ValueError, "applicability"):
            B.validate_export(self.export, self.evidence, self.words)

    def test_selection_requires_hash_and_strict_json(self):
        with tempfile.TemporaryDirectory(prefix="bounded-evidence-") as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{"state":"passed"}')
            checker = B.CHECK.Checker(); identity = checker.identity(path)
            self.assertEqual(B.select(checker, path, identity["sha256"])[1], {"state": "passed"})
            with self.assertRaises(ValueError): B.select(checker, path, "0" * 64)
            path.write_text('{"state":1,"state":2}')
            with self.assertRaises(ValueError): B.select(B.CHECK.Checker(), path, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_restricted_and_symlink_paths_rejected_before_open(self):
        with self.assertRaises(ValueError): B.bootstrap('/tmp/forbidden-' + 'ham' + 'mer' + '/absent')
        with tempfile.TemporaryDirectory(prefix="bounded-path-") as directory:
            link = Path(directory) / "dependency"
            link.symlink_to('/tmp/' + 'vl' + 'si' + '/absent')
            with self.assertRaises(ValueError): B.bootstrap(link)

    def test_historical_identity_can_bind_executed_snapshot(self):
        declared = {"path": "/old/producer.py", "sha256": "a" * 64, "bytes": 12}
        snapshot = dict(declared, path="/saved/producer.executed.py")
        B.content(declared, snapshot, "producer")
        snapshot["sha256"] = "b" * 64
        with self.assertRaises(ValueError): B.content(declared, snapshot, "producer")

    def test_final_stability_rejection_does_not_publish_success(self):
        class Unstable:
            def recheck(self): raise ValueError("changed input")
        with tempfile.TemporaryDirectory(prefix="bounded-publish-") as directory:
            path = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "changed input"):
                B.publish_report(Unstable(), path, {"state": "finite_observations_validated"})
            self.assertFalse(path.exists())


class EventBindingTests(unittest.TestCase):
    def setUp(self):
        def instruction(mnemonic, cycle, line, store):
            return {"mnemonic": mnemonic, "logical_issue_cycle": cycle, "operands": {"rd": 4},
                    "footprint": {"accesses": [
                        {"resource": "vmem", "first": line, "write": store, "age": 3 if store else 1, "count": 32, "step": 1, "anywhere": False, "at_completion": False},
                        {"resource": "mreg", "first": 128, "write": not store, "age": 1 if store else 3, "count": 32, "step": 1, "anywhere": False, "at_completion": False}],
                        "holds": [{"unit": "VLOAD path", "to": 34}, {"unit": "VSTORE path", "to": 34}]}}
        self.export = {"instructions": [instruction("vload", 1, 32, False), instruction("vstore", 36, 96, True),
                                      {"mnemonic": "csrrw", "logical_issue_cycle": 71},
                                      {"mnemonic": "ecall", "logical_issue_cycle": 72, "event_kind": "terminal_acceptance"}],
                       "applicability": {"environment_assumptions": ["sram_read_response_one_cycle_after_request"]}}
        def command(edge, op, line):
            return {"edge": edge, "op": op, "line": line, "mreg": 4,
                    "source_edges": list(range(edge+1, edge+33)), "response_edges": list(range(edge+2, edge+34)),
                    "destination_edges": list(range(edge+3, edge+35)), "release_edge": edge+35}
        panel = {"commands": [command(101, 1, 32), command(136, 2, 96)], "marker_edge": 171, "halt_edge": 172}
        self.boundary = {"panels": [copy.deepcopy(panel) for _ in range(3)]}

    def test_accepts_direct_footprint_and_completion_binding(self):
        self.assertTrue(B.bind_events(self.export, self.boundary)["access_ages"])

    def test_rejects_well_formed_trace_for_different_operands(self):
        # The decoder could independently admit another aligned tile; selected
        # words are unchanged, so the resolver/trace binding must reject it.
        self.boundary["panels"][1]["commands"][0]["line"] = 64
        with self.assertRaisesRegex(ValueError, "operands"):
            B.bind_events(self.export, self.boundary)

    def test_rejects_substituted_issue_gap(self):
        self.boundary["panels"][0]["commands"][1]["edge"] += 1
        with self.assertRaisesRegex(ValueError, "issue gap"):
            B.bind_events(self.export, self.boundary)

    def test_rejects_stream_or_release_or_completion_shift(self):
        for field in ("source_edges", "release_edge", "halt_edge"):
            boundary = copy.deepcopy(self.boundary)
            if field == "source_edges": boundary["panels"][0]["commands"][0][field][0] += 1
            elif field == "release_edge": boundary["panels"][0]["commands"][0][field] += 1
            else: boundary["panels"][0][field] += 1
            with self.assertRaises(ValueError): B.bind_events(self.export, boundary)


if __name__ == "__main__":
    unittest.main()
