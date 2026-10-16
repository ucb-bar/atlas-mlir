"""Fail-closed projection tests; no producer, model or RTL execution."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_model_pairs as pairs


PROOF = {"confirmed": True, "rtl_sha256": "a" * 64, "source": {"path": "proof.json", "sha256": "b" * 64},
         "scope": "Exact issued operation sequence and complete operand relations", "rtl_gap_points": [32]}
BLOCKS = {name: {"first_free_age": 95} for name in ("mxu0.matmul", "mxu0.matmul_acc")}
BLOCKS["mxu1.matmul"] = {"first_free_age": 35}
BLOCKS["scalar_lsu.load"] = {"first_free_age": 4}
RELATION = {"mreg_banks": "disjoint", "accumulator": "other", "weight_slot": "same"}


def pair():
    return dict(first="mxu0.matmul", second="mxu0.matmul", relation=deepcopy(RELATION), swept=[1, 95], min_gap=32,
                negative={"gap": 31, "result": "conflict", "error": {"message": "read port busy"}},
                first_live_window_end=95, verilator=deepcopy(PROOF))


def chain():
    matrix = [dict(first_index=i, second_index=j, relation=dict(RELATION, accumulator="same" if (i, j) == (0, 2) else "other"))
              for i in range(3) for j in range(i + 1, 3)]
    return dict(operations=["mxu0.matmul"] * 3, gaps=[32, 32], relation=deepcopy(matrix[0]["relation"]),
                all_pair_relations=matrix, relation_scope="adjacent", result="accepted", in_flight_count=3,
                witness_only=True, no_stable_suffix_claim=True, verilator=deepcopy(PROOF))


class SmallChunks(BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 17) if size >= 0 else 17)


class ModelPairProjectionTest(unittest.TestCase):
    def select(self, record, kind="pairs"):
        return pairs.select_record(record, kind, 7, BLOCKS)

    def test_confirmed_pair_retains_semantics_and_exact_source_proof_pins(self):
        record = pair()
        record["trial_evidence_sha256"] = ["c" * 64]
        projected, reason = self.select(record)
        self.assertIsNone(reason)
        for key in ("first", "second", "relation", "negative", "swept", "min_gap", "first_live_window_end"):
            self.assertEqual(projected[key], record[key])
        self.assertEqual(projected["source_record"]["json_pointer"], "/pairs/7")
        self.assertEqual(projected["source_record"]["sha256"], pairs.sha256(pairs.canonical(record)))
        self.assertEqual(projected["source_record"]["proof_sha256"], pairs.sha256(pairs.canonical(record["verilator"])))
        self.assertEqual(projected["evidence_summary"]["field_sha256"]["trial_evidence_sha256"],
                         pairs.sha256(pairs.canonical(record["trial_evidence_sha256"])))

    def test_all_three_pair_window_forms_are_exclusive_at_the_same_boundary(self):
        for fields in ({"first_live_window_end": 95}, {"first_completion_age": 94},
                       {"live_window": {"first_completion_age": 94, "high_includes_idle_boundary": True}}):
            record = pair()
            del record["first_live_window_end"]
            record.update(fields)
            with self.subTest(fields=fields):
                self.assertEqual(pairs.boundary(record), 95)
                self.assertIsNone(self.select(record)[1])
        record["first_live_window_end"] = 95
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.select(record)

    def test_unknown_relation_or_operation_and_cross_group_exclude_whole_fact(self):
        for change, reason in (({"relation": dict(RELATION, unknown_bank="other")}, "unsupported_relation"),
                               ({"second": "mxu1.matmul"}, "cross_group"),
                               ({"second": "scalar_lsu.load"}, "cross_group"),
                               ({"second": "unknown"}, "unsupported_operation"),
                               ({"unreviewed_constraint": True}, "unsupported_record_field"),
                               ({"evidence_tier": "full_core_exact_witness_v1"}, "unsupported_protocol")):
            record = pair()
            record.update(change)
            with self.subTest(change=change):
                self.assertEqual(self.select(record), (None, reason))

    def scalar_pair(self, **qualifiers):
        relation = dict(mreg_banks="disjoint", first_mnemonic="lw", second_mnemonic="sw",
                        scalar_address_alignment="word", scalar_registers="independent",
                        vmem_bank="same", vmem_window="disjoint")
        relation.update(qualifiers)
        return dict(first="scalar_lsu.load", second="scalar_lsu.store", relation=relation,
                    swept=[1, 4], min_gap=1, first_completion_age=3,
                    negative={"gap": 0, "result": "inexpressible"}, verilator={"confirmed": True})

    def test_scalar_projection_preserves_every_qualifier_and_legacy_policy_excludes(self):
        blocks = dict(BLOCKS, **{"scalar_lsu.store": {"first_free_age": 2}})
        record = self.scalar_pair()
        selected, reason = pairs.select_record(record, "pairs", 3498, blocks)
        self.assertIsNone(reason)
        self.assertEqual(selected["relation"], record["relation"])
        self.assertEqual(pairs.select_record(record, "pairs", 3498, blocks, False), (None, "unsupported_operation"))
        for key in ("first_mnemonic", "second_mnemonic", "scalar_address_alignment", "scalar_registers", "vmem_bank"):
            changed = deepcopy(record)
            del changed["relation"][key]
            with self.subTest(key=key):
                self.assertEqual(pairs.select_record(changed, "pairs", 0, blocks), (None, "unsupported_relation"))

    def test_complete_scalar_role_vocabulary_and_joint_qualifiers(self):
        for mode in pairs.RELATION_VALUES["scalar_registers"]:
            relation = self.scalar_pair()["relation"]
            relation["scalar_registers"] = mode
            if mode != "independent":
                relation["scalar_other_registers"] = "independent"
            if mode not in {"independent", "shared_base_read_only"}:
                relation["scalar_destination_file"] = "xrf"
            with self.subTest(mode=mode):
                pairs.validate_relation(relation)
                if mode not in {"independent", "shared_base_read_only"}:
                    relation["scalar_destination_file"] = "erf"
                    if mode == "same_load_destination":
                        pairs.validate_relation(relation)
                    else:
                        with self.assertRaises(ValueError):
                            pairs.validate_relation(relation)
        for change in ({"scalar_registers": "unknown"}, {"scalar_destination_file": "xrf"},
                       {"scalar_other_registers": "aliased"}, {"vmem_bank": "unknown"},
                       {"scalar_address_alignment": "byte"}, {"first_scalar_byte_offset": "1"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                pairs.validate_relation(dict(self.scalar_pair()["relation"], **change))

    def test_explicit_byte_offsets_are_mandatory_and_exact(self):
        relation = self.scalar_pair(scalar_address_alignment="explicit_byte_offsets",
                                    first_scalar_byte_offset="1", second_scalar_byte_offset="3")["relation"]
        relation["vmem_word_window"] = relation.pop("vmem_window")
        pairs.validate_relation(relation)
        for key in ("first_scalar_byte_offset", "second_scalar_byte_offset"):
            missing = dict(relation)
            del missing[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                pairs.validate_relation(missing)
        with self.assertRaises(ValueError):
            pairs.validate_relation(dict(relation, first_scalar_byte_offset="4"))

    def test_original_streaming_weight_proof_semantics_are_preserved_and_checked(self):
        record = pair()
        record["verilator"]["relation"] = deepcopy(record["relation"])
        record["verilator"]["result_sha256"] = "a" * 64
        record["verilator"]["same_accumulator"] = False
        selected, reason = self.select(record)
        self.assertIsNone(reason)
        for key in ("relation", "result_sha256", "same_accumulator"):
            self.assertEqual(selected["verilator"][key], record["verilator"][key])
        for key, value in (("relation", {"mreg_banks": "disjoint"}), ("result_sha256", "invalid"),
                           ("same_accumulator", True), ("same_accumulator", "false")):
            changed = deepcopy(record)
            changed["verilator"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.select(changed)

    def test_unknown_proof_protocol_is_never_discarded(self):
        for key, value in (("protocol", "unreviewed"), ("review_protocol", {}), ("new_qualification", True)):
            record = pair()
            record["verilator"][key] = value
            with self.subTest(key=key):
                self.assertEqual(self.select(record), (None, "unsupported_protocol"))

    def test_qualified_metadata_requires_complete_known_value(self):
        # Pin comparison checks the entire value, including every qualifier text.
        text = "Reject unless every extra relation key is explicitly enforced; never discard constraints."
        record = self.scalar_pair()
        record["consumer_requirement"] = text
        self.assertTrue(pairs.constraints_supported(record))
        record["consumer_requirement"] += " except addresses"
        self.assertFalse(pairs.constraints_supported(record))

    def test_model_only_and_nonboolean_confirmations_never_admit(self):
        for proof in (None, {}, {"confirmed": False}):
            record = pair()
            record["verilator"] = proof
            self.assertEqual(self.select(record), (None, "unconfirmed"))
        for value in ("true", 1, None):
            record = pair()
            record["verilator"]["confirmed"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "literal boolean"):
                self.select(record)

    def test_minimum_one_uses_inexpressible_zero_and_preserves_extended_domain(self):
        record = pair()
        record.update(min_gap=1, swept=[1, 132], negative={"gap": 0, "result": "inexpressible"})
        selected, _ = self.select(record)
        self.assertEqual(selected["swept"], [1, 132])
        record["negative"]["result"] = "conflict"
        with self.assertRaisesRegex(ValueError, "inexpressible"):
            self.select(record)

    def test_bad_gap_domain_negative_or_per_op_boundary_fails(self):
        changes = ({"swept": [1, 94]}, {"swept": [32, 1]}, {"min_gap": True}, {"first_live_window_end": 94},
                   {"negative": {"gap": 30, "result": "conflict", "error": {"message": "busy"}}},
                   {"negative": {"gap": 31, "result": "numerical_mismatch"}},
                   {"negative": {"gap": 31, "result": "conflict", "error": {}}})
        for change in changes:
            record = pair()
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(record)

    def test_complete_chain_has_exact_vector_matrix_and_post_progress_peak(self):
        record = chain()
        projected, reason = self.select(record, "chains")
        self.assertIsNone(reason)
        for key in ("operations", "gaps", "all_pair_relations", "in_flight_count", "witness_only", "no_stable_suffix_claim"):
            self.assertEqual(projected[key], record[key])
        record["operations"].append("mxu0.matmul")
        record["gaps"].append(32)
        record["all_pair_relations"] = [dict(first_index=i, second_index=j, relation=deepcopy(RELATION))
                                         for i in range(4) for j in range(i + 1, 4)]
        self.assertIsNone(self.select(record, "chains")[1])  # Four commands, peak three.
        record["in_flight_count"] = 4
        with self.assertRaisesRegex(ValueError, "post-progress"):
            self.select(record, "chains")

    def test_malformed_chain_matrix_gap_count_and_flags_fail(self):
        changes = ({"gaps": [0, 32]}, {"gaps": [32]}, {"in_flight_count": 2}, {"witness_only": "true"},
                   {"no_stable_suffix_claim": False}, {"fixed_prefix_gaps": [31]})
        for change in changes:
            record = chain()
            record.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(record, "chains")
        for matrix in (chain()["all_pair_relations"][:-1], chain()["all_pair_relations"] * 2):
            record = chain()
            record["all_pair_relations"] = matrix
            with self.assertRaises(ValueError):
                self.select(record, "chains")

    def test_streaming_reader_hashes_whole_source_with_tiny_chunks(self):
        value = dict(schema="atlas.model_pairs.v1", model={}, age_convention="issue_edge_zero", pairs=[pair()], chains=[chain()])
        data = json.dumps(value).encode() + b"\n  \n"
        reader = pairs.RecordReader(SmallChunks(data))
        records = list(reader.records())
        self.assertEqual(next(value for key, index, value in records if key == "pairs"), pair())
        self.assertEqual(reader.digest.hexdigest(), hashlib.sha256(data).hexdigest())
        for invalid in (data[:-20], data + b"garbage"):
            with self.subTest(invalid=invalid[-20:]), self.assertRaises(ValueError):
                list(pairs.RecordReader(SmallChunks(invalid)).records())

    def test_wrong_input_pin_and_projection_tamper_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projection.json"
            path.write_text('{"pairs": []}')
            expected = pairs.sha256(path.read_bytes())
            self.assertEqual(pairs.read_pinned(path, expected), {"pairs": []})
            for wrong in ("bad", "f" * 64):
                with self.subTest(wrong=wrong), self.assertRaisesRegex(ValueError, "SHA-256"):
                    pairs.read_pinned(path, wrong)
            path.write_text('{"pairs": [1]}')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                pairs.read_pinned(path, expected)

    def test_unknown_constraint_values_are_excluded_without_stripping(self):
        for fields in ({"in_flight_scope": "universal capacity"}, {"relation_semantics": {}},
                       {"producer_relation_summary": {"unreviewed": "same"}},
                       {"operand_order": {"first": [64], "second": [0]}}):
            record = chain()
            record.update(fields)
            with self.subTest(fields=fields):
                self.assertEqual(self.select(record, "chains"), (None, "unsupported_constraint_value"))

    def test_chain_requires_each_live_pair_but_not_drained_pair_coverage(self):
        record = chain()
        first = pair()
        other = dict(first, relation=dict(RELATION, accumulator="same"))
        index = {(first["first"], first["second"], pairs.canonical(first["relation"])): [first],
                 (other["first"], other["second"], pairs.canonical(other["relation"])): [other]}
        self.assertTrue(pairs.covered_chain(record, index, BLOCKS))
        del index[(other["first"], other["second"], pairs.canonical(other["relation"]))]
        self.assertFalse(pairs.covered_chain(record, index, BLOCKS))
        record["gaps"] = [95, 95]
        self.assertTrue(pairs.covered_chain(record, {}, BLOCKS))

    def test_chain_axis_bounds_remain_bound_to_exact_recorded_vector(self):
        record = chain()
        record.update(fixed_prefix_gaps=[32], swept_last_gap=[1, 95], minimum_last_gap=32)
        self.assertIsNone(self.select(record, "chains")[1])
        for fields in ({"swept_last_gap": [1, 31]}, {"minimum_last_gap": 33},
                       {"swept": [95, 1]}, {"swept": [1, 95], "min_gap": True}):
            changed = dict(record, **fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.select(changed, "chains")

    def project_fixture(self, directory, *, change_master=None, change_validation=None, change_per_op=None):
        config = {"ignore_runtime_errors": False}
        names = list(BLOCKS) + ["unused." + str(i) for i in range(49 - len(BLOCKS))]
        operations = dict(schema="atlas.model_timing.v1", validation={"passed": True},
                          model={"revision": pairs.MODEL_REVISION, "configuration": config},
                          op_timing=[dict(name=name, **BLOCKS.get(name, {"first_free_age": 1})) for name in names])
        master = dict(schema="atlas.model_pairs.v1", age_convention="issue_edge_zero",
                      model={"revision": pairs.MODEL_REVISION, "configuration": config,
                             "configuration_sha256": pairs.sha256(pairs.canonical(config))}, pairs=[pair()], chains=[])
        if change_master:
            change_master(master)
        if change_per_op:
            change_per_op(operations)
        master_bytes, per_op_bytes = pairs.canonical(master), pairs.canonical(operations)
        master_sha, per_op_sha = pairs.sha256(master_bytes), pairs.sha256(per_op_bytes)
        validation = dict(passed=True, facts_sha256=master_sha, pairs=1, chains=0)
        if change_validation:
            change_validation(validation)
        validation_bytes = pairs.canonical(validation)
        for name, data in (("master", master_bytes), ("per-op", per_op_bytes), ("validation", validation_bytes)):
            (directory / (name + ".json")).write_bytes(data)
        with ExitStack() as stack:
            for key, value in (("MASTER_SHA256", master_sha), ("PER_OP_SHA256", per_op_sha),
                               ("VALIDATION_SHA256", pairs.sha256(validation_bytes)),
                               ("CONFIGURATION_SHA256", pairs.sha256(pairs.canonical(config)))):
                stack.enter_context(patch.object(pairs, key, value))
            return pairs.project(directory / "master.json", directory / "validation.json", directory / "per-op.json",
                                 directory / "export")

    def test_complete_projection_binds_receipt_source_records_and_owned_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            receipt = self.project_fixture(directory)
            artifact = directory / "export/model-pairs.json"
            projected = pairs.read_pinned(artifact, receipt["facts_sha256"])
            self.assertTrue(receipt["passed"])
            self.assertEqual(projected["projection"]["policy"], pairs.POLICY)
            self.assertEqual(projected["projection"]["counts"], dict(pairs=1, chains=0, source_pairs=1, source_chains=0))
            self.assertEqual(projected["pairs"][0]["source_record"]["sha256"], pairs.sha256(pairs.canonical(pair())))
            self.assertEqual(projected["projection"]["projector_sha256"], pairs.sha256(Path(pairs.__file__).read_bytes()))

    def test_invalid_bound_inputs_produce_no_admissible_output(self):
        mutations = (dict(change_master=lambda m: m.update(schema="other")),
                     dict(change_validation=lambda v: v.update(passed=False)),
                     dict(change_validation=lambda v: v.update(facts_sha256="0" * 64)),
                     dict(change_validation=lambda v: v.update(pairs=2)),
                     dict(change_per_op=lambda p: p["model"].update(configuration={"ignore_runtime_errors": True})))
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                with self.assertRaises(ValueError):
                    self.project_fixture(directory, **mutation)
                self.assertFalse((directory / "export/model-pairs.json").exists())

    def test_safe_paths_fail_before_prohibited_or_symlink_lookup(self):
        for path in ("untrusted/../facts.json", "untrusted/HaMmEr-data/facts.json", "untrusted/vLsI.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                pairs.safe_path(path)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "target"
            target.write_text("{}")
            link = Path(directory) / "alias"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                pairs.safe_path(link)


if __name__ == "__main__":
    unittest.main()
