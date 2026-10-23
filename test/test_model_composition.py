from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from atlas_model_composition import RuleSet, negative_points, relation_class
from atlas_model_chain_composition import compose, negative_vectors
import atlas_model_composed_projection as projection
from atlas_model_composed_projection import complete_relations


def event(first, count, step=1):
    return dict(first_age=first, last_age=first+(count-1)*step, count=count, step=step)


def reservation(available):
    return dict(first_age=0, release_age=available-1, available_age=available)


def model_rules(blocks, releases):
    return RuleSet(dict(schema="atlas.model_timing.v1", age_convention="issue_edge_zero", model={},
                        op_timing=blocks, reservation_releases=releases))


def rules():
    blocks, releases = [], {}
    for name in ("relu", "exp2", "fp8pack"):
        write = event(3, 32, 2) if name == "fp8pack" else event(2, 64)
        blocks.append(dict(name="vpu."+name, resolved=True, first_free_age=66,
                           events=dict(read0=event(0, 64), write0=write)))
        releases["vpu."+name] = dict(mreg_read=reservation(64), mreg_write=reservation(66))
    for name in ("push_weight", "push_acc_fp8", "matmul"):
        events = (dict(mreg_read0=event(0, 32), acc_write=event(63, 32)) if name == "matmul" else
                  dict(mreg_read1=event(0, 32), **{("weight_write" if name == "push_weight" else "acc_load"): event(1, 32)}))
        blocks.append(dict(name="mxu0."+name, resolved=True, first_free_age=95 if name == "matmul" else 33,
                           events=events))
        releases["mxu0."+name] = dict(mreg_read=reservation(33))
    return model_rules(blocks, releases)


def chain(names, gaps, relations=None):
    return dict(operations=names, gaps=gaps, all_pair_relations=[
        dict(first_index=i, second_index=j, relation=(relations or {}).get((i, j), dict(mreg_banks="disjoint")))
        for i in range(len(names)) for j in range(i+1, len(names))])


class CompositionTests(unittest.TestCase):
    def test_composed_projection_binds_base_master_before_holdouts(self):
        per_op_sha = projection.PER_OP_SHA256
        master_sha, rule_sha = "a" * 64, "c" * 64
        documents = {
            "per-op": dict(model=dict(revision=projection.MODEL_REVISION,
                                      configuration_sha256=projection.CONFIGURATION_SHA256)),
            "pairs": dict(projection=dict(master_sha256=master_sha, per_op_sha256=per_op_sha)),
            "pair-audit": dict(master_sha256=master_sha, per_operation_sha256=per_op_sha,
                               rule_source_sha256=rule_sha),
            "chain-audit": dict(master_sha256=master_sha, per_op_sha256=per_op_sha,
                                rule_source_sha256=rule_sha),
            "queries": {}, "holdouts": dict(passed=False, results=[]),
        }
        with patch.object(projection, "load", side_effect=documents.__getitem__), \
                patch.object(projection, "sha", return_value=per_op_sha) as digest, \
                patch.object(projection, "rule_source_sha256", return_value=rule_sha):
            for base_master, diagnostic in ((master_sha, "successful holdouts are required"),
                                            ("b" * 64, "replay does not bind selected observations")):
                with self.subTest(base_master=base_master):
                    documents["pairs"]["projection"]["master_sha256"] = base_master
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        projection.project("per-op", "pairs", "pair-audit", "chain-audit",
                                           "queries", "holdouts", "unused-output")

            derived_sha = "d" * 64
            digest.return_value = derived_sha
            documents["pairs"]["projection"].update(master_sha256=master_sha, per_op_sha256=derived_sha)
            documents["pair-audit"]["per_operation_sha256"] = derived_sha
            documents["chain-audit"]["per_op_sha256"] = derived_sha
            for pin, diagnostic in ((None, "unsupported per-operation evidence"),
                                    ("e" * 64, "unsupported per-operation evidence"),
                                    (derived_sha, "successful holdouts are required")):
                with self.subTest(per_op_pin=pin), self.assertRaisesRegex(ValueError, diagnostic):
                    kwargs = {} if pin is None else dict(per_op_sha256=pin)
                    projection.project("per-op", "pairs", "pair-audit", "chain-audit", "queries", "holdouts",
                                       "unused-output", **kwargs)

    def test_register_lifetimes_follow_selected_reservation_releases(self):
        for name, events, available in (
                ("mxu0.matmul", dict(mreg_read0=event(0, 32)), 40),
                ("xlu.transpose", dict(read=event(0, 32), write=event(35, 32)), 41)):
            with self.subTest(name=name):
                block = dict(name=name, resolved=True, first_free_age=80, events=events)
                releases = {name: dict(mreg_read=reservation(available), mreg_write=reservation(80))}
                model = model_rules([block], releases)
                self.assertEqual(model.registers(name)[0].release, available)
                self.assertEqual(model.registers(name)[0].requests[-1], (31, 31))

    def test_missing_or_early_register_releases_fail_closed(self):
        name = "mxu0.matmul"
        block = dict(name=name, resolved=True, first_free_age=80, events=dict(mreg_read0=event(0, 32)))
        for releases in ({}, {name: {}}, {name: dict(mreg_read=reservation(31))},
                         {name: dict(mreg_read=dict(first_age=0, release_age=32, available_age=True))}):
            with self.subTest(releases=releases):
                model = model_rules([block], releases)
                with self.assertRaisesRegex(ValueError, "mreg_read reservation release"):
                    model.registers(name)

    def test_port_allocation_is_not_total_capacity(self):
        model = rules()
        record = chain(["mxu0.push_weight", "mxu0.push_acc_fp8", "mxu0.matmul"], [1, 1],
                       {(0, 2): dict(mreg_banks="disjoint", weight_slot="other"),
                        (1, 2): dict(mreg_banks="disjoint", accumulator="other")})
        result = compose(record, record["gaps"], model, model)
        self.assertEqual(result["rejected_pairs"], [])
        self.assertEqual(result["peak_post_progress"], 3)
        self.assertFalse(result["admitted"])
        self.assertIn("port", result["capacity_conflict"])
        self.assertTrue(compose(record, [1, 32], model, model)["admitted"])

    def test_vpu_final_write_frees_slot_but_still_needs_pair(self):
        model = rules()
        record = chain(["vpu.relu", "vpu.exp2", "vpu.fp8pack"], [1, 63])
        self.assertFalse(compose(record, [1, 63], model, model)["admitted"])
        self.assertTrue(compose(record, [1, 64], model, model)["admitted"])
        self.assertEqual(model.minimum_gap("vpu.relu", "vpu.fp8pack", dict(mreg_banks="2=0,3=1")), 66)
        record["all_pair_relations"][1]["relation"] = dict(mreg_banks="2=0,3=1")
        self.assertFalse(compose(record, [1, 64], model, model)["admitted"])

    def test_relation_matrix_must_be_complete_and_unique(self):
        model = rules()
        record = chain(["vpu.relu", "vpu.exp2", "vpu.fp8pack"], [1, 65])
        record["all_pair_relations"][-1] = record["all_pair_relations"][0]
        self.assertFalse(compose(record, record["gaps"], model, model)["admitted"])
        record["all_pair_relations"].pop()
        self.assertFalse(compose(record, record["gaps"], model, model)["admitted"])

    def test_every_nested_pair_negative_including_model_stricter(self):
        record = dict(negative=dict(gap=2, result="conflict"), verilator=dict(
            model_one_short_guard=dict(gap=1, result="conflict"),
            additional_confirmations=[dict(negative_basis=dict(gap=0, kind="inexpressible"))],
            profile=dict(one_short=dict(gap=1, classification="MODEL_STRICTER_ONE_SHORT_CONTROL",
                                        model_status="EXCEPTION", passed=True)),
            passing_shorter_planning_observation=dict(gap=1, result="accepted", is_negative=False)))
        values = list(negative_points(record))
        self.assertEqual(len(values), 4)
        self.assertEqual([value[1] for value in values], [2, 1, 0, 1])

    def test_chain_review_controls_and_rtl_negatives(self):
        record = chain(["vpu.relu", "vpu.exp2", "vpu.fp8pack"], [1, 64])
        record["model_negative_review"] = dict(controls=[
            dict(negative_gaps=[1, 63], error=dict(message="slot capacity"))])
        record["verilator"] = dict(negative=dict(gaps=[0, 64], result="inexpressible"))
        values = list(negative_vectors(record))
        self.assertEqual({tuple(v["gaps"]) for v in values}, {(1, 63), (0, 64)})

    def test_uniform_negative_requires_declared_sweep(self):
        record = chain(["vpu.relu"]*3, [65, 65])
        record["negative"] = dict(gap=64, result="conflict")
        self.assertTrue(list(negative_vectors(record))[0]["unresolved"])
        record.update(swept=[1, 66], min_gap=65)
        self.assertEqual(list(negative_vectors(record))[0]["gaps"], [64, 64])
        record["verilator"] = {key: record[key] for key in ("operations", "gaps", "all_pair_relations")}
        record["verilator"]["negative"] = dict(gap=64, result="conflict")
        self.assertTrue(all(value["gaps"] == [64, 64] for value in negative_vectors(record)))

    def test_relation_family_keeps_slot_and_alias_constraints(self):
        self.assertEqual(relation_class(dict(mreg_banks="2~3")), dict(mreg_banks="bank_alias"))
        self.assertEqual(relation_class(dict(mreg_banks="0=0,1~1", accumulator="same")),
                         dict(mreg_banks="mixed_alias", accumulator="same"))
        with self.assertRaises(ValueError):
            rules().minimum_gap("vpu.relu", "vpu.exp2", dict(mreg_banks="invalid"))

    def test_new_workload_pair_minima_and_one_short_rejections(self):
        model = rules()
        cases = [("relu", "relu", "0=2,1=3", 65), ("fp8pack", "relu", "2~3", 32),
                 ("relu", "fp8pack", "1~2", 1), ("fp8pack", "relu", "2~1", 1),
                 ("exp2", "fp8pack", "2~2", 31), ("fp8pack", "exp2", "2~2", 64),
                 ("relu", "fp8pack", "3~2", 63)]
        for first, second, banks, minimum in cases:
            with self.subTest(first=first, second=second, banks=banks):
                first, second, relation = "vpu."+first, "vpu."+second, dict(mreg_banks=banks)
                self.assertEqual(model.minimum_gap(first, second, relation), minimum)
                self.assertTrue(model.admits_pair(first, second, relation, minimum))
                self.assertFalse(model.admits_pair(first, second, relation, minimum-1))

    def test_holdout_relation_matrix_must_match_actual_operands(self):
        query = dict(operations=["vpu.relu", "vpu.fp8pack"], gaps=[63],
                     operands=[dict(registers=[0, 1, 4, 5]), dict(registers=[8, 9, 37])])
        result = complete_relations(query)
        self.assertEqual(result["all_pair_relations"][0]["relation"], dict(mreg_banks="3~2"))
        result["all_pair_relations"][0]["relation"] = dict(mreg_banks="disjoint")
        with self.assertRaises(ValueError):
            complete_relations(result)

    def test_vector_memory_address_qualifiers_preserve_bank_constraints(self):
        blocks = [dict(name="vlsu."+kind, resolved=True, first_free_age=35, events={
            ("mreg_write" if kind == "vload" else "mreg_read"): event(1, 32),
            ("vmem_read" if kind == "vload" else "vmem_write"): event(1, 32)})
            for kind in ("vload", "vstore")]
        releases = {"vlsu."+kind: {("mreg_write" if kind == "vload" else "mreg_read"): reservation(35)}
                    for kind in ("vload", "vstore")}
        model = model_rules(blocks, releases)
        relation = dict(mreg_banks="disjoint", vmem_bank="other", vmem_window="disjoint",
                        vmem_address_delta_bytes="261120", first_vmem_bank_byte_offset="1024",
                        second_vmem_bank_byte_offset="0")
        self.assertEqual(model.minimum_gap("vlsu.vload", "vlsu.vstore", relation), 1)
        relation["vmem_bank"] = "same"
        with self.assertRaises(ValueError):
            model.minimum_gap("vlsu.vload", "vlsu.vstore", relation)


if __name__ == "__main__":
    unittest.main()
