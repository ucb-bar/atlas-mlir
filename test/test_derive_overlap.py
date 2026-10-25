from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import atlas_derive_overlap as derive


def event(first, last):
    return dict(first_age=first, last_age=last, count=last-first+1, step=1)


def release(available):
    return dict(first_age=0, release_age=available-1, available_age=available)


def fixtures():
    blocks, releases = [], {}
    for unit, first, depth in (("mxu0", 63, 3), ("mxu1", 3, 2)):
        for spec in derive.VOCABULARY["mxu_blocks"]:
            name = unit + "." + spec["block"]
            kind = derive.SourceRules.kind(name)
            pop, compute = kind.startswith("pop"), kind == "compute"
            end = first + 32 if compute else 33
            events = {}
            if pop:
                events["acc_store"] = event(0, 31)
                for index in range(2 if kind == "pop_bf16" else 1):
                    events[f"mreg_write{index}"] = event(1, 32)
            else:
                keys = ("mreg_read0",) if compute else ("mreg_read0", "mreg_read1") if kind == "push_bf16" else ("mreg_read1",)
                events.update({key: event(0, 31) for key in keys})
                events["acc_write" if compute else "weight_write" if kind == "weight" else "acc_load"] = event(first, first+31) if compute else event(1, 32)
                if name.endswith("matmul_acc"):
                    events["acc_read"] = event(0, 31)
            blocks.append(dict(name=name, resolved=True, first_free_age=end, events=events))
            releases[name] = {"mreg_write" if pop else "mreg_read": release(33)}
    for part in ("relu", "exp2", "fp8pack"):
        name = "vpu." + part
        writes = dict(first_age=3, last_age=65, step=2, count=32) if part == "fp8pack" else event(2, 65)
        blocks.append(dict(name=name, resolved=True, first_free_age=66, events=dict(read0=event(0, 63), write0=writes)))
        releases[name] = dict(mreg_read=release(64), mreg_write=release(66))
    facts = dict(schema="atlas.model_timing.v1", age_convention="issue_edge_zero", model={},
                 op_timing=blocks, reservation_releases=releases)
    contract = dict(capacity=dict(sequencer_cycles=32, mxu0_compute=3, mxu1_compute=2, vpu_slots=2),
                    vpu=dict(logic_groups=[], double=[]),
                    mxu=dict(compute_first=dict(mxu0=63, mxu1=3), compute_last=dict(mxu0=94, mxu1=34),
                             acc_push_last=32, mxu0_weight_drain_last=62, accumulator_forward_gap=2,
                             ports=dict(compute=[0], push_bf16=[0, 1], pop_fp8=[2], pop_bf16=[2, 3]),
                             single_push_priority=[1, 0]), scalar_lsu=dict(load_response_gap=3))
    return facts, contract


class DeriveOverlapTests(unittest.TestCase):
    def test_complete_single_register_relations(self):
        domain = derive.mreg_domain("mxu0.matmul", "mxu0.push_weight")
        self.assertEqual(set(domain), {"disjoint", "0=0", "0~0"})

    def test_bf16_domains_preserve_consumer_pair_alignment(self):
        domain = derive.mreg_domain("mxu0.push_acc_bf16", "mxu0.pop_acc_bf16")
        self.assertEqual(set(domain), {"disjoint", "0=0,1=1", "0~0,1~1"})
        for a, b, *_ in domain.values():
            self.assertEqual(a["rs1"] % 2, 0)
            self.assertEqual(b["rd"] % 2, 0)

    def test_slot_relations_only_for_present_resources(self):
        rows = list(derive.pair_domain("mxu0.push_weight", "mxu0.matmul"))
        self.assertEqual(len(rows), 6)
        for relation, _, _ in rows:
            self.assertEqual(set(relation), {"mreg_banks", "weight_slot"})
        rows = list(derive.pair_domain("mxu0.matmul", "mxu0.matmul_acc"))
        self.assertEqual(len(rows), 12)
        self.assertEqual({(r["accumulator"], r["weight_slot"]) for r, *_ in rows},
                         {("same", "same"), ("same", "other"), ("other", "same"), ("other", "other")})

    def test_minimal_vector_pair_domain_keeps_disjoint_source_legal_operands(self):
        for first, second in [("vpu.relu", "vpu.relu"), ("vpu.fp8pack", "vpu.exp2")]:
            rows = list(derive.pair_domain(first, second))
            self.assertEqual(len(rows), 1)
            relation, operands, assembly = rows[0]
            self.assertEqual(relation, {"mreg_banks": "disjoint"})
            self.assertEqual(len(assembly), 2)
            self.assertFalse(set(operands[0]["registers"]) & set(operands[1]["registers"]))

    def test_retired_prefix_samples_keep_source_gaps(self):
        facts, contract = fixtures()
        rules = derive.SourceRules(facts, contract)
        samples = [sample for sample in derive.chain_samples(rules)
                   if sample["mechanism"] == "retired_prefix_read_sequencer_assignment"]
        self.assertEqual(len(samples), 4)
        for sample in samples:
            with self.subTest(sample=sample["id"]):
                first = sample["operations"][0]
                weight_issue, gap = sample["gaps"]
                compute_issue = weight_issue + gap
                self.assertGreaterEqual(compute_issue, rules.first_free_age(first))
                self.assertEqual(sample["gaps"], [2, 32 if sample["expectation"] == "positive" else 31])


if __name__ == "__main__":
    unittest.main()
