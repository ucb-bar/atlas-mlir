"""Source engine predicates and unsupported source refusals."""
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import atlas_derive_admission as derive
from model_evidence import isolate_model_imports


class VpuLifetimeTest(unittest.TestCase):
    def test_native_live_endpoint_may_follow_the_last_row_write(self):
        blocks = {"vpu.rsum": dict(engine="vpu", first_free_age=40,
            events=dict(write0=dict(last_age=38), write1=dict(last_age=38)))}
        plans = [("vpu.rsum", "vredsum.row.bf16", None, SimpleNamespace(write_last=39))]
        with patch("atlas_derive_vpu._plans", return_value=plans):
            final = derive._vpu_live_ages(None, blocks)
        self.assertEqual(final, {"vpu.rsum": 39})
        for changed in (dict(blocks["vpu.rsum"], first_free_age=41),
                        dict(blocks["vpu.rsum"], events=dict(write0=dict(last_age=40)))):
            with patch("atlas_derive_vpu._plans", return_value=plans), self.assertRaises(derive.UnsupportedAdmission):
                derive._vpu_live_ages(None, {"vpu.rsum": changed})


@unittest.skipUnless(os.environ.get("ATLAS_DERIVE_MODEL_ROOT"), "set ATLAS_DERIVE_MODEL_ROOT for engine source/native contracts")
class NativeEngineAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Path(os.environ["ATLAS_DERIVE_MODEL_ROOT"])
        isolate_model_imports(cls, str(cls.model))
        import torch
        torch.set_num_threads(1)
        from atlas_model_timing import configuration, source_digest, numerical_data, encoded, sha256
        from atlas_derive_vpu import derive_vpu
        from atlas_derive_mxu import derive_mxu
        from atlas_derive_memory import derive_memory
        cfg, config = configuration()
        vpu, vr = derive_vpu(cfg)
        mxu, mr = derive_mxu(cfg)
        memory, lr = derive_memory(cfg)
        cls.facts = dict(schema="atlas.model_timing.v1", age_convention="issue_edge_zero",
            model=dict(source_sha256=source_digest(cls.model)[0],
                configuration_sha256=sha256(encoded(config)), numerical_data_sha256=numerical_data(cls.model)[0],
                configuration=config), op_timing=list((vpu | mxu | memory).values()), reservation_releases=vr | mr | lr)
        cls.admission = derive.derive_admission(cls.model, cls.facts)

    def test_construction_or_routing_refactor_refuses_before_publication(self):
        original = derive._source
        def changed(root, name):
            text, tree = original(root, name)
            if name == "core.py":
                import ast
                text = text.replace("self.exu_map = {EXU(type(exu).__name__): exu for exu in self.exus}",
                                    "self.exu_map = {}")
                tree = ast.parse(text)
            return text, tree
        with patch.object(derive, "_source", side_effect=changed), self.assertRaisesRegex(ValueError, "construction/routing"):
            derive.derive_admission(self.model, self.facts)
