"""Fail-closed metadata controls; these do not represent model executions."""
from pathlib import Path
import copy
import ast
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest import mock
import prepare_program as preparation


class ReusedEvidenceAdmission(unittest.TestCase):
    def test_control_payload_accepts_each_known_name_and_refuses_ambiguity(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        names = ("actual-bf16-tiles-0.bin", "actual-bf16-halves-0.bin")
        for present in ((), names, (names[0],), (names[1],)):
            with self.subTest(present=present), mock.patch.object(Path, "exists", lambda path: path.name in present):
                if len(present) != 1:
                    with self.assertRaises(ValueError):preparation.control_payload_path(support, Path("/unread"))
                else:
                    self.assertEqual(preparation.control_payload_path(support, Path("/unread")).name, present[0])

    def test_only_fixed_producer_profiles_are_admitted(self):
        epochs = sorted(preparation.ARITHMETIC_EPOCHS)
        for epoch in epochs:
            code = dict(zip(preparation.PRODUCER_NAMES, epoch))
            self.assertEqual(preparation.producer_epoch({"code_before": code, "code_after": code.copy()}), epoch)
        mixed = dict(zip(preparation.PRODUCER_NAMES, (epochs[0][0], epochs[1][1], epochs[0][2])))
        with self.assertRaisesRegex(ValueError, "Unknown or mixed"):
            preparation.producer_epoch({"code_before": mixed, "code_after": mixed.copy()})

    def test_code_bookend_changes_are_refused(self):
        code = dict(zip(preparation.PRODUCER_NAMES, next(iter(preparation.ARITHMETIC_EPOCHS))))
        changed = dict(code, atlas_tile_arithmetic="unrecognized")
        with self.assertRaises(ValueError):
            preparation.producer_epoch({"code_before": code, "code_after": changed})

    def test_reuse_controls_or_missing_model_root_refuse_before_io(self):
        for controls, root in ((True, Path("/unread")), (False, None)):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                preparation.prepare(None, None, None, None, None, controls,
                                    reuse_model_arithmetic=True, model_root=root)

    def test_ambiguous_actual_payload_names_are_refused_before_contents(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", return_value=True), \
                mock.patch.object(Path, "read_bytes") as contents:
            with self.assertRaisesRegex(ValueError, "exactly one"):
                preparation.model_payload_path(support, Path("/unread"), {})
            contents.assert_not_called()

    def test_actual_payload_requires_a_declared_artifact_hash(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", lambda path: path.name == "actual-bf16-tiles-0.bin"), \
                mock.patch.object(Path, "read_bytes") as contents:
            with self.assertRaises(ValueError):
                preparation.model_payload_path(support, Path("/unread"), {})
            contents.assert_not_called()

    def test_ambiguous_declared_names_refuse_even_if_one_file_is_missing(self):
        support = SimpleNamespace(safe_path=lambda path: path)
        with mock.patch.object(Path, "exists", lambda path: path.name == "actual-bf16-tiles-0.bin"):
            with self.assertRaises(ValueError):
                preparation.model_payload_path(support, Path("/unread"),
                                               {"actual-bf16-tiles-0.bin": "old", "actual-bf16-halves-0.bin": "new"})

    def test_foreign_backend_numerics_or_input_mode_refuse_before_model_access(self):
        root = Path("/unread")
        code = dict(zip(preparation.PRODUCER_NAMES, next(iter(preparation.ARITHMETIC_EPOCHS))))
        before = {"revision": preparation.MODEL_PIN, "tracked_source_diff_exit": 0,
                  "imported_source_sha256": {"npu_model/__init__.py": "unread"}, "curve_sha256": "unread"}
        inputs = {"input": "unread", "manifest": "unread"}
        metadata = {"code_before": code, "code_after": code.copy(), "model_binding_before": before,
                    "invocations": "one_reset_core_per_repeat_all_tiles_in_one_instruction_program",
                    "source_inputs": inputs,
                    "source_bookends": {"unchanged": True, "model_after": before, "input_before": inputs, "input_after": inputs},
                    "model_input": "new_assembler_decoded_instruction_objects",
                    "model": {"revision": preparation.MODEL_PIN, "module_file": str(root / "npu_model/__init__.py"),
                              "imported_source_sha256": before["imported_source_sha256"], "curve_sha256": "unread",
                              "backend": "curve", "numerics": "rtl"}}
        for key, value in (("backend", "foreign"), ("numerics", "foreign"), ("model_input", "foreign")):
            changed = copy.deepcopy(metadata)
            if key == "model_input":changed[key] = value
            else:changed["model"][key] = value
            with self.subTest(key=key), mock.patch("subprocess.check_output") as access:
                with self.assertRaisesRegex(ValueError, "module/curve/source"):
                    preparation.reused_arithmetic(SimpleNamespace(safe_path=lambda path: path), Path("/unread"), changed, b"", root)
                access.assert_not_called()

    def receipt(self, epoch, mode="none", shape=(64, 64, 64)):
        m, k, n = shape
        code = dict(zip(preparation.PRODUCER_NAMES, epoch))
        flags = {"none": [], "resident_a_v1": ["--resident-a"],
                 "resident_b_v1": ["--resident-b"],
                 "resident_b_pretransposed_v1": ["--resident-b", "--pretranspose-resident-b"]}
        value = {"code_before": code, "code_after": code.copy(), "command": ["producer.py"] + flags[mode]}
        if mode != "none":
            operand = "a" if mode == "resident_a_v1" else "b"
            value.update(resident_operand_reuse_claim=True, operand_reuse={"mode": mode,
                         "resident_" + operand + "_bytes": m * k if operand == "a" else k * n,
                         "resident_" + operand + "_vmem_byte_base": 0x40000})
        elif epoch == preparation.RELEASE_EPOCH:
            value.update(resident_operand_reuse_claim=False,
                         operand_reuse={"mode": "none", "resident_b_bytes": 0, "resident_b_vmem_byte_base": None})
        return value

    def test_current_release_and_historical_mode_traffic(self):
        optimized = (preparation.RESIDENT_A_EPOCH, preparation.RESIDENT_B_EPOCH,
                     preparation.PRETRANSPOSED_B_EPOCH, preparation.RELEASE_EPOCH)
        baseline = preparation.ARITHMETIC_EPOCHS - set(optimized)
        self.assertEqual(len(baseline), 2)
        for epoch in baseline:
            for claim in (None, False):
                old = self.receipt(epoch)
                old['resident_operand_reuse_claim'] = claim
                self.assertEqual(preparation.requested_traffic(old, 64, 64, 64),
                                 {"load_bytes": 16384, "store_bytes": 8192})
        modes = ("none", "resident_a_v1", "resident_b_v1", "resident_b_pretransposed_v1")
        for mode in modes:
            expected = 16384 if mode == "none" else 12288
            self.assertEqual(preparation.requested_traffic(self.receipt(preparation.RELEASE_EPOCH, mode), 64, 64, 64),
                             {"load_bytes": expected, "store_bytes": 8192})
        for epoch, qualified in zip(optimized[:3], modes[1:]):
            for mode in modes:
                value = self.receipt(epoch, mode)
                if mode == qualified:
                    self.assertEqual(preparation.requested_traffic(value, 64, 64, 64)["load_bytes"], 12288)
                else:
                    with self.assertRaises(ValueError): preparation.requested_traffic(value, 64, 64, 64)
        for epoch in baseline:
            for mode in modes[1:]:
                with self.assertRaises(ValueError): preparation.requested_traffic(self.receipt(epoch, mode), 64, 64, 64)

    def test_asymmetric_workload_traffic_and_bank_boundaries(self):
        cases = [((3136, 64, 64), "resident_b_v1", 405504, 401408),
                 ((3136, 576, 64), "resident_b_pretransposed_v1", 3649536, 401408),
                 ((3136, 64, 256), "resident_b_v1", 1622016, 1605632),
                 ((192, 960, 960), "resident_a_v1", 5713920, 368640),
                 ((192, 960, 320), "resident_a_v1", 2027520, 122880),
                 ((128, 2048, 2048), "resident_a_v1", 17039360, 524288),
                 ((128, 2048, 256), "resident_a_v1", 2359296, 65536)]
        for shape, mode, loads, stores in cases:
            with self.subTest(shape=shape, mode=mode):
                self.assertEqual(preparation.requested_traffic(self.receipt(preparation.RELEASE_EPOCH, mode, shape), *shape),
                                 {"load_bytes": loads, "store_bytes": stores})
        shape = (32, 1024, 256)
        preparation.requested_traffic(self.receipt(preparation.RELEASE_EPOCH, "resident_b_v1", shape), *shape)
        for shape, mode in [((160, 2048, 256), "resident_a_v1"), ((32, 1056, 256), "resident_b_v1"),
                            ((32, 1056, 256), "resident_b_pretransposed_v1")]:
            with self.assertRaises(ValueError):
                preparation.requested_traffic(self.receipt(preparation.RELEASE_EPOCH, mode, shape), *shape)

    def test_reuse_flags_refuse_aliases_duplicates_and_conflicts(self):
        for mode in ("none", "resident_a_v1", "resident_b_v1", "resident_b_pretransposed_v1"):
            good = self.receipt(preparation.RELEASE_EPOCH, mode)
            for extra in ("--resident-a", "--resident-b", "--pretranspose-resident-b", "--resident-a=true",
                          "--resident-b=false", "--resident-broken", "--pretranspose-resident-b=true"):
                bad = copy.deepcopy(good); bad['command'].append(extra)
                with self.subTest(mode=mode, extra=extra), self.assertRaises(ValueError):
                    preparation.requested_traffic(bad, 64, 64, 64)
        for command in (None, "--resident-a", ["producer.py", 1]):
            bad = self.receipt(preparation.RELEASE_EPOCH); bad['command'] = command
            with self.assertRaises(ValueError): preparation.requested_traffic(bad, 64, 64, 64)

    def test_reuse_markers_and_claims_have_exact_types_and_keys(self):
        for mode in ("none", "resident_a_v1", "resident_b_v1", "resident_b_pretransposed_v1"):
            good = self.receipt(preparation.RELEASE_EPOCH, mode)
            for key, value in (("operand_reuse", None), ("operand_reuse", {}), ("operand_reuse", {"mode": "unknown"}),
                               ("resident_operand_reuse_claim", 0 if mode == "none" else 1)):
                bad = copy.deepcopy(good); bad[key] = value
                with self.assertRaises(ValueError): preparation.requested_traffic(bad, 64, 64, 64)
            for key in [key for key in good['operand_reuse'] if key.endswith(('bytes', 'byte_base'))]:
                original = good['operand_reuse'][key]
                for value in ([0.0, False] if original == 0 else [float(original), True, 0] if original is not None else [0, False]):
                    bad = copy.deepcopy(good); bad['operand_reuse'][key] = value
                    with self.assertRaises(ValueError): preparation.requested_traffic(bad, 64, 64, 64)
            bad = copy.deepcopy(good); bad['operand_reuse']['unknown'] = 1
            with self.assertRaises(ValueError): preparation.requested_traffic(bad, 64, 64, 64)

    def test_unknown_epoch_or_shape_refuses_before_artifact_copy(self):
        good = self.receipt(preparation.RELEASE_EPOCH)
        bad = copy.deepcopy(good); bad['code_before']['run_model_tile.py'] = '0' * 64
        bad['code_after'] = bad['code_before'].copy()
        for receipt in (bad, dict(good, code_after={})):
            with self.assertRaises(ValueError): preparation.requested_traffic(receipt, 64, 64, 64)
        for shape in ((True, 64, 64), (64.0, 64, 64), (0, 64, 64), (64, 33, 64)):
            with self.assertRaises(ValueError): preparation.requested_traffic(good, *shape)
        bad.update(status="passed_model_only", recipe="atlas_mxu0_fp8_tensor_pow2_bf16acc_v1", shape=[64, 64, 64])
        support = SimpleNamespace(safe_path=Path, identity=mock.Mock(return_value={}), copy_source=mock.Mock())
        with mock.patch.object(Path, "read_text", return_value=json.dumps(bad)), mock.patch.object(preparation.np, "load") as arrays:
            with self.assertRaises(ValueError): preparation.prepare(support, Path('/unread'), 'receipt', Path('/oracle'), Path('/out'), False)
            support.copy_source.assert_not_called(); arrays.assert_not_called()

    def test_current_release_tuple_and_cold_replay_pin_match_sources(self):
        root = Path(__file__).absolute().parents[1]
        actual = tuple(hashlib.sha256((root / name).read_bytes()).hexdigest() for name in preparation.PRODUCER_NAMES)
        self.assertEqual(actual, preparation.RELEASE_EPOCH)
        tree = ast.parse(Path(__file__).with_name('prepare_cold_tile.py').read_text())
        replay = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'REPLAY' for target in node.targets))
        self.assertEqual(replay, hashlib.sha256(Path(preparation.__file__).read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
