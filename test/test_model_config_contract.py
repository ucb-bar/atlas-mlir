"""Configuration capabilities using private mutations of opted-in derived facts.

Resealed configuration variants are synthetic test data, written only to each
test's temporary directory. They are not model derivation or production exports.
"""
from copy import deepcopy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from model_evidence import requires_model_evidence
from test_delay_insertion import OPT, ROOT, run, program

sys.path.insert(0, str(ROOT / "tools"))
from atlas_derived_contract import seal_derived, verify_derived


@requires_model_evidence
class ModelConfigContractTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.facts = json.loads((Path(os.environ["ATLAS_TEST_MODEL_FACTS"]) / "model-timing.json").read_text())
        # Check the genuine input before creating any private synthetic variants.
        # Raw observer fixtures must never be promoted to derived facts here.
        verify_derived(self.facts)

    def select(self):
        model = self.facts["model"]
        model["configuration_json"] = json.dumps(model["configuration"], sort_keys=True, separators=(",", ":"))
        model["configuration_sha256"] = hashlib.sha256(model["configuration_json"].encode()).hexdigest()
        # These mutations isolate compiler capability checks from derivation.
        # Resealing is a private test operation, never a production bypass.
        seal_derived(self.facts)
        path = self.directory / "synthetic-config-test-only.json"
        path.write_text(json.dumps(self.facts))
        option = (f"--select-atlas-model-evidence=test-op-timing={path} "
                  f"test-op-timing-sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
        return run(OPT, program([("trap", 'kind = "ecall"')]), option,
                   "--schedule-atlas-stream=provider=model", "--verify-atlas-timing")

    def change_timing(self):
        params = self.facts["model"]["configuration"]["dma_memory_params"]
        params.update(frequency_ghz=.75, queue_depth=16, window_beats=8,
                      curves="selected-curve.json")
        self.facts["dma_timing"].update(frequency_ghz=.75, queue_depth=16, window_beats=8,
                                        curve_sha256="a" * 64)

    def test_accepts_timing_configuration_and_retains_curve_hash(self):
        self.change_timing()
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('dma_curve_sha256 = "' + "a" * 64, result.stdout)
        self.assertNotIn("falling back to isolated lifetimes", result.stderr)

    def test_architecture_change_names_unsupported_capability(self):
        self.facts["model"]["configuration"]["arch_state_config"]["mrf_depth"] = 16
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported capability: arch_state_config.mrf_depth (requires 32)", result.stderr)

    def test_configuration_and_facts_must_agree(self):
        self.change_timing()
        self.facts["dma_timing"]["frequency_ghz"] = .5
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frequency_ghz mismatch with selected configuration", result.stderr)

    def test_null_and_unknown_backend_settings_refuse(self):
        baseline = deepcopy(self.facts)
        for key in ("window_beats", "queue_depth", "min_latency", "converge", "knee", "enforce_peak", "unknown"):
            with self.subTest(key=key):
                self.facts = deepcopy(baseline)
                params = self.facts["model"]["configuration"]["dma_memory_params"]
                params[key] = None
                result = self.select()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unsupported capability: dma_memory_params." + key, result.stderr)

    def test_nullable_backend_defaults_are_supported(self):
        self.facts["model"]["configuration"]["dma_memory_params"].update(
            frequency_ghz=None, channels=None, latency_mode=None)
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_python_float_serialization_matches_configuration_values(self):
        for frequency in (1.0, .000001, .75):
            with self.subTest(frequency=frequency):
                self.facts["model"]["configuration"]["dma_memory_params"]["frequency_ghz"] = frequency
                self.facts["dma_timing"]["frequency_ghz"] = frequency
                result = self.select()
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_legacy_selected_default_curve_metadata_remains_usable(self):
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        import re
        legacy = re.sub(r'dma_curve_sha256 = "[0-9a-f]{64}", ', '', result.stdout)
        self.assertNotEqual(legacy, result.stdout)
        result = run(OPT, legacy, "--verify-atlas-timing")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_selected_engine_capacity_is_bound_not_fixed(self):
        self.facts["model"]["configuration"]["dma_max_in_flight"] = 16
        self.facts["architecture"]["dma_max_in_flight"] = 16
        self.facts["dma_timing"]["max_in_flight"] = 16
        self.facts["model"]["configuration"]["dma_memory_params"]["queue_depth"] = 64
        self.facts["dma_timing"]["queue_depth"] = 64
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.facts["architecture"]["dma_max_in_flight"] = 64
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("derived architecture differs from configuration: dma_max_in_flight", result.stderr)

    def test_dormant_geometry_is_bound_without_default_value_gate(self):
        for key in ("offchip_link_width_bits", "offchip_link_core_cycles_per_beat",
                    "dma_offchip_command_words", "vmem_bus_width_bits",
                    "vmem_bus_core_cycles_per_beat", "vmem_bytes_per_cycle"):
            self.facts["model"]["configuration"][key] = 0
            self.facts["architecture"][key] = 0
        result = self.select()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.facts["architecture"]["vmem_bytes_per_cycle"] = 64
        result = self.select()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("derived architecture differs from configuration: vmem_bytes_per_cycle", result.stderr)
