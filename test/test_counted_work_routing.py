"""Counted compiler exhaustion must propagate through SP5 admission."""

import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from atlas_mlir_native.activation_quantize_application import prepare_plan
from atlas_mlir_native.application import AtlasApplicationAdapter
from atlas_mlir_native.binding import RTL_REVISION
from atlas_mlir_native.options import CodegenOptions, codegen_options
from atlas_mlir_native.work_budget import CountedWorkExhausted
from test_activation_quantize_device import dynamic_source
from test_layout_device import cases


class CountedWorkRoutingTest(unittest.TestCase):
    def test_quantizer_exhaustion_propagates_but_bound_refusal_shrinks(self):
        source = dynamic_source(shape=(2, 16, 64), reduce="head")
        options = CodegenOptions(
            device_kv_quantizer=True,
            native_pipeline="mandatory",
            io_memory_bytes=67108864,
        )
        emitted = json.dumps({
            "selected_rtl_revision": RTL_REVISION, "instructions": [],
        })
        exhausted = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
            stderr="Atlas work budget exhausted: x",
        )
        success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=emitted, stderr="",
        )

        with tempfile.TemporaryDirectory() as tools, \
                patch.dict(os.environ, ATLAS_OOT_BIN_DIR=tools), \
                codegen_options(options):
            with patch("atlas_mlir_native.binding.subprocess.run",
                       return_value=exhausted) as run:
                with self.assertRaises(CountedWorkExhausted) as failure:
                    prepare_plan(source)
                self.assertEqual(failure.exception.status, "resource_limit")
                self.assertIn("Atlas work budget exhausted: x",
                              failure.exception.reason)
                # The first two-head pack aborts; no smaller pack is tried.
                self.assertEqual(run.call_count, 1)

            # A completed proof refuses the two-head pack, then admits one
            # head. The second head reuses that admitted body.
            with patch("atlas_mlir_native.binding.subprocess.run",
                       return_value=success) as run, \
                    patch("atlas_mlir_native.restoration_budget.instruction_bound",
                          side_effect=[
                              {"admitted": False, "max_instructions": 1000000,
                               "instruction_bound": 1000001},
                              {"admitted": True, "max_instructions": 1000000,
                               "instruction_bound": 500000},
                          ]) as bound:
                scheduled = prepare_plan(source)
                self.assertEqual(bound.call_count, 2)
                # Each quantizer proof lowers, verifies, and emits JSON.
                self.assertEqual(run.call_count, 6)
                self.assertEqual(len(scheduled.bodies), 1)
                self.assertEqual(scheduled.bodies[0].nodes[0].type.shape,
                                 (1, 16, 64))
                self.assertEqual([row["code_bytes"] for row in scheduled.launches],
                                 [1024, 1024])
                self.assertEqual([row["exponent_bytes"] for row in scheduled.launches],
                                 [4, 4])

    def test_layout_exhaustion_propagates_but_bound_refusal_returns_false(self):
        source = cases()["ragged_xlu"]
        options = CodegenOptions(
            device_kv_quantizer=True,
            native_pipeline="mandatory",
            io_memory_bytes=67108864,
        )
        emitted = json.dumps({
            "selected_rtl_revision": RTL_REVISION, "instructions": [],
        })
        exhausted = subprocess.CompletedProcess(
            args=[], returncode=1, stdout="",
            stderr="Atlas work budget exhausted: x",
        )
        success = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=emitted, stderr="",
        )

        def supports():
            return AtlasApplicationAdapter._supports_device_source(
                source, source.nodes[-1], "layout_device",
            )

        with tempfile.TemporaryDirectory() as tools, \
                patch.dict(os.environ, ATLAS_OOT_BIN_DIR=tools), \
                codegen_options(options):
            with patch("atlas_mlir_native.binding.subprocess.run",
                       return_value=exhausted) as run:
                with self.assertRaises(CountedWorkExhausted) as failure:
                    supports()
                self.assertEqual(failure.exception.status, "resource_limit")
                self.assertIn("Atlas work budget exhausted: x",
                              failure.exception.reason)
                self.assertEqual(run.call_count, 1)

            with patch("atlas_mlir_native.binding.subprocess.run",
                       return_value=success) as run, \
                    patch("atlas_mlir_native.restoration_budget.instruction_bound",
                          return_value={"admitted": False,
                                        "instruction_bound": 1000001}) as bound:
                self.assertIs(supports(), False)
                bound.assert_called_once()
                # Mandatory layout proof lowers and emits JSON.
                self.assertEqual(run.call_count, 2)
