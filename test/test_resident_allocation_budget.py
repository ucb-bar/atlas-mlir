"""Resident staging allocation uses fixed work and preserves independent checks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

from atlas_mlir_native.binding import AtlasNativeBinding, BF16_RELU_ATTRS, TARGET_IDENTITY
from atlas_mlir_native.resident_graph import (
    is_staged_bf16, mixed_memory_plan, staged_memory_plan,
)
from merlin.semantic_compiler import buffer_allocation as allocation
from merlin.semantic_compiler.allocate import StorageBank
from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType, pointwise_identity_maps
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_mixed_tile_staging import request as mixed_request
from atlas_mlir_native import resident_graph
from atlas_mlir_native.work_budget import BUFFER_SOLVER_RLIMIT

ROOT = Path(__file__).resolve().parents[1]


def staged_request():
    type_ = TensorType((32, 32), "bf16", "raw-bit-pattern")
    nodes = [SemanticNode("x", "input", (), type_, (BF16_RELU_ATTRS[0],), "input")]
    # Exceed native candidate_nodes with complete, individually admitted tiles.
    for index in range(25):
        nodes.append(SemanticNode(f"relu{index}", "relu", (nodes[-1].id,), type_,
            BF16_RELU_ATTRS, index_maps=pointwise_identity_maps(2, 1)))
    return KernelRequest(tuple(nodes), (nodes[-1].id,), ("external",),
        (("x", "external"),), TARGET_IDENTITY, "diagnostic")


class ResidentAllocationBudgetTest(unittest.TestCase):
    def cases(self):
        return (("mixed", mixed_memory_plan, mixed_request(expose=False, transpose=True)),
                ("staged", staged_memory_plan, staged_request()))

    def test_both_routes_check_proposals_without_consulting_wall_budget(self):
        for name, plan, source in self.cases():
            with self.subTest(route=name):
                before = source.record()
                with allocation.allocation_wall_budget(1), \
                     patch.object(allocation, "current_allocation_wall_budget_ms", side_effect=AssertionError("wall budget")), \
                     patch.object(allocation, "allocate_buffers", side_effect=AssertionError("redundant solver")) as allocate, \
                     patch.object(allocation, "check_buffer_allocation", wraps=allocation.check_buffer_allocation) as check:
                    cuts, slots, sizes, buffers = plan(source)
                allocate.assert_not_called()
                self.assertEqual(check.call_count, 3)
                self.assertEqual(check.call_args_list[-2].args[0][0].name, "$readonly")
                self.assertEqual(set(check.call_args_list[-1].args[2]), set(slots))
                self.assertGreater(len(cuts), 1)
                self.assertEqual(set(slots), set(sizes))
                self.assertTrue(next(b for b in buffers if b["name"] == "$readonly")["persistent"])
                self.assertEqual(source.record(), before)
        self.assertTrue(is_staged_bf16(staged_request()))

    def test_both_routes_use_fixed_work_and_unknown_aborts_without_retry(self):
        for name, plan, source in self.cases():
            with self.subTest(route=name):
                with allocation.allocation_wall_budget(23), \
                     patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
                     patch.object(allocation, "allocate_buffers", return_value=allocation.BufferAllocation(
                         "unknown", reason="test deterministic work exhausted")) as allocate, \
                     self.assertRaises(NativeCompilationError) as caught:
                    plan(source)
                allocate.assert_called_once()
                self.assertEqual(allocate.call_args.kwargs, dict(timeout_ms=None, rlimit=BUFFER_SOLVER_RLIMIT))
                self.assertEqual(caught.exception.status, "resource_limit")
                self.assertIn("buffer_solver_rlimit_per_query", caught.exception.reason)
                self.assertIn("test deterministic work exhausted", caught.exception.reason)

    def test_actual_window512_uses_identical_checked_detailed_proposal(self):
        from atlas_mlir_native.row_softmax import prepare_row_softmax
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from test_tiled_softmax import source_request
        proposals = []
        first_fit = allocation.first_fit_buffer_assignment
        def proposal(buffers, banks):
            result = first_fit(buffers, banks); proposals.append((buffers, dict(result))); return result
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            source = source_request((16384, 512), masked=True)
            body = prepare_row_softmax(source).regions[0].request
            before = body.record()
            with allocation.allocation_wall_budget(1), \
                 patch.object(allocation, "first_fit_buffer_assignment", side_effect=proposal), \
                 patch.object(allocation, "allocate_buffers", side_effect=AssertionError("redundant solver")) as allocate, \
                 patch.object(allocation, "check_buffer_allocation", wraps=allocation.check_buffer_allocation) as check:
                cuts, slots, _, _ = staged_memory_plan(body)
            self.assertEqual(len(cuts), 17)
            self.assertEqual(len(proposals[0][0]), 97)
            self.assertEqual({key:value for key,value in proposals[0][1].items() if key != "$readonly"}, slots)
            allocate.assert_not_called()
            self.assertEqual(check.call_count, 3)
            self.assertEqual(check.call_args_list[-2].args[0], proposals[0][0])
            self.assertEqual(check.call_args_list[-2].args[2], slots)
            self.assertEqual(check.call_args_list[-1].args[2], slots)
            self.assertEqual(body.record(), before)


class ResidentFrameFailureTest(unittest.TestCase):
    def setUp(self):
        B = allocation.Buffer
        self.bank = StorageBank("external", "dram", 8, "tiles")
        self.original = (B("source", "external", 1, 0, 3, fixed_start=0, persistent=True),
                         B("result", "external", 1, 1, 3, fixed_start=1),
                         B("early", "external", 1, 1, 1), B("late", "external", 1, 2, 2))
        self.buffers = (B("$readonly", "external", 1, 0, 3, fixed_start=0, persistent=True),
                        replace(self.original[0], view_of="$readonly", view_offset=0), *self.original[1:])
        self.proposal = allocation.first_fit_buffer_assignment(self.buffers, (self.bank,))

    def allocate(self, bank=None):
        return resident_graph._allocate_staging_frame(self.buffers, self.original, bank or self.bank, "test staging")

    def test_frame_mapping_matches_original_pinned_solver_and_reuses_dead_storage(self):
        pinned = tuple(replace(buffer, fixed_start=self.proposal[buffer.name]) for buffer in self.buffers)
        old = allocation.allocate_buffers(pinned, (self.bank,))
        self.assertEqual(old.status, "sat")
        slots = self.allocate()
        self.assertEqual(slots, dict(old.addresses))
        self.assertEqual(slots["early"], slots["late"])
        self.assertGreater(slots["early"], slots["result"])

    def test_missing_proposal_retains_generic_solver_and_overfull_refuses(self):
        with patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
             patch.object(allocation, "allocate_buffers", wraps=allocation.allocate_buffers) as allocate:
            self.assertEqual(self.allocate(), self.proposal)
        self.assertEqual(allocate.call_args.args[0], self.buffers)
        allocate.assert_called_once()
        with self.assertRaises(NativeCompilationError) as caught:
            self.allocate(replace(self.bank, capacity=1))
        self.assertEqual(caught.exception.status, "resource_limit")

    def test_damaged_proposal_and_bank_escape_refuse(self):
        for key in ("source", "early"):
            with self.subTest(buffer=key):
                bad = {**self.proposal, key:self.proposal["result"]}
                with patch.object(allocation, "first_fit_buffer_assignment", return_value=bad), \
                     patch.object(allocation, "allocate_buffers") as allocate, \
                     self.assertRaises(NativeCompilationError):
                    self.allocate()
                allocate.assert_not_called()
        bad = {**self.proposal, "early": self.bank.capacity}
        with patch.object(allocation, "first_fit_buffer_assignment", return_value=bad), \
             self.assertRaisesRegex(NativeCompilationError, "invalid detailed frame"):
            self.allocate()

    def test_changed_fixed_address_and_empty_frame_refuse(self):
        bad = {**self.proposal, "result": self.proposal["result"] + 1}
        with patch.object(allocation, "first_fit_buffer_assignment", return_value=bad), \
             self.assertRaisesRegex(NativeCompilationError, "invalid detailed frame"):
            self.allocate()
        with self.assertRaisesRegex(NativeCompilationError, "nonempty staging buffers"):
            resident_graph._allocate_staging_frame((), (), self.bank, "test staging")

    def test_proposal_and_both_detailed_checks_ignore_clock_progress(self):
        original_check = allocation.check_buffer_allocation
        checked = []
        def check(buffers, banks, slots, **kwargs):
            original_check(buffers, banks, slots, **kwargs)
            checked.append(buffers)
        with allocation.allocation_wall_budget(1), \
             patch.object(allocation, "monotonic", side_effect=AssertionError("wall clock")), \
             patch.object(allocation, "first_fit_buffer_assignment", return_value=dict(self.proposal)), \
             patch.object(allocation, "allocate_buffers") as allocate, \
             patch.object(allocation, "check_buffer_allocation", side_effect=check):
            self.assertEqual(self.allocate(), self.proposal)
        self.assertEqual(checked, [self.buffers, self.original])
        allocate.assert_not_called()

    def test_smaller_context_does_not_change_checked_proposal(self):
        with allocation.allocation_wall_budget(1), \
             patch.object(allocation, "allocate_buffers") as allocate:
            self.assertEqual(self.allocate(), self.proposal)
        allocate.assert_not_called()

    def test_solver_work_exhaustion_is_named_and_never_retried(self):
        with allocation.allocation_wall_budget(1), \
             patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
             patch.object(allocation, "allocate_buffers", return_value=allocation.BufferAllocation(
                 "unknown", reason="bounded allocator refused")) as allocate, \
             self.assertRaises(NativeCompilationError) as caught:
            self.allocate()
        allocate.assert_called_once_with(self.buffers, (self.bank,), timeout_ms=None, rlimit=BUFFER_SOLVER_RLIMIT)
        self.assertEqual(caught.exception.status, "resource_limit")
        self.assertIn("buffer_solver_rlimit_per_query", caught.exception.reason)

    def test_generic_sat_keeps_checked_placement_independent_of_clock(self):
        with allocation.allocation_wall_budget(1), \
             patch.object(allocation, "monotonic", side_effect=AssertionError("wall clock")), \
             patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
             patch.object(allocation, "allocate_buffers", return_value=allocation.BufferAllocation(
                 "sat", tuple(self.proposal.items()))) as allocate:
            self.assertEqual(self.allocate(), self.proposal)
        allocate.assert_called_once_with(self.buffers, (self.bank,), timeout_ms=None, rlimit=BUFFER_SOLVER_RLIMIT)


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                     "select native tools")
class ResidentAllocationNativeTest(unittest.TestCase):
    def test_mixed_and_staged_compile_with_default_search_and_fixed_allocation_work(self):
        from merlin.semantic_compiler.search import SearchLimits
        from merlin.semantic_compiler.snapshot import build_native_snapshot, packaged_egg_bridge_path
        from merlin.semantic_compiler.target_binding import verify_native_publication

        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as temporary:
            work = Path(temporary)
            snapshot = build_native_snapshot(AtlasNativeBinding.profile(), destination=work/"snapshot",
                crate=packaged_egg_bridge_path(),
                cargo_target_dir=Path(os.environ.get("ATLAS_MXU1_CARGO_TARGET", work/"cargo")),
                source_revision="resident-default-allocation-regression")
            for name, source in (("mixed", mixed_request(expose=False)), ("staged", staged_request())):
                with self.subTest(route=name), allocation.allocation_wall_budget(1000), \
                     patch.object(allocation, "allocate_buffers", wraps=allocation.allocate_buffers) as allocate:
                    abi = AtlasNativeBinding.plan_boundary_abi(source)
                    destination = work/name
                    manifest = AtlasNativeBinding.compile(snapshot, source, fixed_inputs=abi["fixed_inputs"],
                        fixed_outputs=tuple(abi["fixed_outputs"]), target_source=ROOT,
                        destination=destination, limits=SearchLimits())
                    verify_native_publication(destination, manifest, engine="merlin_native",
                        request_digest=source.digest(), target_identity=source.target_identity)
                    execution = json.loads((destination/"execution_plan.json").read_text())
                    self.assertGreater(len(execution["regions"]), 1)
                    self.assertTrue(all(call.kwargs == dict(timeout_ms=None, rlimit=BUFFER_SOLVER_RLIMIT)
                                        for call in allocate.call_args_list))


if __name__ == "__main__":
    unittest.main()
