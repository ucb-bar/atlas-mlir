"""Both resident staging routes honor the caller's allocation wall budget."""
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

    def test_both_real_allocations_use_default_1000_without_override(self):
        self.assertEqual(allocation.current_allocation_wall_budget_ms(), 1000)
        for name, plan, source in self.cases():
            with self.subTest(route=name):
                before = source.record()
                selected = []
                current_budget = allocation.current_allocation_wall_budget_ms
                def read_budget():
                    value = current_budget(); selected.append(value); return value
                with allocation.allocation_wall_budget(1000), \
                     patch.object(allocation, "current_allocation_wall_budget_ms",
                                  side_effect=read_budget), \
                     patch.object(allocation, "allocate_buffers", side_effect=AssertionError("redundant solver")) as allocate, \
                     patch.object(allocation, "check_buffer_allocation", wraps=allocation.check_buffer_allocation) as check:
                    cuts, slots, sizes, buffers = plan(source)
                # A 5000ms explicit override bypasses the context lookup. Check
                # the lookup and checked allocation, rather than receipt metadata.
                self.assertEqual(selected, [1000])
                allocate.assert_not_called()
                # First fit checks its proposal; staging then checks both
                # the transformed views and the original logical buffers.
                self.assertEqual(check.call_count, 3)
                self.assertEqual(check.call_args_list[-2].args[0][0].name, "$readonly")
                self.assertEqual(set(check.call_args_list[-1].args[2]), set(slots))
                self.assertGreater(len(cuts), 1)
                self.assertEqual(set(slots), set(sizes))
                self.assertTrue(next(b for b in buffers if b["name"] == "$readonly")["persistent"])
                self.assertEqual(source.record(), before)
        self.assertTrue(is_staged_bf16(staged_request()))

    def test_both_routes_inherit_a_smaller_context_and_keep_timeout_fail_closed(self):
        for name, plan, source in self.cases():
            with self.subTest(route=name):
                selected = []
                def timeout(*args, **kwargs):
                    self.assertNotIn("timeout_ms", kwargs)
                    selected.append(allocation.current_allocation_wall_budget_ms())
                    return allocation.BufferAllocation("unknown", reason="test allocation wall exhausted")
                with allocation.allocation_wall_budget(23), \
                     patch.object(resident_graph, "monotonic", return_value=0), \
                     patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
                     patch.object(allocation, "allocate_buffers", side_effect=timeout) as allocate, \
                     self.assertRaises(NativeCompilationError) as caught:
                    plan(source)
                self.assertEqual(selected, [23])
                allocate.assert_called_once()
                self.assertEqual(caught.exception.status, "search_timeout")
                self.assertIn("allocation wall exhausted", str(caught.exception))

    def test_actual_window512_uses_identical_checked_detailed_proposal(self):
        from atlas_mlir_native.row_softmax import prepare_row_softmax
        from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
        from test_tiled_softmax import source_request
        proposals = []; times = []
        first_fit = allocation.first_fit_buffer_assignment
        clock = resident_graph.monotonic
        def proposal(buffers, banks):
            result = first_fit(buffers, banks); proposals.append((buffers, dict(result))); return result
        def timestamp():
            value = clock(); times.append(value); return value
        with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
            source = source_request((16384, 512), masked=True)
            body = prepare_row_softmax(source).regions[0].request
            before = body.record()
            with allocation.allocation_wall_budget(1000), \
                 patch.object(allocation, "first_fit_buffer_assignment", side_effect=proposal), \
                 patch.object(allocation, "allocate_buffers", side_effect=AssertionError("redundant solver")) as allocate, \
                 patch.object(allocation, "check_buffer_allocation", wraps=allocation.check_buffer_allocation) as check, \
                 patch.object(resident_graph, "monotonic", side_effect=timestamp):
                cuts, slots, _, _ = staged_memory_plan(body)
            self.assertEqual(len(cuts), 17)
            self.assertEqual(len(proposals[0][0]), 97)
            self.assertEqual({key:value for key,value in proposals[0][1].items() if key != "$readonly"}, slots)
            allocate.assert_not_called()
            self.assertEqual(check.call_count, 3)
            self.assertEqual(check.call_args_list[-2].args[0], proposals[0][0])
            self.assertEqual(check.call_args_list[-2].args[2], slots)
            self.assertEqual(check.call_args_list[-1].args[2], slots)
            self.assertLessEqual((times[-1] - times[0]) * 1000, 1000)
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

    def test_one_outer_deadline_covers_proposal_and_both_detailed_checks(self):
        for phase in ("proposal", "transformed", "original"):
            with self.subTest(phase=phase):
                now = [0.0]
                original_check = allocation.check_buffer_allocation
                def propose(buffers, banks):
                    if phase == "proposal": now[0] = 1.001
                    return dict(self.proposal)
                def check(buffers, banks, slots, **kwargs):
                    original_check(buffers, banks, slots, **kwargs)
                    selected = "transformed" if buffers[0].name == "$readonly" else "original"
                    if phase == selected: now[0] = 1.001
                with patch.object(resident_graph, "monotonic", side_effect=lambda:now[0]), \
                     patch.object(allocation, "first_fit_buffer_assignment", side_effect=propose), \
                     patch.object(allocation, "allocate_buffers") as allocate, \
                     patch.object(allocation, "check_buffer_allocation", side_effect=check), \
                     self.assertRaises(NativeCompilationError) as caught:
                    self.allocate()
                self.assertEqual(caught.exception.status, "search_timeout")
                allocate.assert_not_called()

    def test_smaller_context_bounds_checked_proposal_without_backend_import(self):
        now = [0.0]
        def propose(buffers, banks):
            now[0] = .024
            return dict(self.proposal)
        with allocation.allocation_wall_budget(23), \
             patch.object(resident_graph, "monotonic", side_effect=lambda:now[0]), \
             patch.object(allocation, "first_fit_buffer_assignment", side_effect=propose), \
             patch.object(allocation, "allocate_buffers") as allocate, \
             self.assertRaises(NativeCompilationError) as caught:
            self.allocate()
        self.assertEqual(caught.exception.status, "search_timeout")
        allocate.assert_not_called()

    def test_solver_only_gets_remaining_smaller_context_without_retry(self):
        now = [0.0]; observed = []
        def propose(buffers, banks):now[0] = .005; return None
        def solve(buffers, banks):
            observed.append(allocation.current_allocation_wall_budget_ms())
            return allocation.BufferAllocation("unknown", reason="bounded allocator refused")
        with allocation.allocation_wall_budget(23), \
             patch.object(resident_graph, "monotonic", side_effect=lambda:now[0]), \
             patch.object(allocation, "first_fit_buffer_assignment", side_effect=propose), \
             patch.object(allocation, "allocate_buffers", side_effect=solve) as allocate, \
             self.assertRaises(NativeCompilationError):
            self.allocate()
        self.assertEqual(len(observed), 1)
        self.assertTrue(0 < observed[0] <= 18)
        allocate.assert_called_once()

    def test_generic_late_sat_refuses_without_retry(self):
        now = [0.0]
        def solve(buffers, banks):
            now[0] = 1.001
            return allocation.BufferAllocation("sat", tuple(self.proposal.items()))
        with patch.object(resident_graph, "monotonic", side_effect=lambda:now[0]), \
             patch.object(allocation, "first_fit_buffer_assignment", return_value=None), \
             patch.object(allocation, "allocate_buffers", side_effect=solve) as allocate, \
             self.assertRaises(NativeCompilationError) as caught:
            self.allocate()
        self.assertEqual(caught.exception.status, "search_timeout")
        allocate.assert_called_once()


@unittest.skipUnless(os.environ.get("MERLIN_NATIVE_ROOT") and os.environ.get("ATLAS_OOT_BIN_DIR"),
                     "select native tools")
class ResidentAllocationNativeTest(unittest.TestCase):
    def test_mixed_and_staged_compile_at_default_search_and_1000ms_allocation(self):
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
                    self.assertTrue(all("timeout_ms" not in call.kwargs for call in allocate.call_args_list))


if __name__ == "__main__":
    unittest.main()
