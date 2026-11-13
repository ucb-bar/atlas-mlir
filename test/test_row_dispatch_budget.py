"""Row batching uses the selected budget while preserving checked replay."""
import json
import os

import pytest

from atlas_mlir_native.row_loop import prepare_plan
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.target_binding import NativeCompilationError
from test_tiled_softmax import source_request


def test_default_envelope_and_larger_budget_keep_complete_rows():
    source = source_request((12288, 1024))
    with codegen_options(CodegenOptions(io_memory_bytes=67108864)):
        old = prepare_plan(source)
        assert (old.groups, old.groups_per_launch, old.launches) == (384, 16, 24)
        assert old.record()['schema'] == 'atlas.row_loop_plan.v1'
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=128000000)):
        new = prepare_plan(source)
        assert (new.groups, new.groups_per_launch, new.launches) == (384, 384, 1)
        assert new.record()['dispatch_instruction_budget'] == 128000000
        assert (new.readonly_bytes, new.writable_bytes, new.end) == (
            old.readonly_bytes, old.writable_bytes, old.end)
        # Saved legacy artifacts reconstruct their old loop even in a larger
        # invocation; their independent execution bound is checked separately.
        assert prepare_plan(source, instruction_budget=1000000) == old


@pytest.mark.parametrize('budget', [False, 0, -1, 1.5, '128000000'])
def test_invalid_budget_is_refused(budget):
    with pytest.raises(ValueError, match='budget must be a positive integer'):
        prepare_plan(source_request((1, 32)), instruction_budget=budget)


def test_prime_group_count_and_tail_keep_the_complete_source_frame():
    source = source_request((8193, 64), masked=True)
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=2000000)):
        plan = prepare_plan(source)
        assert (plan.groups, plan.groups_per_launch, plan.launches) == (257, 257, 1)
        assert plan.tiling.regions[-1].valid_rows == 1
        assert plan.end == prepare_plan(source, instruction_budget=1000000).end
    with codegen_options(CodegenOptions(io_memory_bytes=65536,
                                       execution_instruction_limit=128000000)):
        with pytest.raises(NativeCompilationError, match='I/O window'):
            prepare_plan(source)


@pytest.mark.skipif(not all(os.environ.get(key) for key in
    ('ATLAS_RUST_NPU_MODEL', 'ATLAS_OOT_BIN_DIR')), reason='select Rust and Atlas tools')
@pytest.mark.parametrize('shape,kind,dtype', [
    ((65, 64), 'softmax', 'bf16'), ((97, 65), 'softmax', 'f32'),
    ((65, 64), 'masked_softmax', 'bf16'), ((65, 64), 'guarded_softmax', 'bf16')])
def test_larger_batch_generated_c_rust_outputs_and_late_refusal(tmp_path, monkeypatch, shape, kind, dtype):
    from atlas_mlir_native import row_loop
    from test_softmax_input_widening import test_generated_c_and_rust_preserve_score_codes_results_and_rejections

    # Exercise the real batching boundary without needing a multi-million-word
    # unit fixture. Full-layer validation separately uses the production 512.
    monkeypatch.setattr(row_loop, 'ROW_LOOP_COLUMN_TILES_PER_LAUNCH', 4)
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=4000000)):
        test_generated_c_and_rust_preserve_score_codes_results_and_rejections(
            tmp_path, shape, kind, dtype)
    execution = json.loads((tmp_path/'compiled/execution_plan.json').read_text())
    assert execution['schema'] == 'atlas.row_loop_execution_plan.v2'
    assert execution['launches_per_invocation'] == 1
    assert execution['dispatch_instruction_budget'] == 4000000


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_native_instruction_proof_still_rejects_an_insufficient_budget(tmp_path):
    from atlas_mlir_native.row_loop import compile_row_loop
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import build_snapshot, ROOT

    compiled = tmp_path/'compiled'
    with codegen_options(CodegenOptions(io_memory_bytes=67108864, execution_instruction_limit=1)):
        snapshot = build_snapshot(tmp_path/'snapshot')
        with pytest.raises(NativeCompilationError) as refused:
            compile_row_loop(snapshot, source_request((33, 65)), target_source=ROOT,
                             destination=compiled, limits=SearchLimits())
    assert refused.value.status == 'resource_limit', refused.value.reason
    assert 'instruction' in refused.value.reason
    assert not (compiled/'manifest.json').exists()


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
@pytest.mark.parametrize('budget', [1000000, 4000000])
def test_rebinding_preserves_saved_batching_under_a_larger_invocation(tmp_path, monkeypatch, budget):
    from atlas_mlir_native import row_loop
    from atlas_mlir_native.selection_materialization import SelectionCapture, bind_row_loop
    from merlin.semantic_compiler.materialization import canonical_request
    from merlin.semantic_compiler.search import SearchLimits
    from test_looped_mnk import build_snapshot, ROOT

    original = source_request((65, 64))
    canonical = canonical_request(original, source_identity_policy='provenance')
    monkeypatch.setattr(row_loop, 'ROW_LOOP_COLUMN_TILES_PER_LAUNCH', 4)
    compiled, rebound = tmp_path/'compiled', tmp_path/'rebound'
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=budget)):
        capture = SelectionCapture(build_snapshot(tmp_path/'snapshot'))
        row_loop.compile_row_loop(capture, canonical, target_source=ROOT,
                                  destination=compiled, limits=SearchLimits(candidate_nodes=256))
    with codegen_options(CodegenOptions(io_memory_bytes=67108864,
                                       execution_instruction_limit=128000000)):
        bind_row_loop(original, canonical, compiled, rebound, capture, target_source=ROOT)
        plan, _ = row_loop.verify_row_loop(original, rebound, target_source=ROOT)
        assert plan.dispatch_instruction_budget == budget
        assert plan.launches == (3 if budget == 1000000 else 1)
        assert (rebound/'program.bin').read_bytes() == (compiled/'program.bin').read_bytes()
        assert (rebound/'atlas-virtual.mlir').read_bytes() == (compiled/'atlas-virtual.mlir').read_bytes()
        # An otherwise valid retained image cannot stand in for changed source
        # shape, or for an unauthenticated edit to its batching publication.
        changed = source_request((97, 64))
        with pytest.raises(ValueError, match='identical lowered semantics'):
            bind_row_loop(changed, canonical, compiled, tmp_path/'wrong', capture, target_source=ROOT)
        record = json.loads((rebound/'execution_plan.json').read_text())
        record['dispatch_instruction_budget'] = budget + 1
        (rebound/'execution_plan.json').write_text(json.dumps(record))
        with pytest.raises(ValueError):
            row_loop.verify_row_loop(original, rebound, target_source=ROOT)
