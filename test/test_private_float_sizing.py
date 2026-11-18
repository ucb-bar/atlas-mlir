"""Size private binary slices from resources, then prove each native body."""
from dataclasses import replace
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from atlas_mlir_native.application_materialization import bind_exact_float
from merlin.semantic_compiler.codegen import codegen_options, current_codegen_options
from test_exact_float_device import request

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('budget,capacity', [(1000000,1024), (1500000,2048),
                                            (2000000,3072), (16000000,24576),
                                            (128000000,199680)])
def test_instruction_envelope_preserves_raw_policy_and_full_tail_slices(budget, capacity):
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=budget,
                                io_memory_bytes=67108864)):
        source = request('bf16_add', 2*capacity+35)
        schedule = application.prepare_plan(source, private_output=True)
        assert schedule.chunk_elements == capacity
        assert [row['output_bytes']//2 for row in schedule.launches] == [capacity,capacity,35]
        assert [row['output_offset_bytes']//2 for row in schedule.launches] == [0,capacity,2*capacity]
        raw = device.prepare_plan(source)
        assert device.max_leaf_elements(raw) == max(1,budget//1000000)*1024
        assert all(device.prepare_plan(body, private_output=True).workspace_bytes <= 67108864
                   for body in schedule.bodies)


@pytest.mark.parametrize('window,capacity', [(8192,1024),(32768,5120)])
def test_device_memory_still_bounds_every_slice(window, capacity):
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=128000000,
                                io_memory_bytes=window)):
        schedule = application.prepare_plan(request('bf16_sub', 100000), private_output=True)
        assert schedule.chunk_elements == capacity
        for body in schedule.bodies:
            assert device.prepare_plan(body, private_output=True).workspace_bytes <= window
        for value in (True, 0, -1024, 33, capacity+1024):
            with pytest.raises(ValueError, match='resource envelope'):
                application.prepare_plan(schedule.source, private_output=True, _chunk_elements=value)


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
@pytest.mark.parametrize('pipeline,groups,vector,operation,shared,budget', [
    ('mandatory',False,False,'bf16_add',False,128000000),
    ('mandatory',True,True,'bf16_sub',False,128000000),
    ('optimized',True,True,'bf16_add',False,128000000),
    ('optimized',True,True,'bf16_sub',True,128000000),
    ('mandatory',False,True,'bf16_sub',False,128000000),
    ('optimized',True,False,'bf16_sub',False,128000000),
    ('optimized',False,False,'bf16_add',True,128000000),
    ('optimized',True,True,'bf16_add',False,1000000000),
])
def test_largest_slice_has_independent_native_execution_and_image_proofs(
        tmp_path, pipeline, groups, vector, operation, shared, budget):
    options = replace(current_codegen_options(), execution_instruction_limit=budget,
        io_memory_bytes=67108864, native_pipeline=pipeline,
        exact_binary_lane_groups=groups, exact_binary_vector_preflight=vector)
    with codegen_options(options):
        capacity = application.prepare_plan(request(operation,1024,shared=shared),
                                            private_output=True).chunk_elements
        source = request(operation,capacity,shared=shared)
        compiled = tmp_path/'compiled'
        device.compile_source(None,source,target_source=ROOT,destination=compiled,
                              limits=None,private_output=True)
        bound = json.loads((compiled/'float-instruction-bound.json').read_text())
        assert bound['admitted'] and bound['instruction_bound'] <= budget
        assert (compiled/'program.bin').stat().st_size <= 131072
        device.prepare_c_package(source,compiled,'checked',target_source=ROOT,private_output=True)
        with pytest.raises(ValueError, match='at most'):
            device.compile_source(None,request(operation,capacity+1024,shared=shared),
                target_source=ROOT,destination=tmp_path/'over',limits=None,private_output=True)
        if budget == 1000000000:
            assert capacity > 512*1024  # Private kernels have no per-tile route table.


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_replay_and_rebinding_preserve_an_older_smaller_slice_schedule(tmp_path):
    options = replace(current_codegen_options(), execution_instruction_limit=4000000,
                      io_memory_bytes=67108864)
    source = request('bf16_add', 2051)
    with codegen_options(options):
        compiled = tmp_path/'old'
        with patch.object(device,'chunk_elements',return_value=1024):
            application.compile_source(None,source,target_source=ROOT,destination=compiled,
                                       limits=None,private_output=True)
            old = application.prepare_c_package(source,compiled,'checked',target_source=ROOT)
        assert application.prepare_plan(source,private_output=True).chunk_elements > 1024
        assert application.prepare_c_package(source,compiled,'checked',target_source=ROOT) == old
        rebound = tmp_path/'rebound'
        bind_exact_float(source,source,compiled,rebound)
        assert application.prepare_c_package(source,rebound,'checked',target_source=ROOT) == old
