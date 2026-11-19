"""Tile-local validation may write only explicitly private device results."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas_mlir_native import exact_float_application as application
from atlas_mlir_native import exact_float_device as device
from merlin.semantic_compiler.codegen import codegen_options, current_codegen_options
from test_exact_float_device import request

ROOT = Path(__file__).resolve().parents[1]


def test_private_ownership_is_explicit_and_limited_to_binary_leaves():
    source = request('bf16_add', 2048)
    raw = device.prepare_plan(source)
    assert not raw.private_output and 'private_output' not in raw.record()
    private = device.prepare_plan(source, private_output=True)
    assert private.record()['private_output'] is True
    assert 'private_complete' in device.virtual_program(private)
    assert 'private_complete' not in device.virtual_program(raw)
    for value in (1, 'true', None):
        with pytest.raises(ValueError):
            device.prepare_plan(source, private_output=value)
    with pytest.raises(ValueError):
        device.prepare_plan(request('f32_to_bf16', 32), private_output=True)


def test_application_replay_preserves_legacy_and_private_contracts(tmp_path):
    if not os.environ.get('ATLAS_OOT_BIN_DIR'):
        pytest.skip('select Atlas tools')
    source = request('bf16_add', 1059)
    from atlas_mlir_native.application_materialization import bind_exact_float
    for private in (False, True):
        compiled=tmp_path/str(private)
        application.compile_source(None,source,target_source=ROOT,destination=compiled,
                                   limits=None,private_output=private)
        package=application.prepare_c_package(source,compiled,'checked',target_source=ROOT)
        assert package.outputs == source.outputs
        rebound=tmp_path/('rebound-'+str(private))
        bind_exact_float(source,source,compiled,rebound)
        assert application.prepare_c_package(source,rebound,'checked',target_source=ROOT) == package
        schedule=json.loads((compiled/'float-schedule.json').read_text())
        assert schedule.get('private_output',False) == private
        # Rehashing the parent flag cannot relabel the native child programs.
        schedule['private_output']=not private
        path=compiled/'float-schedule.json'
        path.write_text(json.dumps(schedule))
        from hashlib import sha256
        manifest_path=compiled/'float-manifest.json'
        manifest=json.loads(manifest_path.read_text())
        manifest['stage_artifacts']['float-schedule.json']=sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError):
            application.prepare_c_package(source,compiled,'wrong',target_source=ROOT)
        with pytest.raises(ValueError):
            bind_exact_float(source,source,compiled,tmp_path/('wrong-'+str(private)))


@pytest.mark.skipif(not os.environ.get('ATLAS_OOT_BIN_DIR'), reason='select Atlas tools')
def test_private_results_do_not_bypass_execution_or_image_limits(tmp_path):
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    source = request('bf16_add', 2048)
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=4000000)):
        with pytest.raises(NativeCompilationError, match='instruction bound exceeds 1:'):
            device.compile_source(None, source, target_source=ROOT, destination=tmp_path/'execution',
                limits=SimpleNamespace(functional_instruction_limit=1), private_output=True)
    with codegen_options(replace(current_codegen_options(), execution_instruction_limit=4000000,
                                program_size_limit_bytes=4)):
        with pytest.raises(NativeCompilationError, match='program.*budget|program.*limit'):
            device.compile_source(None, source, target_source=ROOT, destination=tmp_path/'image',
                                  limits=None, private_output=True)
    assert not (tmp_path/'execution').exists()
    assert not (tmp_path/'image').exists()
