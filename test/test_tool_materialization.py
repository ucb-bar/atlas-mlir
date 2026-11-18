"""Pure tool reuse retains original output and diagnostics."""
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from atlas_mlir_native.binding import _run
from merlin.semantic_compiler.materialization import materialization_session


def test_tool_output_and_diagnostics_are_exact():
    with tempfile.TemporaryDirectory() as directory:
        tool = Path(directory) / 'atlas-opt'
        tool.write_bytes(b'tool identity')
        responses = subprocess.CompletedProcess([], 0, 'machine\n', 'diagnostic\n')
        diagnostics = []
        with patch('subprocess.run', return_value=responses) as run:
            fresh = _run([str(tool), '-'], input_text='module {}', diagnostics=diagnostics)
            with materialization_session(enabled=True) as stats:
                first = _run([str(tool), '-'], input_text='module {}', diagnostics=diagnostics)
                hit = _run([str(tool), '-'], input_text='module {}', diagnostics=diagnostics)
            assert run.call_count == 2
        assert fresh == first == hit == 'machine\n'
        assert diagnostics == ['diagnostic\n'] * 3
        assert stats['hits'] == 1


def test_semantic_inputs_options_and_tool_changes_miss():
    with tempfile.TemporaryDirectory() as directory:
        tool = Path(directory) / 'atlas-opt'
        tool.write_bytes(b'first tool')
        result = subprocess.CompletedProcess([], 0, 'machine', '')
        with patch('subprocess.run', return_value=result) as run, materialization_session(enabled=True):
            _run([str(tool), '-'], input_text='first')
            _run([str(tool), '-'], input_text='second')
            _run([str(tool), '--verify-atlas-timing', '-'], input_text='second')
            with patch.dict(os.environ, {'ATLAS_TEST_TOOL_IDENTITY': 'changed'}):
                _run([str(tool), '-'], input_text='first')
            tool.write_bytes(b'second tool')
            _run([str(tool), '-'], input_text='first')
            assert run.call_count == 4


def test_output_file_invocations_are_never_cached():
    with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
        with materialization_session(enabled=True):
            for _ in range(2):
                _run(['atlas-opt', '-o', 'output.mlir', '-'], input_text='module {}')
        assert run.call_count == 2
