"""The pure tool cache preserves output and checks every execution input."""

import subprocess

import pytest

from atlas_mlir_native import binding
from merlin.semantic_compiler.codegen import CodegenOptions, codegen_options
from merlin.semantic_compiler.materialization import materialization_session
from merlin.semantic_compiler.target_binding import NativeCompilationError


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    executable = tmp_path / "atlas-opt"
    executable.write_bytes(b"compiler-v1")
    calls = []

    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "compiled output\n", "diagnostic\n")

    monkeypatch.setattr(binding.subprocess, "run", invoke)
    return executable, calls


def test_stdout_and_stderr_replayed_with_fresh_equivalence(compiler):
    executable, calls = compiler
    outputs = []
    for enabled in (False, True):
        diagnostics = []
        with materialization_session(enabled=enabled) as stats:
            result = [binding._run([str(executable), "--atlas-compile", "-"], input_text="module {}",
                                   diagnostics=diagnostics) for _ in range(2)]
        assert stats["hits"] == int(enabled)
        assert stats["misses"] == (1 if enabled else 2)
        outputs.append((result, diagnostics))
    assert outputs[0] == outputs[1]
    assert len(calls) == 3


def test_execution_inputs_and_executable_changes_miss(compiler, monkeypatch, tmp_path):
    executable, calls = compiler
    command = [str(executable), "--atlas-compile", "-"]
    with materialization_session(enabled=True) as stats:
        binding._run(command, input_text="module {}")
        binding._run(command, input_text="module {changed}")
        binding._run([str(executable), "--canonicalize", "-"], input_text="module {}")
        with codegen_options(CodegenOptions(program_size_limit_bytes=4096)):
            binding._run(command, input_text="module {}")
        monkeypatch.setenv("ATLAS_TOOL_CACHE_TEST", "changed")
        binding._run(command, input_text="module {}")
        monkeypatch.delenv("ATLAS_TOOL_CACHE_TEST")
        executable.write_bytes(b"compiler-v2-with-different-size")
        binding._run(command, input_text="module {}")
        directory = tmp_path / "working"
        directory.mkdir()
        monkeypatch.chdir(directory)
        binding._run(command, input_text="module {}")
    assert stats["hits"] == 2
    assert stats["misses"] == len(calls) == 5


def test_failed_tool_diagnostics_are_preserved_and_retry_executes(compiler, monkeypatch):
    executable, calls = compiler
    results = iter((subprocess.CompletedProcess([], 1, "", "rejected"),
                    subprocess.CompletedProcess([], 0, "valid", "accepted")))

    def invoke(*args, **kwargs):
        calls.append(True)
        return next(results)

    monkeypatch.setattr(binding.subprocess, "run", invoke)
    diagnostics = []
    with materialization_session(enabled=True) as stats:
        with pytest.raises(NativeCompilationError, match="rejected"):
            binding._run([str(executable), "-"], input_text="input", diagnostics=diagnostics)
        assert binding._run([str(executable), "-"], input_text="input", diagnostics=diagnostics) == "valid"
        assert binding._run([str(executable), "-"], input_text="input", diagnostics=diagnostics) == "valid"
    assert diagnostics == ["rejected", "accepted", "accepted"]
    assert stats["misses"] == len(calls) == 2
    assert stats["hits"] == 1


def test_output_files_and_unknown_tools_are_never_reused(compiler):
    executable, calls = compiler
    unknown = executable.with_name("other-tool")
    unknown.write_bytes(b"compiler")
    with materialization_session(enabled=True) as stats:
        for _ in range(2):
            binding._run([str(executable), "-o", "output", "-"], input_text="input")
            binding._run([str(unknown), "-"], input_text="input")
    assert len(calls) == 4
    assert stats["calls"] == 0
