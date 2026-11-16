"""Shared scheduling chooses actual delays; the OOT only verifies and encodes."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("merlin.compile.scheduling.static_schedule")
from atlas_mlir_native.shared_static_schedule import compile_fixed

ROOT = Path(__file__).resolve().parents[1]
BINS = Path(os.environ.get("ATLAS_OOT_BIN_DIR", str(ROOT / "build/bin")))


@pytest.mark.parametrize(
    "fixture,register,expected", [("vpu", 6, 1), ("matmul", 2, 32)]
)
def test_shared_schedule_emits_verified_words_and_executes(
    tmp_path, fixture, register, expected
):
    if not (BINS / "atlas-emit").exists():
        pytest.skip("build selected OOT tools")
    dst = tmp_path / "compiled"
    manifest = compile_fixed(
        (ROOT / f"test/examples/shared_static_{fixture}.mlir").read_text(),
        destination=dst,
        atlas_bin_dir=BINS,
    )
    assert manifest["scheduler"] == "merlin-static-source-order-serial"
    plan = (dst / "schedule.mlir").read_text()
    assert "schedule.drain" in plan and "schedule.time_annotation" in plan
    assert manifest["words"] > (7 if fixture == "vpu" else 8)
    # Removing all generated delays must fail final timing verification.
    rows = json.loads((dst / "program.json").read_text())["instructions"]
    from atlas_mlir_native.shared_static_schedule import render

    bad = tmp_path / "bad.mlir"
    bad.write_text(render([r for r in rows if r["operation"] != "atlas.delay"]))
    result = subprocess.run(
        [str(BINS / "atlas-opt"), str(bad), "--verify-atlas-timing"],
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0

    model = pytest.importorskip("npu_model")
    from atlas_mlir_native.shared_dense import MODEL_REVISION

    model_root = Path(model.__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(model_root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == MODEL_REVISION
    subprocess.run(
        [
            "git",
            "-C",
            str(model_root),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "npu_model",
        ],
        check=True,
        capture_output=True,
    )
    import torch
    from npu_model.configs.hardware.default import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram

    from test.run_shared_model_case import instructions
    from tools.check_program_decode import audit_program
    from tools.check_variant_inventory import load_inventory

    binary = (dst / "program.bin").read_bytes()
    assert (
        audit_program(
            load_inventory(), json.loads((dst / "program.json").read_text()), binary
        )["status"]
        == "PASS"
    )
    program = InstantiableProgram(instructions(binary))
    config = DefaultHardwareConfig()
    config.arch_state_config = replace(config.arch_state_config, numerics="rtl")
    sim = Simulation(
        config,
        LoggerConfig(filename=str(tmp_path / "trace.json")),
        program,
        verbose=False,
    )
    try:
        sim.run(max_cycles=10000)
        assert sim.core.is_finished() and not sim.get_stats().runtime_errors
        # VLI(-1) -> ReLU -> + VLI(1), independently expected as exact BF16 1.
        actual = sim.core.arch_state.read_mrf_bf16_tile(register)
        torch.testing.assert_close(
            actual, torch.full_like(actual, expected), rtol=0, atol=0
        )
        (dst / "execution.json").write_text(
            json.dumps(
                {
                    "correct": True,
                    "cycles": sim.get_stats().cycles,
                    "model_revision": revision,
                }
            )
        )
    finally:
        sim.close()


def test_event_work_cannot_be_given_fixed_completion(tmp_path):
    if not (BINS / "atlas-emit").exists():
        pytest.skip("build selected OOT tools")
    with pytest.raises(subprocess.CalledProcessError) as exc:
        compile_fixed(
            (ROOT / "test/examples/vpu_relu_pair.mlir").read_text(),
            destination=tmp_path / "event",
            atlas_bin_dir=BINS,
        )
    assert "separate completion boundary" in exc.value.stderr
