"""Preset expansion is an alias for checked, individually recorded choices."""
from argparse import Namespace
from dataclasses import fields

import pytest
from atlas_mlir_native.options import (
    CodegenOptions, DISPATCH_REDUCTION, SOPHIALAKE, options_from_args,
)
from merlin.compiler.semantic.codegen import CodegenOptions as GenericOptions


def test_recipe_matches_explicit_fields_and_has_no_alias_identity():
    preset = options_from_args(Namespace(recipe="dispatch-reduction"))
    explicit = CodegenOptions(**DISPATCH_REDUCTION)
    assert preset == explicit
    assert preset.identity_record() == explicit.identity_record()
    assert "recipe" not in preset.identity_record()
    assert preset.device_kv_quantizer


def test_board_matches_explicit_fields_and_excludes_host_reservation():
    preset = options_from_args(Namespace(board="sophialake",
        host_memory_reserved_bytes=4096, io_memory_bytes=6144))
    explicit = CodegenOptions(io_memory_base=0x80001000, io_memory_capacity_base=0x80001000,
        io_memory_capacity_bytes=(1 << 30) - 4096, io_memory_bytes=6144,
        application_residency=True, quantized_operand_reuse="wide-cache")
    assert preset.identity_record() == explicit.identity_record()
    assert preset.io_memory_base == SOPHIALAKE.ddr_base + 4096
    assert preset.io_memory_capacity_base + preset.io_memory_capacity_bytes == SOPHIALAKE.ddr_base + SOPHIALAKE.ddr_capacity_bytes
    assert preset.quantized_operand_reuse == "wide-cache"
    assert preset.native_pipeline == "mandatory"
    assert preset.application_residency
    assert preset.io_memory_bytes == 6144


@pytest.mark.parametrize("window", [None, 8 << 20, 16 << 20, 786432])
def test_recipe_board_refuses_known_spill_work_limit_before_memory_selection(monkeypatch, window):
    def unexpected_memory_selection(self):
        pytest.fail("unsupported preset combination reached board memory selection")
    monkeypatch.setattr(type(SOPHIALAKE), "options", unexpected_memory_selection)
    with pytest.raises(ValueError, match="known spill rewrite work limit reached with a 16 MiB staging window"):
        options_from_args(Namespace(recipe="dispatch-reduction", board="sophialake",
                                    io_memory_bytes=window))


def test_explicit_flags_override_recipe():
    override = options_from_args(Namespace(recipe="dispatch-reduction",
        native_pipeline="mandatory", device_kv_quantizer=False, application_residency=False,
        exponent_page_cache=False, io_memory_bytes=786432, quantized_operand_reuse="ordered"))
    assert override.native_pipeline == "mandatory"
    assert not override.device_kv_quantizer
    assert not override.application_residency
    assert not override.exponent_page_cache
    assert override.io_memory_bytes == 786432
    assert override.quantized_operand_reuse == "ordered"


def test_explicit_flags_override_board():
    override = options_from_args(Namespace(board="sophialake", host_memory_reserved_bytes=4096, application_residency=False,
                                          io_memory_bytes=786432, quantized_operand_reuse="ordered"))
    assert not override.application_residency
    assert override.io_memory_bytes == 786432
    assert override.quantized_operand_reuse == "ordered"


@pytest.mark.parametrize("presets", [
    ["--recipe", "dispatch-reduction", "--board", "sophialake"],
    ["--board", "sophialake", "--recipe", "dispatch-reduction"],
])
def test_cli_refuses_recipe_board_before_loading_compile_inputs(tmp_path, presets):
    from atlas_mlir_native.binding import AtlasNativeBinding
    from merlin.development.targetgen.cli import build_parser
    from merlin.development.targetgen.execution.native.native_codegen_cli import prepare_native_arguments

    binding = AtlasNativeBinding()
    args = build_parser(binding=binding).parse_args([
        "native-compile", "--engine", "merlin_native", "--support", "atlas",
        "--snapshot", str(tmp_path / "missing-snapshot"), "--compile-inputs", "missing-inputs",
        "--target-source", "missing-target", "--application", "--out", str(tmp_path / "output"),
        *presets,
    ])
    with pytest.raises(ValueError, match="spill rewrite work limit reached"):
        prepare_native_arguments(args, binding)
    assert not (tmp_path / "output").exists()


def test_explicit_host_reservation_changes_usable_geometry():
    options = options_from_args(Namespace(board="sophialake",
        host_memory_reserved_bytes=128 << 20, io_memory_bytes=2048))
    assert options.io_memory_capacity_base == 0x88000000
    assert options.io_memory_capacity_bytes == 0x38000000


@pytest.mark.parametrize("overrides", [
    dict(host_memory_reserved_bytes=0), dict(host_memory_reserved_bytes=1),
    dict(io_memory_capacity_base=0x80000000), dict(io_memory_capacity_bytes=1 << 30),
])
def test_board_refuses_geometry_outside_usable_memory(overrides):
    with pytest.raises(ValueError):
        measured = dict(host_memory_reserved_bytes=4096, io_memory_bytes=6144)
        measured.update(overrides)
        options_from_args(Namespace(board="sophialake", **measured))


@pytest.mark.parametrize("measured,flag", [
    ({}, "--host-memory-reserved-bytes"),
    (dict(io_memory_bytes=2048), "--host-memory-reserved-bytes"),
    (dict(host_memory_reserved_bytes=4096), "--io-memory-bytes"),
])
def test_board_requires_named_measurement_flags(measured, flag):
    with pytest.raises(ValueError, match=flag):
        options_from_args(Namespace(board="sophialake", **measured))


@pytest.mark.parametrize("host_bytes", [0, -2048, 1, True])
def test_board_requires_positive_measured_host_prefix(host_bytes):
    with pytest.raises(ValueError, match="host reservation must be positive"):
        options_from_args(Namespace(board="sophialake",
            host_memory_reserved_bytes=host_bytes, io_memory_bytes=2048))


def test_generic_options_do_not_contain_atlas_choices():
    assert not set(DISPATCH_REDUCTION).difference({"exact_float_unroll", "execution_instruction_limit", "io_memory_bytes"}).intersection(field.name for field in fields(GenericOptions))


def test_install_snapshot_and_application_defaults(monkeypatch):
    from merlin.development.targetgen.execution.native.native_codegen_cli import prepare_native_arguments
    monkeypatch.setenv("MERLIN_NATIVE_SNAPSHOT", "installed-snapshot")
    args = Namespace(snapshot=None, mode=None, application=True)
    prepare_native_arguments(args)
    assert args.snapshot == "installed-snapshot"
    assert args.mode == "diagnostic"


@pytest.mark.parametrize("option,manifest", [
    ("--no-exact-float-unroll", dict(exact_float_unroll=True, program_size_limit_bytes=None,
                                   execution_instruction_limit=None)),
    ("--no-application-residency", dict(application_residency=True)),
])
def test_verify_refuses_explicit_negative_override_of_published_policy(tmp_path, option, manifest):
    import json
    from atlas_mlir_native.binding import AtlasNativeBinding
    from merlin.development.targetgen.cli import build_parser
    from merlin.development.targetgen.execution.native.native_codegen_cli import prepare_native_arguments
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "status.json"
    binding = AtlasNativeBinding()
    args = build_parser(binding=binding).parse_args([
        "native-verify", "--engine", "merlin_native", "--support", "unused",
        "--snapshot", str(tmp_path / "snapshot"), "--compile-inputs", "inputs",
        "--application", "--artifact", str(artifact), "--out", str(output), option,
    ])
    prepare_native_arguments(args, binding)
    assert args.func(args) == 2
    assert "differs from compiled application" in json.loads(output.read_text())["reason"]
