"""Mandatory options have no dispatch recipe or optimization switches."""
from dataclasses import fields
import pytest
from atlas_mlir_native.options import CodegenOptions, configure_codegen_parser
from merlin.compiler.semantic.codegen import CodegenOptions as GenericOptions

def test_parser_has_no_optimized_pipeline_switches():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    configure_codegen_parser(parser)
    options = {option for action in parser._actions for option in action.option_strings}
    for name in ("recipe", "adjacent-patterns", "exponent-page-cache", "exact-binary-lane-groups", "exact-binary-vector-preflight",
                 "exact-cast-vector-preflight", "static-quantizer-sizing", "exact-binary-routes", "capture-postspill",
                 "kernel-library", "device-kv-quantizer"):
        assert "--" + name not in options and "--no-" + name not in options

def test_generic_options_do_not_contain_atlas_choices():
    atlas = {field.name for field in fields(CodegenOptions)} - {field.name for field in fields(GenericOptions)}
    assert {"native_pipeline", "adjacent_patterns", "quantized_operand_reuse", "device_kv_quantizer"} <= atlas
