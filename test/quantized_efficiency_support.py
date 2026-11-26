"""Shared declared geometry for retained prefix-transfer regression controls."""
from dataclasses import replace
from atlas_mlir_native.options import codegen_options, current_codegen_options


def options():
    return codegen_options(replace(current_codegen_options(), io_memory_bytes=67108864,
                                   execution_instruction_limit=128000000, exponent_page_cache=True))
