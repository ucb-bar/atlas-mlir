"""Scoped compiler recipe selection for provider tests."""
from atlas_mlir_native.options import codegen_options, current_codegen_options
from contextlib import ExitStack
from dataclasses import replace

import pytest


@pytest.fixture
def select_codegen_options():
    """Apply successive overrides and restore the caller's options after a test."""
    with codegen_options(current_codegen_options()), ExitStack() as stack:
        def select(**updates):
            return stack.enter_context(codegen_options(replace(current_codegen_options(), **updates)))
        yield select
