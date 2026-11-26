"""Scoped compiler recipe selection for provider tests."""
from atlas_mlir.deployment_gate import CHECKER
from atlas_mlir_native.options import CodegenOptions, codegen_options, current_codegen_options
from contextlib import ExitStack
from dataclasses import replace
import sys

import pytest

# Tests import the checker's runtime by module name, as check_model.py and the step workers do.
sys.path.insert(0, str(CHECKER))



@pytest.fixture
def select_codegen_options():
    """Apply successive overrides and restore the caller's options after a test."""
    with codegen_options(current_codegen_options()), ExitStack() as stack:
        def select(**updates):
            return stack.enter_context(codegen_options(replace(current_codegen_options(), **updates)))
        yield select
