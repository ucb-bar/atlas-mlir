"""Immutable Atlas export intake for model2MLIR's external adapter protocol."""
from .adapter import atlas_fp8_adapter, load_program, tensor_record

__all__ = ["atlas_fp8_adapter", "load_program", "tensor_record"]
