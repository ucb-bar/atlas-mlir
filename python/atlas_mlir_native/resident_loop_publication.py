"""Replay a shared resident runtime against original native and current sources.

The descriptor is an untrusted publication. Its hashes identify content but
cannot prove that the compiler admitted an allocation or an instruction edit.
Only fresh replay of the original template and ordered source requests does so.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from merlin.compiler.semantic.model import KernelRequest

from .resident_loop_image import BoundResidentImage, ResidentLoopImage


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _frozen(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _frozen(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_frozen(item) for item in value)
    return value


@dataclass(frozen=True)
class CheckedResidentLoopRuntime:
    """Fresh compiler-owned images, ordered exactly as the supplied source calls.

    Bound images are freshly constructed, with no aliases into published JSON.
    The immutable image receipt describes the one generated template ABI.
    """

    bound: tuple[BoundResidentImage, ...]
    image_receipt: Mapping[str, object]
    table_sha256: str
    runtime_sha256: str


def verify_resident_loop_runtime(
    representative_source: KernelRequest,
    compiled_root: Path,
    sources: Sequence[KernelRequest],
    published: dict,
    *,
    name: str,
    target_source: Path,
) -> CheckedResidentLoopRuntime:
    """Authenticate a publication under the active complete codegen options.

    Callers supply ordered current sources from their source/dispatch contract;
    neither the descriptor nor a self-consistent set of its hashes supplies
    source truth. This API admits no deployment placement by itself.
    """
    if type(published) is not dict:
        raise ValueError('resident runtime publication must be a JSON object')
    sources = tuple(sources)
    if not sources or not all(isinstance(source, KernelRequest) for source in sources):
        raise ValueError('resident runtime requires ordered current source requests')
    image = ResidentLoopImage(representative_source, compiled_root, target_source=target_source)
    emitted = image.emit_c(name+'_resident_image')
    bound = tuple(image.bind(source) for source in sources)
    rows = [dict(binding=current.receipt, execution_plan=current.execution_plan,
                 addresses=list(image.binding_addresses(current))) for current in bound]
    expected = dict(schema='atlas.resident_loop_c_runtime.v1', image=emitted.receipt,
                    trip_bindings=rows, mutable_image_storage='source_context_workspace_tail',
                    load_convention='forced_load_each_dispatch',
                    table_sha256=sha256(_encoded(rows)).hexdigest())
    # Byte comparison distinguishes JSON true from 1 and refuses extra fields.
    # Every expected byte comes from the authenticated native template, current
    # source policies and current allocations, rather than published assertions.
    encoded = _encoded(expected)
    if _encoded(published) != encoded:
        raise ValueError('resident runtime publication differs from current source replay')
    return CheckedResidentLoopRuntime(bound, _frozen(emitted.receipt), expected['table_sha256'],
                                      sha256(encoded).hexdigest())
