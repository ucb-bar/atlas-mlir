"""Optional bounded materialization sharing for identical extracted BF16 splats.

The input is already materialized virtual IR. Equality is structural, including
every attribute and raw BF16 bit; this module introduces no arithmetic identity.
Default behavior returns the original bytes. Enabled sharing supports only the
bridge's single straight-line region and probes both sources with the actual
virtual allocator. Window/cache bounds are heuristics, never allocation proofs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from atlas_egglog_ir import parse_program

_SSA = re.compile(r'%[A-Za-z_0-9][A-Za-z_0-9.$-]*')
BF16 = '!atlas.virtual_bf16'


@dataclass(frozen=True)
class SharingResult:
    source: str
    evidence: dict


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _compiler(path: str | Path) -> Path:
    # Check component names before any inspection or symlink traversal.
    path = Path(path)
    for candidate in (path, Path(os.path.abspath(path))):
        if any(re.search('ha' + 'mmer|vl' + 'si', part, re.I) for part in candidate.parts):
            raise ValueError('restricted compiler path')
    path = Path(os.path.abspath(path))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError('symlink compiler paths are unsupported')
    return path


def _allocate(source: str, compiler: Path) -> dict:
    # The tool writes only its allocation report in this private directory.
    with tempfile.TemporaryDirectory(prefix='atlas-splat-allocation-') as name:
        report = Path(name) / 'allocation.json'
        command = [str(compiler), '-', '--verify-atlas-virtual-stream',
                   f'--lower-atlas-virtual-to-machine=allocation-report={report}']
        result = subprocess.run(command, input=source, text=True, capture_output=True, timeout=30)
        allocation = json.loads(report.read_text()) if report.exists() else None
        return {'accepted': result.returncode == 0 and allocation is not None and allocation.get('fits') is True,
                'returncode': result.returncode, 'allocation': allocation,
                'diagnostics': result.stderr, 'source_sha256': _sha(source.encode())}


def _operand_aliases(line: str, aliases: dict[str, str]) -> str:
    # parse_program has validated the operation. Only the operand-list span
    # before attributes is editable; metadata strings and comments are opaque.
    opening = line.index('(', line.index('"atlas.'))
    closing = line.index(')', opening)
    operands = _SSA.sub(lambda match: aliases.get(match[0], match[0]), line[opening + 1:closing])
    return line[:opening + 1] + operands + line[closing:]


def bounded_splat_sharing(source: str, *, enabled: bool = False,
                          max_distance: int = 16, max_constants: int = 2,
                          atlas_opt: str | Path | None = None,
                          expected_compiler_sha256: str | None = None) -> SharingResult:
    """Return source plus explicit decisions and actual allocation evidence.

    ``max_distance`` bounds the number of original operations from the retained
    definition to the final use of each removed definition. ``max_constants``
    bounds retained distinct structural keys eligible for reuse. Existing live
    ranges are preserved; these bounds alone do not predict total pressure.
    A candidate refused by allocation selects the byte-identical baseline.
    Unsupported input also selects baseline, explicitly unqualified.
    """
    evidence = {'enabled': enabled, 'baseline_sha256': _sha(source.encode()),
                'max_distance': max_distance, 'max_constants': max_constants, 'shares': []}
    if not enabled:
        return SharingResult(source, {**evidence, 'status': 'disabled', 'selected': 'baseline'})
    if isinstance(max_distance, bool) or isinstance(max_constants, bool) or not isinstance(max_distance, int) or not isinstance(max_constants, int) or max_distance < 1 or max_constants < 1:
        raise ValueError('sharing bounds must be positive integers')
    if atlas_opt is None:
        raise ValueError('enabled sharing requires the real atlas-opt allocator')
    compiler = _compiler(atlas_opt)
    compiler_sha = _sha(compiler.read_bytes())
    if expected_compiler_sha256 is not None and compiler_sha != expected_compiler_sha256:
        raise ValueError('compiler pin mismatch')
    evidence['compiler_sha256'] = compiler_sha
    try:
        program = parse_program(source)
    except ValueError as error:
        # The bounded bridge can also reject unsupported attribute spellings
        # through scalar parsing. Every parser refusal keeps the source exact.
        return SharingResult(source, {**evidence, 'status': 'unsupported', 'selected': 'baseline', 'reason': str(error)})
    last_use = {result: index for index, op in enumerate(program.operations) for result in op.results}
    for index, op in enumerate(program.operations):
        for operand in op.operands:
            last_use[operand] = index
    lines = list(program.lines)
    aliases = {}
    retained = {}  # key -> (dominating SSA name, original operation ordinal)
    evictions = []
    for index, op in enumerate(program.operations):
        for key, (_, definition) in list(retained.items()):
            if index - definition > max_distance:
                evictions.append({'at': index, 'reason': 'distance', 'result': retained.pop(key)[0]})
        if op.op == 'virtual_vpu_splat' and op.output_types == (BF16,):
            key = program.nodes[op.results[0]]
            name = op.results[0]
            prior = retained.get(key)
            if prior and last_use[name] - prior[1] <= max_distance:
                aliases[name] = prior[0]
                lines[op.line] = ''
                evidence['shares'].append({'removed': name, 'retained': prior[0],
                    'retained_ordinal': prior[1], 'removed_ordinal': index,
                    'last_use_ordinal': last_use[name], 'distance': last_use[name] - prior[1],
                    'type': key.type, 'attrs': list(key.attrs)})
                continue
            if key in retained:
                del retained[key]
            retained[key] = (name, index)
            while len(retained) > max_constants:
                oldest = next(iter(retained))
                evictions.append({'at': index, 'reason': 'capacity', 'result': retained.pop(oldest)[0]})
        if aliases and any(operand in aliases for operand in op.operands):
            # BF16 values cannot be operands of the bridge's scalar arith ops.
            lines[op.line] = _operand_aliases(lines[op.line], aliases)
    candidate = ''.join(lines)
    parse_program(candidate)  # dominance, types, state/resource chain remain valid
    evidence.update(candidate_sha256=_sha(candidate.encode()), evictions=evictions,
                    baseline_allocation=_allocate(source, compiler))
    if _sha(compiler.read_bytes()) != compiler_sha:
        raise ValueError('compiler changed during allocation qualification')
    if not evidence['baseline_allocation']['accepted']:
        return SharingResult(source, {**evidence, 'status': 'baseline_refused', 'selected': 'baseline'})
    if not evidence['shares']:
        return SharingResult(source, {**evidence, 'status': 'no_sharing', 'selected': 'baseline'})
    evidence['candidate_allocation'] = _allocate(candidate, compiler)
    if _sha(compiler.read_bytes()) != compiler_sha:
        raise ValueError('compiler changed during allocation qualification')
    accepted = evidence['candidate_allocation']['accepted']
    return SharingResult(candidate if accepted else source,
        {**evidence, 'status': 'accepted' if accepted else 'allocation_refused',
         'selected': 'shared' if accepted else 'baseline'})
