"""Reusable source-footprint event and operand-relation algebra.

This module contains no measured catalog reader, observation fitter, artifact
projector, or pinned hardware capacity. Structural rules come from the guarded
source contract supplied by atlas_derive_overlap.SourceRules.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import re
from atlas_model_diagnostics import ModelTimingEvidenceError
from atlas_model_schema import RELATION_KEYS, RELATION_VALUES


def relation_key(relation):
    return tuple(sorted(relation.items()))

def relation_class(relation):
    result = dict(relation)
    banks = result['mreg_banks']
    if banks != 'disjoint':
        result['mreg_banks'] = 'mixed_alias' if '=' in banks and '~' in banks else 'register_alias' if '=' in banks else 'bank_alias'
    return result

def engine(name):
    return name.split('.', 1)[0]

@dataclass(frozen=True)
class Register:
    write: bool
    requests: tuple[tuple[int, int], ...]
    release: int


@lru_cache(maxsize=None)
def bank_bound(left, right, sign, coalescing):
    """Suffix-safe bank bound for immutable source event shapes.

    Many opcodes share the same row planners. Cache their event algebra, not
    a measured gap or a concrete benchmark relation.
    """
    bound = 1
    for age, row in left.requests:
        for later, later_row in right.requests:
            if left.write == right.write:
                shared = not left.write and coalescing and sign == '=' and row == later_row
                collision = not shared
            else:
                collision = sign == '=' and row == later_row
            if collision and age >= later:
                bound = max(bound, age - later + 1)
    return bound

def validate_relation(relation):
    if not isinstance(relation, dict) or not set(relation) <= RELATION_KEYS or 'mreg_banks' not in relation:
        raise ModelTimingEvidenceError('unsupported or missing relation constraint')
    for key, value in relation.items():
        if key == 'mreg_banks':
            if value != 'disjoint' and (not isinstance(value, str) or not re.fullmatch('\\d+[=~]\\d+(,\\d+[=~]\\d+)*', value)):
                raise ModelTimingEvidenceError('invalid MREG relation')
        elif not isinstance(value, str) or value not in RELATION_VALUES[key]:
            raise ModelTimingEvidenceError('invalid relation qualifier')
    mode = relation.get('scalar_registers')
    if mode is None:
        if set(relation) & {'scalar_destination_file', 'scalar_other_registers'}:
            raise ModelTimingEvidenceError('scalar roles lack a joint register mode')
    elif mode == 'independent':
        if 'scalar_destination_file' in relation:
            raise ModelTimingEvidenceError('independent roles cannot specify a destination alias')
    else:
        if relation.get('scalar_other_registers') != 'independent':
            raise ModelTimingEvidenceError('scalar alias must constrain other registers')
        if mode == 'shared_base_read_only':
            if 'scalar_destination_file' in relation:
                raise ModelTimingEvidenceError('shared read-only base has no destination alias')
        elif relation.get('scalar_destination_file') not in ({'xrf', 'erf'} if mode == 'same_load_destination' else {'xrf'}):
            raise ModelTimingEvidenceError('scalar alias has an invalid destination file')
    if relation.get('scalar_address_alignment') == 'explicit_byte_offsets':
        for side in ('first', 'second'):
            if relation.get(side + '_mnemonic') not in {'vload', 'vstore'} and side + '_scalar_byte_offset' not in relation:
                raise ModelTimingEvidenceError('explicit alignment lacks scalar byte offset')
    elif any((key.endswith('scalar_byte_offset') for key in relation)):
        raise ModelTimingEvidenceError('scalar byte offsets require explicit alignment')

class RuleSet:

    def __init__(self, per_op):
        if per_op.get('schema') != 'atlas.model_timing.v1' or per_op.get('age_convention') != 'issue_edge_zero':
            raise ModelTimingEvidenceError('unsupported isolated footprint schema')
        self.blocks = {block['name']: block for block in per_op['op_timing'] if block.get('resolved')}
        self.model = per_op['model']
        self.releases = per_op.get('reservation_releases', {})
        if not isinstance(self.releases, dict):
            raise ModelTimingEvidenceError('invalid isolated reservation releases')

    def first_free_age(self, name):
        return self.blocks[name]['first_free_age']

    @staticmethod
    def kind(name):
        part = name.split('.', 1)[1]
        if part.startswith('matmul'):
            return 'compute'
        if part == 'push_weight':
            return 'weight'
        if part.startswith('push_acc'):
            return 'push_bf16' if part.endswith('bf16') else 'push_fp8'
        if part.startswith('pop_acc'):
            return 'pop_bf16' if part.endswith('bf16') else 'pop_fp8'
        return part

    def ages(self, name, key):
        event = self.blocks[name]['events'][key]
        return tuple(range(event['first_age'], event['last_age'] + 1, event['step']))

    @lru_cache(maxsize=None)
    def registers(self, name):
        events = self.blocks[name]['events']
        unit, part = name.split('.', 1)
        reads, writes = ([], [])

        def add(key, write, count=1):
            ages = self.ages(name, key)
            direction = 'mreg_write' if write else 'mreg_read'
            operation = self.releases.get(name)
            release = operation.get(direction) if isinstance(operation, dict) else None
            if not isinstance(release, dict) or type(release.get('first_age')) is not int or release['first_age'] != 0 or (type(release.get('release_age')) is not int) or (type(release.get('available_age')) is not int) or (release['available_age'] != release['release_age'] + 1) or (not max(ages) < release['available_age'] <= self.first_free_age(name)):
                raise ModelTimingEvidenceError(f'missing or invalid {direction} reservation release for {name}')
            for bank in range(count):
                requests = tuple(((age, row % 32) for row, age in enumerate(ages) if row // 32 % count == bank))
                (writes if write else reads).append(Register(write, requests, release['available_age']))
        if unit == 'vpu':
            row_reduce = part in {'rsum', 'rmax', 'rmin'}
            if row_reduce:
                for key in ('read0', 'read1', 'write0', 'write1'):
                    add(key, key.startswith('write'))
            else:
                for key in ('read0', 'read1'):
                    if key in events:
                        add(key, False, 1 if part == 'fp8unpack' else 2)
                add('write0', True, 1 if part in {'fp8pack', 'vliCol', 'vliOne'} else 2)
        elif unit in {'mxu0', 'mxu1'}:
            for key in ('mreg_read0', 'mreg_read1', 'mreg_write0', 'mreg_write1'):
                if key in events:
                    add(key, key.startswith('mreg_write'))
        elif unit == 'xlu':
            add('read', False)
            add('write', True)
        elif unit == 'vlsu':
            add('mreg_write' if part == 'vload' else 'mreg_read', part == 'vload')
        elif unit != 'scalar_lsu':
            raise ModelTimingEvidenceError('unmodelled engine: ' + unit)
        return tuple(reads + writes)

    @lru_cache(maxsize=None)
    def bounds(self, first, second, relation_items):
        relation = dict(relation_items)
        if first not in self.blocks or second not in self.blocks:
            raise ModelTimingEvidenceError('operation lacks isolated footprint')
        unknown = set(relation) - {'mreg_banks', 'accumulator', 'weight_slot', 'vmem_bank', 'vmem_window', 'vmem_word_window', 'first_mnemonic', 'second_mnemonic', 'scalar_registers', 'scalar_destination_file', 'scalar_other_registers', 'scalar_address_alignment', 'first_scalar_byte_offset', 'second_scalar_byte_offset'}
        if unknown:
            raise ModelTimingEvidenceError('unsupported relation keys: ' + ','.join(sorted(unknown)))
        bounds = dict(structural_issue=self._structural_bound(first, second, relation), logical_reservation=1, bank_or_row=1, vmem_bank=1)
        ra, rb = (self.registers(first), self.registers(second))
        banks = relation.get('mreg_banks')
        if banks is None:
            raise ModelTimingEvidenceError('missing complete MREG relation')
        if banks != 'disjoint':
            for term in banks.split(','):
                match = re.fullmatch('(\\d+)([=~])(\\d+)', term)
                if not match:
                    raise ModelTimingEvidenceError('unsupported MREG relation term')
                i, sign, j = (int(match[1]), match[2], int(match[3]))
                if i >= len(ra) or j >= len(rb):
                    raise ModelTimingEvidenceError('MREG relation operand outside isolated footprint')
                left, right = (ra[i], rb[j])
                if sign == '=' and (left.write or (right.write and second != 'vlsu.vload')):
                    bounds['logical_reservation'] = max(bounds['logical_reservation'], left.release)
                bounds['bank_or_row'] = max(bounds['bank_or_row'],
                    bank_bound(left, right, sign, engine(first) == engine(second) == 'vpu'))
        if relation.get('vmem_bank') == 'same':
            ea, eb = (self.blocks[first]['events'], self.blocks[second]['events'])
            aa = [age for key in ('vmem_read', 'vmem_write') if key in ea for age in self.ages(first, key)]
            bb = [age for key in ('vmem_read', 'vmem_write') if key in eb for age in self.ages(second, key)]
            if aa and bb:
                bounds['vmem_bank'] = max(bounds['vmem_bank'], max(aa) - min(bb) + 1)
        return bounds

    def minimum_gap(self, first, second, relation):
        """First gap of the suffix with no derived collision at any larger gap."""
        validate_relation(relation)
        return max(self.bounds(first, second, relation_key(relation)).values())


    def _structural_bound(self, first, second, relation):
        raise NotImplementedError('source-derived structural contract required')
