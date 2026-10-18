"""Exact guarded PACK recipes for the shared Atlas egglog rule library.

Recipes are pure equality alternatives, not stateful operations. Materialize
only after extraction, with a free accumulator, locally rematerialized bounds,
virtual verification and the real allocator. A failed candidate retains native.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from egglog import EGraph, eq, rewrite, ruleset
from atlas_egglog_ir import Node, Program, parse_program
from atlas_egglog_schema import Value, Inputs, attrs_json, encode_node
from atlas_egglog_rules import EXACT_RULES, finite, register_facts
from atlas_egglog_facts import ProgramFacts, analyze_program, node_digest

BF16 = '!atlas.virtual_bf16'
FP8 = '!atlas.virtual_fp8'
STATE = '!atlas.virtual_state'
RECIPE = 'egg_pack_mxu'


def splat(bits):
    return Node('virtual_vpu_splat', (), (('bits', f'{bits} : i32'),), BF16)


def binary(kind, a, b):
    return Node('virtual_vpu_binary', (a,b), (('kind',json.dumps(kind)),), BF16)


def recipe(pack: Node, unit: int) -> Node:
    if pack.op != 'virtual_pack_fp8' or unit not in (0,1):
        raise ValueError('PACK recipe needs an original PACK and supported unit')
    scale=pack.attribute('scale_code'); source=pack.operands[0]; clamped=source
    if scale <= 246:
        cap=((scale+8)<<7)|96
        clamped=binary('min',binary('max',source,splat(cap|0x8000)),splat(cap))
    return Node(RECIPE,(source,clamped),tuple(sorted((('unit',f'{unit} : i32'),('scale_code',f'{scale} : i32'),('pack_attrs',json.dumps(attrs_json(pack.attrs))),('source_key',json.dumps(node_digest(source)))))),FP8)


def pack_rules(program: Program, facts: ProgramFacts):
    """Source-bound rules use shared finite facts and expose clamp expressions."""
    if facts.program_sha256 != program.sha256 or facts.contract_context != (False, ()):
        raise ValueError('PACK rules require current automatically derived facts')
    rules=[]
    for op in program.operations:
        if op.op != 'virtual_pack_fp8':continue
        node=program.nodes[op.results[0]]
        if not facts.values[op.operands[0]].finite:continue
        for unit in (0,1):
            rules.append(rewrite(encode_node(node)).to(encode_node(recipe(node,unit)),finite(encode_node(node.operands[0]))))
    return ruleset(*rules)


def restore_native(node: Node) -> Node:
    if node.op == RECIPE:
        attrs=tuple(tuple(pair) for pair in json.loads(json.loads(dict(node.attrs)['pack_attrs'])))
        return Node('virtual_pack_fp8',(restore_native(node.operands[0]),),attrs,FP8)
    return Node(node.op,tuple(restore_native(child) for child in node.operands),node.attrs,node.type)


@dataclass(frozen=True)
class PackChoice:
    result: str
    unit: int
    scale: int
    source: Node
    clamped: Node


def choices_from_roots(original: Program, roots: Mapping[str,Node]) -> tuple[PackChoice,...]:
    choices=[]
    for op in original.operations:
        if op.op != 'virtual_pack_fp8':continue
        node=roots[op.results[0]]
        if node.op == RECIPE:
            choices.append(PackChoice(op.results[0],node.attribute('unit'),node.attribute('scale_code'),restore_native(node.operands[0]),restore_native(node.operands[1])))
        elif node.op != 'virtual_pack_fp8':
            raise ValueError('unsupported extracted PACK implementation')
    return tuple(choices)


def materialize(program: Program, choices: tuple[PackChoice,...]) -> str:
    """Insert per-site recipes into a validated straight-line effect chain.

    Input program has already decoded all pure choices with recipes restored
    to their native forms. Each original accumulator version remains outside
    the graph. No constant sharing crosses a conversion site.
    """
    selected={c.result:c for c in choices}
    if len(selected)!=len(choices):raise ValueError('duplicate PACK selection')
    available={op.results[0] for op in program.operations if op.op=='virtual_pack_fp8'}
    if set(selected)-available:raise ValueError('selected PACK disappeared during decoding')
    facts=analyze_program(program)
    lines=list(program.lines); names={name for op in program.operations for name in op.results};counter=0
    state=None;pending_state=None;live_acc=[set(),set()]
    def fresh():
        nonlocal counter
        while True:
            name=f'%recipe{counter}';counter+=1
            if name not in names:names.add(name);return name
    def emit_op(results,op,operands,in_types,out_types,attrs=()):
        fields=', '.join(f'{key} = {value}' for key,value in attrs)
        attributes=f' {{{fields}}}' if fields else ''
        output=out_types[0] if len(out_types)==1 else '('+', '.join(out_types)+')'
        return f'    {", ".join(results)} = "atlas.{op}"({", ".join(operands)}){attributes} : ({", ".join(in_types)}) -> {output}\n'
    for op in program.operations:
        if op.input_types and op.input_types[0]==STATE:
            if pending_state is not None:
                old=op.operands[0];needle=f'"({old}'
                # Only the current operation's first state operand changes.
                if lines[op.line].count(needle)!=1:raise ValueError('state edge rendering not recognized')
                lines[op.line]=lines[op.line].replace(needle,f'"({pending_state}',1)
                pending_state=None
        if op.op in ('virtual_mxu_load_acc_bf16','virtual_mxu_load_acc_fp8','virtual_mxu_reset'):
            unit=int(op.output_types[1][-2]);live_acc[unit].add(op.results[1])
        elif op.op=='virtual_mxu_accumulate':
            unit=int(op.output_types[1][-2]);live_acc[unit].remove(op.operands[3]);live_acc[unit].add(op.results[1])
        elif op.op in ('virtual_mxu_readout_bf16','virtual_mxu_readout_fp8'):
            unit=int(op.input_types[1][-2]);live_acc[unit].remove(op.operands[1])
        if op.output_types and op.output_types[0]==STATE:
            state=op.results[0]
        if op.op!='virtual_pack_fp8' or op.results[0] not in selected:continue
        choice=selected[op.results[0]]
        if choice.unit not in (0,1) or len(live_acc[choice.unit])>=2 or state is None:
            raise ValueError('PACK recipe has no free accumulator/state edge')
        actual=program.nodes[op.results[0]]
        if choice.scale!=actual.attribute('scale_code') or choice.source!=actual.operands[0]:
            raise ValueError('PACK choice does not bind to the decoded producer/scale')
        if not facts.values[op.operands[0]].finite:
            raise ValueError('PACK source is not automatically finite in decoded program')
        canonical=recipe(actual,choice.unit).operands[1]
        proof=EGraph();proof.register(encode_node(canonical),encode_node(choice.clamped))
        register_facts(proof,(canonical,choice.clamped),program_facts=facts,program=program)
        proof.run(20,ruleset=EXACT_RULES)
        try:proof.check(eq(encode_node(canonical)).to(encode_node(choice.clamped)))
        except Exception as error:raise ValueError('extracted clamp lacks an exact equality proof') from error
        emitted=[];memo={choice.source:op.operands[0]}
        def emit_tree(node):
            if node in memo:return memo[node]
            if node.op not in ('virtual_vpu_splat','virtual_vpu_binary'):
                raise ValueError('unexpected operation in extracted clamp')
            if node.type!=BF16:raise ValueError('clamp requires BF16')
            operands=tuple(emit_tree(c) for c in node.operands);name=fresh()
            emitted.append(emit_op((name,),node.op,operands,tuple(c.type for c in node.operands),(node.type,),node.attrs));memo[node]=name;return name
        clamped=emit_tree(choice.clamped)
        scale=fresh();push=fresh();acc=fresh();done=fresh();acc_type=f'!atlas.virtual_mxu_acc<{choice.unit}>'
        emitted.append(emit_op((scale,),'virtual_scale_constant',(),(),('!atlas.virtual_scale',),(('code',f'{254-min(choice.scale,254)} : i32'),)))
        emitted.append(emit_op((push,acc),'virtual_mxu_load_acc_bf16',(pending_state or state,clamped),(STATE,BF16),(STATE,acc_type),(('unit',f'{choice.unit} : i32'),)))
        emitted.append(emit_op((done,choice.result),'virtual_mxu_readout_fp8',(push,acc,scale),(STATE,acc_type,'!atlas.virtual_scale'),(STATE,FP8)))
        lines[op.line]=''.join(emitted);pending_state=done
    if pending_state is not None:
        # A final PACK before return still requires a unique outgoing state use.
        expected=f'return {state} : {STATE}'
        matches=[i for i,line in enumerate(lines) if expected in line]
        if len(matches)!=1:raise ValueError('missing outgoing PACK state edge')
        lines[matches[0]]=lines[matches[0]].replace(expected,f'return {pending_state} : {STATE}')
    result=''.join(lines);parse_program(result);return result
