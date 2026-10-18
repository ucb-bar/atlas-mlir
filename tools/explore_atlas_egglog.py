#!/usr/bin/env python3
"""Compose exact Atlas families and emit a bounded, auditable shortlist.

Use an explicitly selected Merlin src on PYTHONPATH. This tool generates
candidates; select_atlas_candidates.py compiles and measures them. No estimate
is a runtime result, and a rejected materialization never replaces native.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import inspect
from pathlib import Path
import time

from egglog.deconstruct import get_callable_args, get_literal_value
from merlin.semantic_compiler.egraph_terms import saturate, extract, unforest
from merlin.semantic_compiler.rewrite_selection import RewriteCandidate, shortlist
from atlas_egglog_ir import load_program, parse_program, pure_roots, decode_program
from atlas_egglog_schema import Value, encode_node, decode_node
from atlas_egglog_facts import analyze_program
from atlas_egglog_rules import EXACT_RULES, register_facts
from atlas_egglog_pack import RECIPE, pack_rules, recipe, restore_native, choices_from_roots, materialize
from atlas_egglog_cost import TimingCosts


def digest(raw):return hashlib.sha256(raw).hexdigest()

def implementation_pins():
    import atlas_egglog_ir, atlas_egglog_schema, atlas_egglog_facts, atlas_egglog_rules, atlas_egglog_pack, atlas_egglog_cost
    from merlin.semantic_compiler import egraph_terms, rewrite_selection
    modules=(atlas_egglog_ir,atlas_egglog_schema,atlas_egglog_facts,atlas_egglog_rules,atlas_egglog_pack,atlas_egglog_cost,egraph_terms,rewrite_selection)
    paths=[Path(inspect.getfile(module)).resolve() for module in modules]+[Path(__file__).resolve()]
    return {str(path):digest(path.read_bytes()) for path in paths}

def search(program, timing, *, attempts=18, limit=8, seed=194):
    if not 2<=attempts<=64 or not 3<=limit<=64:raise ValueError('bounded attempts/shortlist required')
    started=time.perf_counter();roots=pure_roots(program);facts=analyze_program(program)
    admitted=[op for op in program.operations if op.op=='virtual_pack_fp8' and facts.values[op.operands[0]].finite]
    seed_nodes=list(roots.values())
    for op in admitted:
        for unit in (0,1):seed_nodes.append(recipe(program.nodes[op.results[0]],unit).operands[1])
    proof=dict(schema='atlas.exact-egglog.v1',source_sha256=program.sha256,
        assumptions='derived_only',cost_facts_sha256=timing.sha256,
        finite_sites=[op.results[0] for op in admitted],
        facts={name:asdict(value) for name,value in facts.values.items()},
        families=['copy','transpose_involution','ordered_minmax','guarded_relu','guarded_pack'],
        scope='Straight-line virtual dataflow; original effect/resource sequence external; real allocation and measured byte gates required.')
    certificate=digest(json.dumps(proof,sort_keys=True).encode())
    candidates=[];details={};rejected=[]
    def remember(name,text,kind,choices=()):
        parsed=parse_program(text);cost=timing.estimate(parsed)
        families=[]
        if choices:families.append('guarded_pack')
        if kind=='egglog' and text!=''.join(program.lines):families.append('joint_exact_rules')
        c=RewriteCandidate(name,text.encode(),certificate,tuple(families),cost)
        candidates.append(c);details[name]=dict(kind=kind,source_sha256=c.source_sha256,choices=[dict(result=x.result,unit=x.unit,scale=x.scale) for x in choices],cost=asdict(cost))
    remember('native',''.join(program.lines),'native')
    # Explicit hand controls apply the same derived finite proof and full clamp
    # equation, uniformly assigning each admitted conversion to one engine.
    for unit in (0,1):
        hand=dict(roots)
        for op in admitted:hand[op.results[0]]=recipe(program.nodes[op.results[0]],unit)
        choices=choices_from_roots(program,hand)
        try:remember(f'hand-u{unit}',materialize(program,choices),'hand',choices)
        except ValueError as error:rejected.append(dict(id=f'hand-u{unit}',stage='materialization',reason=str(error)))
    if roots:
        result=saturate({name:encode_node(node) for name,node in roots.items()},rules=EXACT_RULES|pack_rules(program,facts),
            register_facts=lambda graph:register_facts(graph,seed_nodes,program_facts=facts,program=program))
        # Perturb only exact recipe ties. Mixed masks depend on source value
        # structure; unperturbed timing and pressure rescore every decoded IR.
        for attempt in range(attempts):
            def marginal(graph,node):
                base=timing.marginal(graph,node);args=get_callable_args(node,Value.apply)
                if args is None or get_literal_value(args[0])!=RECIPE:return base*1024
                attrs=dict(json.loads(get_literal_value(args[1])));unit=int(attrs['unit'].split(':')[0])
                if attempt<2:preferred=attempt
                else:
                    source_key=json.loads(attrs['source_key'])
                    preferred=hashlib.sha256((source_key+':'+str(seed+attempt)).encode()).digest()[0]&1
                return base*1024+int(unit!=preferred)
            extracted,_=extract(result,marginal_cost=marginal)
            selected={name:decode_node(value) for name,value in extracted.items()}
            choices=choices_from_roots(program,selected)
            try:
                # Decode every pure root before external state insertion. Keep
                # PACK roots alive until their selected recipes are inserted.
                native_roots={name:restore_native(node) for name,node in selected.items()}
                decoded=decode_program(program,native_roots)
                converted=materialize(parse_program(decoded),choices)
                final_program=parse_program(converted)
                final=decode_program(final_program,pure_roots(final_program),prune_dead_pure=True)
                remember(f'egg-{attempt:02}',final,'egglog',choices)
            except ValueError as error:rejected.append(dict(id=f'egg-{attempt:02}',stage='materialization',reason=str(error)))
        stats=dict(rounds=result.rounds,function_rows=result.function_rows,stop_reason=result.stop_reason)
    else:stats=dict(rounds=0,function_rows=0,stop_reason='no_pure_values')
    controls=tuple(c.id for c in candidates if c.id.startswith('hand-'))
    chosen=shortlist(candidates,baseline='native',required=controls,limit=limit)
    return chosen,dict(proof=proof,certificate_sha256=certificate,details=details,rejected=rejected,
        candidate_pool=len(candidates),distinct_sources=len({c.source_sha256 for c in candidates}),
        selected=[c.id for c in chosen],stats=stats,seconds=time.perf_counter()-started,
        estimate_scope=timing.omitted,attempts=attempts,seed=seed,
        hand_scope='Uniform engine0 and engine1 with full exact clamps, same automatically derived guards; not exhaustive enumeration.')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source',type=Path);ap.add_argument('--timing-facts',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--attempts',type=int,default=18);ap.add_argument('--limit',type=int,default=8);args=ap.parse_args()
    program=load_program(args.source);timing=TimingCosts(args.timing_facts);candidates,report=search(program,timing,attempts=args.attempts,limit=args.limit)
    args.output.mkdir(parents=True,exist_ok=True)
    if any(args.output.iterdir()):raise ValueError('output directory must be empty')
    proof=report['proof']
    proof['qualification']='Provider derivation declaration; candidate admission still requires compiler verification and exact measured output gates.'
    proof['implementation_pins']=implementation_pins()
    proof['candidate_bindings']={c.id:dict(source_sha256=c.source_sha256,choices=report['details'][c.id]['choices'],families=list(c.families)) for c in candidates}
    proof_path=args.output/'proof.json';proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    rows=[]
    for candidate in candidates:
        source=args.output/(candidate.id+'.mlir');source.write_bytes(candidate.source)
        rows.append(dict(id=candidate.id,source=str(source.resolve()),source_sha256=candidate.source_sha256,
            admitted=True,estimated_cost=candidate.cost.score,proof=dict(path=str(proof_path.resolve()),sha256=digest(proof_path.read_bytes()),schema=proof['schema']),families=candidate.families))
    (args.output/'candidates.json').write_text(json.dumps(dict(candidates=rows),indent=2)+'\n')
    (args.output/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='GENERATED_NOT_MEASURED',selected=report['selected'],rejected=report['rejected'],seconds=report['seconds'])))


if __name__=='__main__':main()
