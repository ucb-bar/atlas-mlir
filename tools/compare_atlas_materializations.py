"""Compare exact hand/egglog candidates with identical bounded sharing policies.

This is an optional provider stage after dataflow extraction. Structural splat
identity permits reuse; the real allocator can refuse it and preserve baseline.
The output is a pool/shortlist, never a claim of measured performance.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from atlas_egglog_ir import load_program, parse_program
from atlas_egglog_cost import TimingCosts
from atlas_hand_reference import enumerate_hand
from atlas_egglog_sharing import bounded_splat_sharing, _compiler
from explore_atlas_egglog import search, implementation_pins


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compare(program, timing, *, atlas_opt, policies=((16, 2), (64, 2)), budget=4, attempts=64):
    """Apply the same policy set and measurement budget to both strategies."""
    if not 1 <= budget <= 16:
        raise ValueError('budget must be 1..16 distinct sources per strategy')
    if len(policies) != len(set(policies)) or len(policies) > 4:
        raise ValueError('at most four distinct sharing policies are supported')
    if any(type(d) is not int or type(c) is not int or d < 1 or c < 1 for d, c in policies):
        raise ValueError('sharing bounds must be positive integers')
    compiler = _compiler(atlas_opt)
    compiler_sha = digest(compiler.read_bytes())
    started = time.perf_counter()
    egg_candidates, report = search(program, timing, attempts=attempts, limit=64)
    source_by_sha = {candidate.source_sha256: candidate.source.decode() for candidate in egg_candidates}
    # Recover generation identities deduplicated by the shared shortlist.
    egg_base = []
    controls = []
    for name, detail in report['details'].items():
        if detail['source_sha256'] not in source_by_sha:
            raise ValueError('bounded base extraction omitted a distinct source')
        row = dict(id=name, source=source_by_sha[detail['source_sha256']], choices=detail['choices'],
                   strategy='egglog' if name.startswith('egg-') else 'control')
        (egg_base if name.startswith('egg-') else controls).append(row)
    hand_started = time.perf_counter()
    hand_base = []
    rejected = []
    for name, source, assignment, error in enumerate_hand(program):
        if name == 'native':
            continue
        if error:
            rejected.append(dict(id=name, reason=error, choices=assignment))
        else:
            hand_base.append(dict(id=name, source=source, choices=assignment, strategy='hand'))
    hand_seconds = time.perf_counter() - hand_started
    cache = {}
    decisions = []

    def expand(rows):
        expanded = []
        for base in rows:
            original_sha = digest(base['source'].encode())
            for policy in (None, *policies):
                if policy is None:
                    source = base['source']
                    evidence = dict(status='disabled', selected='baseline', baseline_sha256=original_sha, shares=[])
                    name = base['id']
                else:
                    key = (original_sha, policy)
                    if key not in cache:
                        cache[key] = bounded_splat_sharing(base['source'], enabled=True,
                            max_distance=policy[0], max_constants=policy[1], atlas_opt=compiler,
                            expected_compiler_sha256=compiler_sha)
                    result = cache[key]
                    source, evidence = result.source, result.evidence
                    name = f"{base['id']}-share-d{policy[0]}-c{policy[1]}"
                cost = timing.estimate(parse_program(source))
                applied = len(evidence['shares']) if evidence['selected'] == 'shared' else 0
                # Provider membership is not a claim that every local family fired.
                families = ['exact_atlas_provider']
                if applied:
                    families.append('identical_splat_sharing')
                expanded.append(dict(id=name, source=source, source_sha256=digest(source.encode()),
                    strategy=base['strategy'], choices=base['choices'], policy=policy,
                    effective_policy=policy if applied else None, applied_shares=applied,
                    families=families, cost=asdict(cost), score=cost.score))
                decisions.append(dict(id=name, baseline_id=base['id'], policy=policy, evidence=evidence))
        return expanded

    control_rows, hand_rows, egg_rows = expand(controls), expand(hand_base), expand(egg_base)
    all_rows = control_rows + hand_rows + egg_rows
    canonical = {}
    aliases = {}
    unique = []
    for row in all_rows:
        identity = row['source_sha256']
        if identity not in canonical:
            canonical[identity] = row['id']
            unique.append(row)
        aliases[row['id']] = canonical[identity]

    def best(rows):
        selected, seen = [], set()
        for row in sorted(rows, key=lambda r: (r['score'], r['source_sha256'], r['id'])):
            if row['source_sha256'] in seen:
                continue
            selected.append(row)
            seen.add(row['source_sha256'])
            if len(selected) == budget:
                break
        return selected

    selected = control_rows + best(hand_rows) + best(egg_rows)
    selected_ids = set(aliases[row['id']] for row in selected)
    if digest(compiler.read_bytes()) != compiler_sha:
        raise ValueError('compiler changed during candidate generation')
    clean = lambda row: {key: value for key, value in row.items() if key != 'source'}
    return unique, dict(base_search=report, policies=list(policies), compiler=dict(path=str(compiler), sha256=compiler_sha),
        hand_generation_seconds=hand_seconds, total_generation_seconds=time.perf_counter()-started,
        sharing_probe_keys=len(cache), hand_base_candidates=len(hand_base), egg_base_candidates=len(egg_base),
        distinct_hand_sources=len({r['source_sha256'] for r in hand_rows}),
        distinct_egg_sources=len({r['source_sha256'] for r in egg_rows}), distinct_union_sources=len(unique),
        hand_rejected=rejected, decisions=decisions, measurement_budget_per_strategy=budget,
        aliases=aliases, strategy_pool=[clean(row) for row in all_rows],
        strategy_selections=[clean(row) for row in selected],
        selected_canonical_ids=[row['id'] for row in unique if row['id'] in selected_ids],
        scope='Same exact local laws, derived-only guards and identical sharing policies for both strategies. Hand enumerates every native/unit0/unit1 assignment at at most six admitted PACK sites, then the bounded policy set. Native and all legacy controls/policies are common mandatory controls. Allocation probes precede the final scheduled compiler and measured byte gates; no global algebraic/scheduling optimum is claimed.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('source', type=Path)
    ap.add_argument('--timing-facts', type=Path, required=True)
    ap.add_argument('--atlas-opt', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--budget', type=int, default=4)
    ap.add_argument('--attempts', type=int, default=64)
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    import atlas_egglog_sharing, atlas_hand_reference
    pins = implementation_pins()
    for path in (Path(__file__).resolve(), Path(atlas_egglog_sharing.__file__).resolve(),
                 Path(atlas_hand_reference.__file__).resolve(), _compiler(args.atlas_opt)):
        pins[str(path)] = digest(path.read_bytes())
    rows, report = compare(load_program(args.source), TimingCosts(args.timing_facts),
                          atlas_opt=args.atlas_opt, budget=args.budget, attempts=args.attempts)
    if any(digest(Path(path).read_bytes()) != expected for path, expected in pins.items()):
        raise ValueError('provider identity changed during generation')
    proof = report['base_search']['proof'] | dict(implementation_pins=pins,
        qualification='Provider derivation declaration including exact structural sharing and real allocator fallback; final compile and measured full-output gates remain required.',
        candidate_bindings={r['id']:dict(source_sha256=r['source_sha256'], choices=r['choices'],
            sharing_policy=r['policy'], effective_policy=r['effective_policy'], applied_shares=r['applied_shares']) for r in rows})
    proof_path = args.output/'proof.json'
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    proof_ref = dict(path=str(proof_path.resolve()), sha256=digest(proof_path.read_bytes()), schema=proof['schema'])
    candidates = []
    for row in rows:
        path = args.output/(row['id']+'.mlir')
        path.write_text(row['source'])
        candidates.append(dict(id=row['id'], source=str(path.resolve()), source_sha256=row['source_sha256'],
            admitted=True, estimated_cost=row['score'], proof=proof_ref, families=row['families']))
    selected = set(report['selected_canonical_ids'])
    (args.output/'pool.json').write_text(json.dumps(dict(candidates=candidates), indent=2)+'\n')
    (args.output/'candidates.json').write_text(json.dumps(dict(candidates=[r for r in candidates if r['id'] in selected]), indent=2)+'\n')
    (args.output/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status='GENERATED_NOT_MEASURED', unique_sources=len(rows), shortlisted=len(selected), seconds=report['total_generation_seconds'])))


if __name__ == '__main__':
    main()
