#!/usr/bin/env python3
"""Replay measured-pair composition and explicit structural capacity; emit no admissions."""
from __future__ import annotations

from atlas_model_diagnostics import model_timing_cli

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from atlas_model_pairs import MeasuredPairs, RecordReader, canonical, issue_ages, safe_path


def engine_class(operations):
    return '+'.join(sorted({op.split('.')[0] for op in operations}))


def negative_vectors(record):
    """Keep every explicit negative descriptor, including nested model witnesses."""
    gaps = record['gaps']
    seen_paths = set()

    def walk(value, path, context):
        if isinstance(value, list):
            for i, child in enumerate(value):
                yield from walk(child, f'{path}/{i}', context)
        elif isinstance(value, dict):
            if 'operations' in value and 'all_pair_relations' in value:
                same = all(value.get(key) == context.get(key) for key in ('operations', 'gaps', 'all_pair_relations'))
                context = dict(context, **value) if same else value
            result = value.get('result', value.get('classification'))
            if result is None and 'negative_gaps' in value and value.get('error'):
                result = 'model_exception'
            if result in ('conflict', 'inexpressible', 'model_exception', 'rejected', 'error'):
                vector = value.get('negative_gaps', value.get('gaps'))
                if vector is None and 'issue_ages' in value:
                    ages = value['issue_ages']
                    vector = [b-a for a, b in zip(ages, ages[1:])]
                if vector is None and type(value.get('gap')) is int:
                    vector = list(context['gaps'])
                    axis = context.get('swept_gap_index')
                    if axis is not None:
                        vector[axis] = value['gap']
                    elif value['gap'] == 0 and result == 'inexpressible':
                        # Zero spacing is unencodable regardless of the unspecified axis.
                        vector[-1] = 0
                    elif 'swept' in context and 'min_gap' in context and len(set(vector)) == 1:
                        vector = [value['gap']] * len(vector)
                    else:
                        vector = None
                if vector is not None and path not in seen_paths:
                    seen_paths.add(path)
                    yield dict(path=path, gaps=vector, operations=context['operations'],
                               all_pair_relations=context['all_pair_relations'], expected=result)
                elif path not in seen_paths:
                    yield dict(path=path, unresolved=True, expected=result)
            for key, child in value.items():
                if key not in ('error', 'witness_edges', 'full_membership_trace', 'singletons'):
                    yield from walk(child, path+'/'+key, context)

    yield from walk(record, '', record)
    if record.get('earliest_pair_bounds_result') in ('conflict', 'model_exception', 'rejected'):
        yield dict(path='/earliest_pair_bounds_result', gaps=record['initial_gaps'], operations=record['operations'],
                   all_pair_relations=record['all_pair_relations'], expected=record['earliest_pair_bounds_result'])
    for i, observation in enumerate(record.get('gap_axis_observations', [])):
        if observation.get('observation') in ('specific_one_short_rejected', 'zero_gap_inexpressible'):
            vector = list(gaps)
            vector[observation['gap_index']] = observation['accepted_gap']-1
            yield dict(path=f'/gap_axis_observations/{i}', gaps=vector, operations=record['operations'],
                       all_pair_relations=record['all_pair_relations'], expected=observation['observation'])
    for i, sweep in enumerate(record.get('first_axis_full_live_sweep', [])):
        lo, hi = sweep['swept_first_gap']
        accepted = set(sweep['accepted_first_gaps'])
        rejected = [gap for gap in range(lo, hi+1) if gap not in accepted]
        if sweep.get('classifications', {}).get('model_exception') == len(rejected):
            for gap in rejected:
                vector = list(gaps)
                vector[0] = gap
                yield dict(path=f'/first_axis_full_live_sweep/{i}/gap/{gap}', gaps=vector,
                           operations=record['operations'], all_pair_relations=record['all_pair_relations'],
                           expected='model_exception')


def compose(record, gaps, rules, pairs):
    names = record['operations']
    if len(gaps) != len(names)-1 or any(type(gap) is not int or gap < 1 for gap in gaps):
        return dict(admitted=False, reason='inexpressible_gap')
    issues = issue_ages(gaps)
    relations = {(entry['first_index'], entry['second_index']): entry['relation']
                 for entry in record['all_pair_relations']}
    expected = {(i, j) for i in range(len(names)) for j in range(i+1, len(names))}
    if set(relations) != expected or len(record['all_pair_relations']) != len(expected):
        return dict(admitted=False, reason='incomplete_or_duplicate_relations')
    missing, rejected = [], []
    for i, first in enumerate(names):
        for j in range(i+1, len(names)):
            gap = issues[j]-issues[i]
            if gap >= rules.first_free_age(first):
                continue
            relation = relations.get((i, j))
            if relation is None:
                missing.append([i, j])
            else:
                error = None
                try:
                    admitted = pairs.admits_pair(first, names[j], relation, gap)
                except ValueError as exc:
                    admitted, error = False, str(exc)
                if not admitted:
                    rejected.append(dict(first_index=i, second_index=j, first=first, second=names[j],
                                         gap=gap, relation=relation, unsupported=error))
    capacity = rules.capacity_conflict(names, issues)
    peak = max(sum(start <= now < start + rules.first_free_age(name)-1
                   for name, start in zip(names, issues)) for now in issues)
    return dict(admitted=not (missing or rejected or capacity), peak_post_progress=peak,
                capacity_conflict=capacity, missing_relations=missing, rejected_pairs=rejected)


def compact_master(master):
    pairs, chains = [], []
    with master.open('rb') as stream:
        reader = RecordReader(stream)
        for section, index, record in reader.records():
            if section == 'pairs':
                kept = {key: record[key] for key in ('first', 'second', 'relation', 'min_gap', 'swept')}
                kept.update(confirmed=record.get('verilator', {}).get('confirmed') is True, source_index=index)
                pairs.append(kept)
            elif section == 'chains':
                # Negative extraction precedes trimming: nested records can have their own relations.
                negatives = list(negative_vectors(record))
                kept = {key: record[key] for key in ('operations', 'gaps', 'all_pair_relations', 'in_flight_count')}
                kept.update(source_index=index, confirmed=record.get('verilator', {}).get('confirmed') is True,
                            name=record.get('name', record.get('case_name')), negatives=negatives)
                chains.append(kept)
    return dict(pairs=pairs, chains=chains, master_sha256=reader.digest.hexdigest())


def replay(data, per_op, output, per_op_sha256):
    from atlas_model_composition import RuleSet, rule_source_sha256
    rules = RuleSet(per_op)
    modes = {'measured_confirmed': MeasuredPairs(data['pairs']),
             'measured_all': MeasuredPairs(data['pairs'], False), 'derived': rules}
    totals = defaultdict(Counter)
    disagreements, results = [], []
    for record in data['chains']:
        tier = 'confirmed' if record['confirmed'] else 'model_only'
        engine = engine_class(record['operations'])
        negatives = record.get('negatives')
        if negatives is None:
            negatives = list(negative_vectors(record))
        for mode, pairs in modes.items():
            counter = totals[mode+'/'+tier+'/'+engine]
            verdict = compose(record, record['gaps'], rules, pairs)
            counter['positives'] += 1
            counter['accepted_positives' if verdict['admitted'] else 'rejected_positives'] += 1
            entry = dict(source_index=record['source_index'], name=record.get('name'), mode=mode, tier=tier,
                         engine=engine, operations=record['operations'], gaps=record['gaps'], verdict=verdict)
            if not verdict['admitted']:
                disagreements.append(dict(entry, expected='accepted'))
            if mode == 'measured_confirmed':
                results.append(entry)
            unique_negatives = set()
            for negative in negatives:
                counter['negative_descriptors'] += 1
                if negative.get('unresolved'):
                    counter['unresolved_negative_descriptors'] += 1
                    disagreements.append(dict(source_index=record['source_index'], mode=mode, tier=tier,
                                              engine=engine, negative=negative, expected='rejected', unresolved=True))
                    continue
                verdict = compose(negative, negative['gaps'], rules, pairs)
                counter['accepted_negatives' if verdict['admitted'] else 'rejected_negatives'] += 1
                point = canonical({key: negative[key] for key in ('operations', 'gaps', 'all_pair_relations')})
                if point not in unique_negatives:
                    unique_negatives.add(point)
                    counter['unique_accepted_negatives' if verdict['admitted'] else 'unique_rejected_negatives'] += 1
                if verdict['admitted']:
                    disagreements.append(dict(source_index=record['source_index'], mode=mode, tier=tier,
                                              engine=engine, negative=negative, verdict=verdict, expected='rejected'))
    summary = dict(schema='atlas.chain_composition_replay.v1', master_sha256=data['master_sha256'],
                   rule_source_sha256=rule_source_sha256(),
                   per_op_sha256=per_op_sha256, chain_count=len(data['chains']),
                   counts={key: dict(value) for key, value in sorted(totals.items())},
                   scope='Diagnostic only; measured_all and derived do not establish admission authority.')
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary', summary), ('disagreements', disagreements), ('confirmed-pair-results', results)]:
        (output/(name+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    return summary


@model_timing_cli
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--per-op', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.master and not args.cache:
        parser.error('--master or --cache is required')
    data = compact_master(safe_path(args.master)) if args.master else json.loads(safe_path(args.cache).read_text())
    raw = safe_path(args.per_op).read_bytes()
    per_op = json.loads(raw)
    summary = replay(data, per_op, safe_path(args.output), hashlib.sha256(raw).hexdigest())
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
