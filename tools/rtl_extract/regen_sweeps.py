"""Sweep one recipe with ordered pair shards and a bounded total worker budget."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path

from . import facts, pairs, sequences
from .ir import load_document
from .regen_cache import safe_path, sha256
from .runner import modules

CONTEXT = None
def partition(names, count):
    if count < 1:
        raise ValueError('shards must be positive')
    names = list(names)
    count = min(count, len(names))
    return [names[i * len(names) // count:(i + 1) * len(names) // count] for i in range(count)]

def merge_records(names, shards):
    records = {}
    for shard in shards:
        for record in shard:
            name = record['name']
            if name in records:
                raise ValueError(f'duplicate shard record: {name}')
            records[name] = record
    missing, extra = set(names) - records.keys(), records.keys() - set(names)
    if missing or extra:
        raise ValueError(f'shard records differ: missing {sorted(missing)}, extra {sorted(extra)}')
    return [records[name] for name in names]

def extract_shard(names):
    document, raw, spec, gap_jobs = CONTEXT
    return pairs.extract(document, {**raw, 'pairs': {name: raw['pairs'][name] for name in names}}, spec, gap_jobs)

def extract_sharded(document, raw, spec, jobs, shards):
    global CONTEXT
    names = list(raw['pairs'])
    chunks = partition(names, shards)
    workers = min(len(chunks), jobs)
    if workers < 1:
        raise ValueError('jobs must be positive and pair spec must be nonempty')
    if workers == 1:
        return pairs.extract(document, raw, spec, jobs)
    CONTEXT = document, raw, spec, max(1, jobs // workers)
    results = []
    try:
        # Non-daemon fork workers permit the existing nested gap pool.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
            futures = {pool.submit(extract_shard, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                result = future.result(); results.append(result)
                print(f'shard {futures[future] + 1}/{len(chunks)}: {len(result)} records', flush=True)
    finally:
        CONTEXT = None
    return merge_records([f"{spec['engine']}.{name}" for name in names], results)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('hw-ir', 'exporter', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    parser.add_argument('--kind', choices=('pairs', 'sequences'), required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    if args.shards < 1 or args.kind == 'sequences' and args.shards != 1:
        parser.error('positive shards required; sequences only support one shard')
    for path in (args.hw_ir, args.exporter, args.output): safe_path(path)
    recipe = pairs if args.kind == 'pairs' else sequences
    raw, spec = recipe.load([args.name])[args.name]
    document = load_document(args.hw_ir, modules(spec), args.exporter, pairs.needs_assertions(raw, spec))
    records = (extract_sharded(document, raw, spec, args.jobs, args.shards)
               if args.kind == 'pairs' and args.shards > 1 else recipe.extract(document, raw, spec, args.jobs))
    facts.write(args.output, {'schema': recipe.SCHEMA,
                            'hw_ir': {'path': str(args.hw_ir.resolve()), 'sha256': sha256(args.hw_ir)},
                            'op_' + args.kind: records})
    print(f'{args.kind}/{args.name}: {len(records)} records -> {args.output}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
