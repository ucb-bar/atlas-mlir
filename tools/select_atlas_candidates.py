#!/usr/bin/env python3
"""Compile and measure an admitted, bounded implementation shortlist.

The JSON manifest supplies extraction/compilation/measurement argv adapters.
Adapters own target semantics and exact-output/guard checks. This driver never
relaxes a numerical contract or substitutes an additive cost for target timing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time


class CorrectnessDifference(RuntimeError):
    pass


def safe(path):
    path = Path(path)
    if any(word in part.lower() for part in path.parts
           for word in ('hammer', 'vlsi')):
        raise ValueError('Path outside the permitted domain')
    path = Path(os.path.abspath(path))
    # Reject aliases before entering a descendant; never resolve a link first.
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError('Source/output paths must not traverse symlinks')
    return path


def identifier(value):
    if (not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-'
                         for c in value)
            or any(word in value.lower() for word in ('hammer', 'vlsi'))):
        raise ValueError('Unsafe candidate or target ID')
    return value


def load(path):
    return json.loads(safe(path).read_text())


def save(path, value):
    safe(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def rank(rows, baseline, target_ids):
    """Cold-cycle winners and worst-case normalized regret on the given grid."""
    if not target_ids:
        return {'status': 'NO_TARGETS'}
    complete = {name: values for name, values in rows.items()
                if all(target in values for target in target_ids)}
    if baseline not in complete:
        return {'status': 'INCOMPLETE_BASELINE'}
    best = {target: min(values[target] for values in complete.values())
            for target in target_ids}
    winners = {target: min(complete, key=lambda name: (complete[name][target], name))
               for target in target_ids}
    metrics = {}
    for name, values in complete.items():
        dominators = [other for other, costs in complete.items()
                      if all(costs[t] <= values[t] for t in target_ids)
                      and any(costs[t] < values[t] for t in target_ids)]
        metrics[name] = {
            'worst_normalized_regret': max(values[t] / best[t] - 1 for t in target_ids),
            'worst_baseline_ratio': max(values[t] / complete[baseline][t] for t in target_ids),
            'dominated_by': sorted(dominators),
        }
    return {'status': 'PASS', 'metric': 'cold_first_issue_to_drained_ecall_inclusive',
            'winners': winners, 'metrics': metrics,
            'minimax_regret_candidate': min(metrics, key=lambda name: (
                metrics[name]['worst_normalized_regret'], name)),
            'within_five_percent_all_targets': sorted(name for name, value in metrics.items()
                if value['worst_normalized_regret'] <= .05),
            'limit': 'Only this measured shortlist and target grid; no global optimum or FPGA claim.'}


def check_measurement(receipt, program_sha):
    checks = receipt.get('correctness', {})
    if (receipt.get('status') == 'CORRECTNESS_DIFFERENCE'
            or any(checks.get(key) is False for key in ('outputs', 'oracle', 'guards'))):
        raise CorrectnessDifference(json.dumps(receipt))
    if receipt.get('status') != 'PASS':
        raise ValueError('Measurement adapter did not return PASS')
    if not all(checks.get(key) is True for key in ('outputs', 'oracle', 'guards')):
        raise ValueError('Measurement lacks explicit output/oracle/guard qualification')
    if receipt['program_sha256'] != program_sha:
        raise ValueError('Measured binary does not match the compiled candidate')
    cycles = receipt['cycles']
    if not cycles or any(type(value) is not int or value <= 0 for value in cycles):
        raise ValueError('Expected positive integer invocation cycle counts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest_path = safe(args.manifest)
    manifest = load(manifest_path)
    if manifest['schema'] != 'atlas.measured-selection.v1':
        raise ValueError('Unsupported manifest schema')
    if not manifest['targets'] or len({t['id'] for t in manifest['targets']}) != len(manifest['targets']):
        raise ValueError('Expected nonempty, uniquely named targets')
    if any(target['kind'] not in ('model', 'rtl') for target in manifest['targets']):
        raise ValueError('Target kind must be model or rtl')
    deadline_value = datetime.fromisoformat(manifest['deadline_utc'])
    if deadline_value.tzinfo is None:
        raise ValueError('Deadline must include its timezone')
    out = safe(args.output)
    out.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cache', 'compiled', 'measurements'):
        (out / name).mkdir()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
                       TMPDIR=str(out / 'tmp'), TMP=str(out / 'tmp'),
                       TEMP=str(out / 'tmp'), XDG_CACHE_HOME=str(out / 'cache'))
    commands = []
    result = {'schema': manifest['schema'], 'status': 'RUNNING',
              'manifest': str(manifest_path), 'manifest_sha256': sha(manifest_path),
              'driver_sha256': sha(__file__), 'candidates': {}, 'measurements': {}}
    deadline = deadline_value.timestamp()

    def poll():
        if time.time() >= deadline:
            raise TimeoutError('Declared deadline reached')
        for path in manifest.get('stop_files', []):
            if safe(path).exists():
                raise CorrectnessDifference('A coordinated correctness stop is active')

    def command(template, values, folder):
        poll()
        argv = [argument.format_map(values) for argument in template]
        started = time.perf_counter()
        process = subprocess.Popen(argv, cwd=folder, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            while True:
                poll()
                try:
                    stdout, stderr = process.communicate(timeout=min(1, max(.01, deadline-time.time())))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            except ProcessLookupError:
                stdout, stderr = process.communicate()
            (folder / 'stdout.log').write_text(stdout)
            (folder / 'stderr.log').write_text(stderr)
            raise
        (folder / 'stdout.log').write_text(stdout)
        (folder / 'stderr.log').write_text(stderr)
        commands.append({'argv': argv, 'cwd': str(folder), 'returncode': process.returncode,
                         'seconds': time.perf_counter() - started})
        save(out / 'commands.json', commands)
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)

    try:
        candidates = manifest.get('candidates', [])
        if manifest.get('extract_argv'):
            folder = out / 'extraction'
            folder.mkdir()
            process = command(manifest['extract_argv'], {'output': str(folder)}, folder)
            if process.returncode:
                raise ValueError('Candidate extraction failed; see extraction logs')
            candidates = load(folder / 'candidates.json')['candidates']
        if len({row['id'] for row in candidates}) != len(candidates):
            raise ValueError('Duplicate candidate IDs')
        admitted = []
        for candidate in candidates:
            if not candidate.get('admitted'):
                continue
            name = identifier(candidate['id'])
            if sha(candidate['source']) != candidate['source_sha256']:
                raise ValueError('Candidate source identity changed')
            proof = candidate['proof']
            if sha(proof['path']) != proof['sha256']:
                raise ValueError('Admission proof identity changed')
            admitted.append(candidate)
        baseline = manifest['baseline']
        control = next(row for row in admitted if row['id'] == baseline)
        budget = manifest['max_candidates']
        if type(budget) is not int or budget < 1:
            raise ValueError('Candidate budget must be a positive integer')
        selected = [control] + sorted((row for row in admitted if row['id'] != baseline),
                                     key=lambda row: (row['estimated_cost'], row['id']))[:budget - 1]
        save(out / 'shortlist.json', selected)
        costs = {}
        for candidate in selected:
            poll()
            name = candidate['id']
            folder = out / 'compiled' / name
            folder.mkdir()
            values = {'source': str(safe(candidate['source'])), 'output': str(folder),
                      'candidate_id': name}
            process = command(manifest['compile_argv'], values, folder)
            if process.returncode:
                result['candidates'][name] = {'status': 'COMPILE_REJECTED'}
                save(out / 'results.json', result)
                continue
            compiled = load(folder / 'receipt.json')
            if compiled['status'] != 'PASS' or compiled['source_sha256'] != candidate['source_sha256']:
                raise ValueError('Compilation receipt not bound to candidate source')
            result['candidates'][name] = compiled
            costs[name] = {}
            for target in manifest['targets']:
                poll()
                target_id = identifier(target['id'])
                folder = out / 'measurements' / name / target_id
                folder.mkdir(parents=True)
                save(folder / 'target.json', target)
                values = {'program': compiled['program'], 'output': str(folder),
                          'target': str(folder / 'target.json'), 'candidate_id': name}
                process = command(manifest['measure_argv'], values, folder)
                receipt_path = folder / 'receipt.json'
                if not receipt_path.exists():
                    result['measurements'][name + '/' + target_id] = {'status': 'ADAPTER_ERROR', 'returncode': process.returncode}
                    raise ValueError('Measurement returned no receipt; stop before another candidate or target')
                else:
                    measured = load(receipt_path)
                    result['measurements'][name + '/' + target_id] = measured
                    check_measurement(measured, compiled['program_sha256'])
                    if measured.get('target') != target:
                        raise ValueError('Measurement receipt target mismatch')
                    if process.returncode:
                        raise ValueError('Measurement adapter returned nonzero after PASS')
                    result['measurements'][name + '/' + target_id] = measured
                    costs[name][target_id] = measured['cycles'][0]
                save(out / 'results.json', result)
        targets = [target['id'] for target in manifest['targets']]
        result['ranking'] = rank(costs, baseline, targets)
        result['model_ranking'] = rank(costs, baseline, [t['id'] for t in manifest['targets'] if t['kind'] == 'model'])
        complete = all(len(values) == len(targets) for values in costs.values()) and len(costs) == len(selected)
        result['status'] = 'PASS' if complete else 'INCOMPLETE'
    except CorrectnessDifference as error:
        result.update(status='CORRECTNESS_DIFFERENCE', error=str(error))
        save(out / 'CORRECTNESS_STOP.json', result)
        raise
    except Exception as error:
        result.update(status='ERROR', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save(out / 'results.json', result)


if __name__ == '__main__':
    main()
