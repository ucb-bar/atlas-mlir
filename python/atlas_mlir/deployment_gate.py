"""Bind authenticated invocations and execute the existing model checker."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from .deploy import Gate, Refusal, check_hashes, digest, safe_path

# Completion deadline of one composed step, in simulation and on the bench. The slowest staged step measured in
# simulation takes under 30 s, so 300 s leaves a tenfold margin and bounds how long a hung step holds the bench.
STEP_TIMEOUT_SECONDS = 300


def _json(path):
    return json.loads(safe_path(path).read_text())


def _write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


def _member(root, name):
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise Refusal('unsafe artifact member')
    from atlas_mlir_native.reference_eval.source_io import checked_path
    return safe_path(checked_path(root / relative))


def _archive(path):
    import numpy as np
    with np.load(safe_path(path), allow_pickle=False) as archive:
        return {name: value.copy() for name, value in archive.items()}


def prepare_binding(context):
    """Reuse compiler binding with original raw inputs and a packaging clock."""
    from atlas_mlir_native.reference_eval.source_io import capture_inputs, capture_sources
    from merlin.compiler.build.handoff import bind_deployment_payloads

    out = context['out']
    application = out / 'compiled' / 'application'
    manifest = _json(application / 'manifest.json')
    artifacts = manifest.get('stage_artifacts', {})
    if not artifacts or 'deployment.json' not in artifacts:
        raise Refusal('verified application artifact inventory missing')
    check_hashes({str(_member(application, name)): expected for name, expected in artifacts.items()})
    deployment = _json(application / 'deployment.json')
    if deployment.get('schema') != 'merlin.onchip_deployment.v1' or deployment.get('paging'):
        raise Refusal('only complete nonpaged onchip deployment is qualified')
    capture = out / 'capture'
    capture_root, nodes, _ = capture_sources(capture, receipt_sha256=digest(capture / 'capture_receipt.json'))
    inputs = list(capture_inputs(capture_root, nodes).values())
    if sorted(row['index'] for row in deployment['inputs']) != list(range(len(inputs))):
        raise Refusal('source and deployment input indices differ')
    payloads = {}
    for row in deployment['inputs']:
        tensor = inputs[row['index']]
        if (tensor.dtype != row['type']['dtype'] or list(tensor.raw.shape) != row['type']['shape']
                or tensor.raw.nbytes != row['bytes']):
            raise Refusal('source input type or size differs from deployment')
        payloads[row['path']] = tensor.raw.tobytes(order='C')
    for row in deployment.get('state_inputs', []):
        if (row.get('role') != 'persistent_state_input'
                or row.get('state_mode') != 'read_only'
                or row.get('initialization') != 'captured'):
            raise Refusal('deployment binding supports only captured read-only state')
        name = row.get('path')
        if (not isinstance(name, str) or not name.startswith('handoff/state/')
                or name in payloads or artifacts.get(name) != row.get('sha256')):
            raise Refusal('captured state payload inventory differs from deployment')
        raw = _member(application, name).read_bytes()
        if (type(row.get('bytes')) is not int or len(raw) != row['bytes']
                or hashlib.sha256(raw).hexdigest() != row.get('sha256')):
            raise Refusal('captured read-only state bytes changed')
        payloads[name] = raw
    if deployment.get('parameters'):
        row = deployment['parameters']
        path = _member(application, row['path'])
        if len(path.read_bytes()) != row['bytes'] or digest(path) != row['sha256']:
            raise Refusal('compiled immutable parameter bytes changed')
        payloads[row['path']] = path.read_bytes()
    # The resident arena is the compiler's published image of its resident loads, linked as one read-only load.
    for row in [deployment['resident']] if deployment.get('resident') else []:
        path = _member(application, row['path'])
        if len(path.read_bytes()) != row['bytes'] or digest(path) != row['sha256']:
            raise Refusal('compiled resident arena bytes changed')
        payloads[row['path']] = path.read_bytes()
    # The compile published the deploy's clock; the kit records only that one.
    if deployment['provider'].get('sys_clk_freq_hz') != context['settings']['clock_hz']:
        raise Refusal('compiled host clock differs from the deploy clock')
    # Keep the verified publication unchanged; the arithmetic checker consumes that original deployment.
    platform = copy.deepcopy(deployment)
    platform['provider']['sys_clk_freq_hz'] = context['settings']['clock_hz']
    generated = bind_deployment_payloads(platform, payloads)
    return dict(deployment=deployment, platform_deployment=platform,
                payloads=payloads, generated=generated)


def _reference(context, deployment):
    root = context['reference']
    record = _json(root / 'reference.json')
    if (record.get('status') != 'EXPECTED'
            or record.get('ordinary_framework_kernels_called') is not False
            or record.get('compiler_arithmetic_or_emitted_program_inputs') is not False
            or digest(root / 'expected.npz') != record.get('expected_sha256')):
        raise Refusal('independent source reference authentication missing')
    if record.get('authentication', {}).get('receipt_sha256') != digest(context['out'] / 'capture/capture_receipt.json'):
        raise Refusal('reference capture identity differs')
    identities = record.get('compiler_decision_inputs', {}).get('identities', {})
    app_manifest = context['out'] / 'compiled/application/manifest.json'
    if identities.get(str(app_manifest)) != digest(app_manifest):
        raise Refusal('reference selected application identity differs')
    values = _archive(root / 'expected.npz')
    rows = deployment['outputs']
    if not rows or set(values) != {f"output_{row['index']}" for row in rows}:
        raise Refusal('reference must cover every deployment output exactly once')
    expected_hashes = []
    storage = {'bf16': 'uint16', 'f32': 'uint32', 'i64': 'int64', 'i32': 'int32', 'u8': 'uint8', 'bool': 'bool'}
    import hashlib
    import numpy as np
    nonzero = False
    for row in rows:
        value = values[f"output_{row['index']}"]
        declared = record.get('outputs', {}).get(f"output_{row['index']}", {})
        if (str(value.dtype) != storage.get(row['type']['dtype'])
                or list(value.shape) != row['type']['shape'] or value.nbytes != row['bytes']
                or declared.get('shape') != list(value.shape)
                or declared.get('dtype') != str(value.dtype)):
            raise Refusal('reference output type or size differs from deployment')
        raw = value.tobytes(order='C')
        nonzero |= bool(np.count_nonzero(value))
        expected_hashes.append(dict(name=row['id'], role=row['role'], bytes=len(raw),
                                    sha256=hashlib.sha256(raw).hexdigest()))
    rules = context['out'] / 'target-rules.json'
    if not rules.is_file() or digest(rules) != record.get('target_rules_sha256'):
        raise Refusal('reference target rules identity differs')
    if not nonzero:
        raise Refusal('all-zero independent reference refuses coverage')
    return record, expected_hashes


def bind(context):
    """Publish a checker bundle without modifying the verified application."""
    prepared = prepare_binding(context)
    deployment = prepared['deployment']
    record, expected_hashes = _reference(context, deployment)
    objects = context.get('objects', {})
    if not objects.get('measurements') or not objects.get('hashes'):
        raise Refusal('measured host objects required before binding')
    check_hashes(objects['hashes'])
    from .deployment_link import host_fit
    host_fit(objects, deployment, prepared['generated']['handoff/window.bin'])
    bundle = context['out'] / 'bundle'
    bundle.mkdir(exist_ok=False)
    application = context['out'] / 'compiled/application'
    manifest = _json(application / 'manifest.json')
    hashes = dict(objects['hashes'])
    # Copy only compiler-authenticated members; never traverse a source tree.
    for name, expected in {**manifest['stage_artifacts'], 'manifest.json': digest(application / 'manifest.json')}.items():
        source = _member(application, name)
        destination = _member(bundle / 'application', name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if digest(destination) != expected:
            raise Refusal('application changed while binding')
        hashes[str(source)] = expected
    bundle_members = list(manifest['stage_artifacts'])
    selected_members = ['application/' + name for name in bundle_members + ['manifest.json']]
    for name, data in prepared['generated'].items():
        selected_members.append(name)
        destination = _member(bundle, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        measured = objects.get('generated')
        if measured and (safe_path(measured) / Path(name).name).is_file():
            if (safe_path(measured) / Path(name).name).read_bytes() != data:
                raise Refusal('platform artifact differs from measured object inputs')
    selected_members.append('window.bin')
    (bundle / 'window.bin').write_bytes(prepared['generated']['handoff/window.bin'])
    for row in deployment['inputs'] + deployment.get('state_inputs', []):
        selected_members.append(row['path'])
        destination = _member(bundle, row['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(prepared['payloads'][row['path']])
    reference = bundle / 'reference'
    reference.mkdir()
    source_reference = context['reference']
    for name in ('reference.json', 'expected.npz'):
        source = source_reference / name
        hashes[str(source)] = digest(source)
        (reference / ('source-reference.json' if name == 'reference.json' else name)).write_bytes(source.read_bytes())
    from atlas_mlir_native import reference_eval
    evaluator_root = Path(reference_eval.__file__).parent
    members = {'expected.npz': digest(reference / 'expected.npz'),
               'source-reference.json': digest(reference / 'source-reference.json')}
    if not record.get('implementation_sha256'):
        raise Refusal('independent evaluator implementation identity missing')
    for name, expected in record['implementation_sha256'].items():
        source = _member(evaluator_root, name)
        if digest(source) != expected:
            raise Refusal('independent evaluator source changed')
        destination = _member(reference / 'evaluator', name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        members['evaluator/' + name] = expected
        hashes[str(source)] = expected
    _write(reference / 'reference.json', dict(record, evaluator_members=members))
    _write(bundle / 'binding.json', dict(settings=context['settings'],
        source_application_manifest_sha256=digest(application / 'manifest.json'),
        original_deployment_sha256=digest(application / 'deployment.json'),
        platform_deployment=prepared['platform_deployment'], objects_receipt_sha256=digest(objects['receipt'])))
    _write(bundle / 'manifest.json', dict(deployment=deployment,
        scope='unchanged authenticated source export fixture zero; no hardware claim',
        expected_hashes=expected_hashes, window_sha256=digest(bundle / 'window.bin'),
        binding_sha256=digest(bundle / 'binding.json')))
    selected_members += ['reference/reference.json', 'binding.json', 'manifest.json']
    selected_members += ['reference/' + name for name in members]
    for name in selected_members:
        path = _member(bundle, name)
        hashes[str(path)] = digest(path)
    for name in ('capture_receipt.json', 'frontend-trace.json', 'inputs.npz', 'weights.safetensors'):
        path = context['out'] / 'capture' / name
        hashes[str(path)] = digest(path)
    rules = context['out'] / 'target-rules.json'
    hashes[str(rules)] = digest(rules)
    policy = context.get('policy') or context['out'] / 'precision-policy.json'
    hashes[str(policy)] = digest(policy)
    context.update(bundle=bundle, deployment=bundle, bound_hashes=hashes)


def _case(bundle, output, label, record, deployment):
    if (record.get('case') != label or record.get('bind_status') != 0 or record.get('run_status') != 0
            or record.get('launch_count', 0) <= 0 or record.get('total_instructions', 0) <= 0
            or record.get('public_buffers_hashed_at_published_addresses') is not True
            or record.get('immutable_buffers_unchanged') is not True):
        raise Refusal('executed model case evidence missing')
    dma = record.get('dma', {})
    if (dma.get('bounds_checked_before_execution') is not True
            or any(dma.get('commands', {}).get(name, 0) <= 0 for name in ('load', 'store'))):
        raise Refusal('executed bounded DMA evidence missing')
    runtime = _member(output / label, record['runtime_path'])
    if digest(runtime) != record['runtime_sha256']:
        raise Refusal('model runtime evidence changed')
    if _json(output / label / 'result.json') != record:
        raise Refusal('case receipt differs from checker receipt')
    rows = {row['index']: row for row in deployment['hash_ranges']}
    if len(record['outputs']) != len(rows) or {row['index'] for row in record['outputs']} != set(rows):
        raise Refusal('executed case must check every public output')
    expected = _archive(bundle / 'reference/expected.npz')
    import hashlib
    mismatched = False
    checked_files = [runtime, output / label / 'result.json']
    for row in record['outputs']:
        declared = rows[row['index']]
        path = _member(output / label, row['path'])
        checked_files.append(path)
        raw = path.read_bytes()
        reference = expected[f"output_{row['index']}"].tobytes(order='C')
        differences = sum(a != b for a, b in zip(raw, reference, strict=True))
        mismatched |= bool(differences)
        if (row['name'] != declared['id'] or row['address'] != declared['address']
                or len(raw) != declared['bytes'] or row['bytes'] != len(raw)
                or digest(path) != row['sha256']
                or hashlib.sha256(reference).hexdigest() != row['expected_sha256']
                or row['mismatched_bytes'] != differences or row['exact'] != (differences == 0)):
            raise Refusal('executed output evidence differs from bound bytes')
    if (record.get('result') != ('MISMATCH' if mismatched else 'EXACT')
            or (label == 'positive' and mismatched) or (label == 'negative' and not mismatched)):
        raise Refusal('case result differs from executed output bytes')
    if mismatched:
        first = record.get('first_mismatch', {})
        declared = rows.get(first.get('output_index'))
        if declared is None:
            raise Refusal('mutation mismatch output is undeclared')
        actual_row = next(row for row in record['outputs'] if row['index'] == declared['index'])
        raw = _member(output / label, actual_row['path']).read_bytes()
        reference = expected[f"output_{declared['index']}"].tobytes(order='C')
        offset = first.get('byte_offset', -1)
        if (not 0 <= offset < len(raw) or raw[offset] == reference[offset]
                or first.get('address') != declared['address'] + offset
                or first.get('actual_byte') != raw[offset]
                or first.get('expected_byte') != reference[offset]):
            raise Refusal('mutation mismatch location differs from executed bytes')
    return checked_files


def _sign_offset(bundle):
    """Byte offset of the most significant byte of the first nonzero element of input zero, or None."""
    from math import prod
    deployment = _json(bundle / 'manifest.json')['deployment']
    row = next(row for row in deployment['inputs'] if row['index'] == 0)
    start = row['address'] - deployment['window']['address']
    data = (bundle / 'window.bin').read_bytes()[start:start + row['bytes']]
    size = row['bytes'] // max(1, prod(row['type']['shape']))
    first = next((i for i in range(0, len(data), size) if any(data[i:i + size])), None)
    return None if first is None else first + size - 1


def exact(context):
    """Run both cases and retain the checker PASS plus all bound identities."""
    checker_value = os.environ.get('ATLAS_MODEL_CHECKER')
    if not checker_value:
        raise Refusal('model checker path unavailable: set ATLAS_MODEL_CHECKER')
    checker = safe_path(checker_value)
    hashes = dict(context['bound_hashes'])
    hashes[str(checker)] = digest(checker)
    # The existing checker imports this runtime from its configured environment.
    runtime = importlib.util.find_spec('functional_runtime')
    if runtime is None or not runtime.origin:
        raise Refusal('qualified functional runtime unavailable')
    runtime_root = safe_path(runtime.origin).parent
    for name in ('functional_runtime.py', 'model_runtime.py', 'execute_graph.py', 'qualified_native_dot.py'):
        path = _member(runtime_root, name)
        hashes[str(path)] = digest(path)
    baseline_value = os.environ.get('W1_BASELINE')
    if not baseline_value:
        raise Refusal('qualified model baseline unavailable: set W1_BASELINE')
    for name in ('qualification.json', 'memory-qualification.json'):
        path = _member(safe_path(baseline_value) / 'out/qualification', name)
        hashes[str(path)] = digest(path)
    check_hashes(hashes)
    output = context['out'] / 'exact'
    receipt = context['out'] / 'check.json'
    if output.exists() or receipt.exists():
        raise Refusal('model checker receipt must be fresh')
    interpreter_command = os.environ.get('ATLAS_MODEL_CHECKER_PYTHON', sys.executable)
    interpreter = safe_path(Path(interpreter_command).resolve(strict=True))
    hashes[str(interpreter)] = digest(interpreter)
    environment = os.environ.copy()
    extra_imports = environment.get('ATLAS_MODEL_CHECKER_PYTHONPATH')
    if extra_imports:
        environment['PYTHONPATH'] = extra_imports + os.pathsep + environment.get('PYTHONPATH', '')
    # The checker's mutation flips bit 7 of the first byte of input zero with a nonzero low seven bits. When that
    # reaches no output (a low mantissa bit), the one retry flips the sign of the first nonzero element instead. The
    # checker writes no receipt when it refuses, so its log line is the only signal of that case.
    sign = _sign_offset(context['bundle'])
    for retry in (False, True):
        output.mkdir()
        # Preserve the virtual environment selected by the executable path.
        command = [str(interpreter_command), str(checker), '--bundle', str(context['bundle']),
                   '--out', str(output), '--receipt', str(receipt), *(['--flip-offset', str(sign)] if retry else [])]
        with (context['out'] / 'exact.log').open('w') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        check_hashes(hashes)
        if (retry or not completed.returncode or sign is None
                or 'negative byte flip did not influence a declared output' not in (context['out'] / 'exact.log').read_text()):
            break
        output.rename(context['out'] / 'exact-first-flip')
        (context['out'] / 'exact.log').rename(context['out'] / 'exact-first-flip.log')
    if completed.returncode:
        raise Refusal('bound model execution refused; see ' + str(context['out'] / 'exact.log'))
    record = _json(receipt)
    if record.get('checker_sha256') != hashes[str(checker)] or record.get('bundle_manifest_sha256') != digest(context['bundle'] / 'manifest.json'):
        raise Refusal('checker or bundle identity differs from execution receipt')
    deployment = _json(context['bundle'] / 'manifest.json')['deployment']
    checked_files = []
    for label in ('positive', 'negative'):
        checked_files += _case(context['bundle'], output, label, record[label], deployment)
    pin = 'b97b2f5a97e432141c4ef8504bec396651d77a8f'
    identity = record.get('model_checkout', {})
    if (record.get('model_revision') != pin or record.get('backend') != 'qualified_native'
            or identity.get('revision') != pin or identity.get('clean') is not True
            or identity.get('status_porcelain') != ''):
        raise Refusal('qualified pinned model identity missing')
    expected_images = [{key: row[key] for key in ('path', 'sha256', 'bytes')}
                       for row in deployment['provider']['images']]
    if record.get('instruction_images') != expected_images:
        raise Refusal('executed instruction image identities differ')
    build = record['build']
    if _json(output / 'host-build.json') != build:
        raise Refusal('host build receipt differs')
    handoff = context['bundle'] / 'application/handoff'
    for name, key in [('capsule_combined.c', 'generated_source_sha256'),
                      ('capsule_combined.h', 'generated_header_sha256'),
                      ('deployment_adapter.h', 'adapter_header_sha256')]:
        if digest(handoff / name) != build.get(key):
            raise Refusal('executed generated host source identity differs')
    if digest(output / 'model-shim.c') != build.get('shim_sha256'):
        raise Refusal('executed model shim identity differs')
    library = output / 'application.so'
    if digest(library) != build['library_sha256']:
        raise Refusal('executed application library changed')
    flip = record['byte_flip']
    original = (context['bundle'] / 'window.bin').read_bytes()
    offset = flip['window_byte_offset']
    input_row = next(row for row in deployment['inputs'] if row['index'] == 0)
    input_start = input_row['address'] - deployment['window']['address']
    if (flip.get('input_index') != 0 or flip.get('input_name') != input_row['id']
            or not input_start <= offset < input_start + input_row['bytes']
            or flip.get('input_byte_offset') != offset - input_start
            or flip.get('address') != deployment['window']['address'] + offset):
        raise Refusal('executed mutation is outside bound input zero')
    changed = bytearray(original)
    changed[offset] ^= flip['xor_mask']
    import hashlib
    if (flip['xor_mask'] != 128 or flip['before_byte'] != original[offset]
            or flip['after_byte'] != changed[offset]
            or flip['window_before_sha256'] != hashlib.sha256(original).hexdigest()
            or flip['window_after_sha256'] != hashlib.sha256(changed).hexdigest()
            or flip['input_before_sha256'] != hashlib.sha256(original[input_start:input_start + input_row['bytes']]).hexdigest()
            or flip['input_after_sha256'] != hashlib.sha256(changed[input_start:input_start + input_row['bytes']]).hexdigest()
            or _json(output / 'negative-byte-flip.json') != flip):
        raise Refusal('executed mutation identity differs from bound input')
    checked_files += [library, output / 'model-shim.c', output / 'host-build.json',
                      output / 'negative-byte-flip.json']
    for path in checked_files:
        hashes[str(path)] = digest(path)
    hashes[str(receipt)] = digest(receipt)
    gate = Gate(receipt=receipt, hashes=hashes)
    gate.verify()
    return gate


class ComposedGate(Gate):
    """Pre-link composed EXACT of a staged plan: every step executed, and the declared negative rejected."""

    def verify(self):
        check_hashes(self.hashes)
        if str(self.receipt) not in self.hashes:
            raise Refusal('composed EXACT receipt is not authenticated')
        record = json.loads(self.receipt.read_text())
        positive, negative = record.get('positive', {}), record.get('negative', {})
        if (record.get('schema') != 'atlas.composed_check.v1' or record.get('status') != 'PASS'
                or positive.get('result') != 'EXACT' or not positive.get('regions')
                or positive.get('completed') != positive.get('regions') or not positive.get('referenced_values')
                or negative.get('result') != 'MISMATCH' or not negative.get('errors')
                or record.get('bundle_inventory_sha256') != self.hashes.get(record.get('bundle_inventory'))):
            raise Refusal('bound composed EXACT and executed output-changing mutation required')


NEGATIVE_MASKS = (0x40, 0x80)


def _negative_candidates(data, width):
    """Input mutations to try, in a fixed order: the first, middle, and last nonzero elements, each with bit 6 and
    then bit 7 of its most significant byte flipped (for floats the top exponent bit, then the sign)."""
    nonzero = [i for i in range(0, len(data), width) if any(data[i:i + width])]
    if not nonzero:
        return []
    middle = (len(data) // width // 2) * width
    elements = []
    for element in (nonzero[0], next((i for i in nonzero if i >= middle), nonzero[-1]), nonzero[-1]):
        if element not in elements:
            elements.append(element)
    return [(element + width - 1, mask) for element in elements for mask in NEGATIVE_MASKS]


def _mutated_reference(context, inputs, destination):
    """The independent source reference of this deployment on supplied inputs; its expected outputs."""
    from .deployment_stages import _evaluate
    _evaluate(context['out'], destination, destination.with_name(destination.name + '.log'), inputs)
    return _archive(destination / 'expected.npz'), destination


def _choose_negative(context, sources, index, value, data, width, expected, hashes):
    """The first candidate input change whose independent reference changes an application output, recorded with
    every candidate tried; refuse if none does."""
    import hashlib
    import numpy as np
    storage = {'bf16': np.uint16, 'f32': np.float32}
    root = context['out'] / 'negative-reference'
    root.mkdir(exist_ok=False)
    tried = []
    for number, (offset, mask) in enumerate(_negative_candidates(data, width)):
        changed = bytearray(data)
        changed[offset] ^= mask
        arrays = {}
        for i, tensor in enumerate(sources):
            raw = np.ascontiguousarray(tensor.raw)
            if i == index:
                raw = np.frombuffer(bytes(changed), dtype=raw.dtype).reshape(raw.shape)
            arrays['input_%03d' % i] = raw.view(storage[tensor.dtype]) if tensor.dtype in storage else raw
        inputs = root / ('candidate-%d.npz' % number)
        np.savez(inputs, **arrays)
        row = dict(candidate=number, byte_offset=offset, xor_mask=mask, before=data[offset], after=data[offset] ^ mask)
        try:
            outputs, destination = _mutated_reference(context, inputs, root / ('candidate-%d' % number))
        except Refusal as error:
            tried.append(dict(row, result='reference refused: ' + str(error)))
            continue
        if set(outputs) != set(expected):
            raise Refusal('mutated reference outputs differ from the deployment outputs')
        differs = [name for name in sorted(expected)
                   if outputs[name].tobytes(order='C') != expected[name].tobytes(order='C')]
        if not differs:
            tried.append(dict(row, result='unchanged'))
            continue
        for path in (inputs, destination / 'reference.json', destination / 'expected.npz'):
            hashes[str(path)] = digest(path)
        reference = dict(inputs_sha256=digest(inputs), expected_sha256=digest(destination / 'expected.npz'),
                         changed_outputs=differs)
        return dict(value=value, byte_offset=offset, before=data[offset], after=data[offset] ^ mask, xor_mask=mask,
                    candidate=number, reference=reference, tried=tried + [dict(row, result='changed')])
    raise Refusal('no candidate input mutation changes an application output in the independent reference; tried '
                  + '; '.join('byte %(byte_offset)d xor 0x%(xor_mask)02x: %(result)s' % row for row in tried)
                  if tried else 'the first application input is all zero; no output-changing negative is declared')


def bind_staged(context):
    """Publish the composed plan of a staged application for its pre-link EXACT run.

    Values are whole tensors the bench host holds; steps read and write byte ranges of
    them in schedule order. Application inputs come from the authenticated capture,
    application outputs are referenced by the independent source reference, and the
    declared negative is the first candidate change of the first application input that the independent
    reference shows changes an application output.
    """
    import hashlib
    from math import prod
    import numpy as np
    from atlas_mlir_native.reference_eval.source_io import capture_inputs, capture_sources
    from .chiprunner import composed
    from .chiprunner.step_inputs import DTYPES
    from .deployment_link import RECORD_TAG

    out = context['out']
    application = out / 'compiled' / 'application'
    artifacts = _json(application / 'manifest.json').get('stage_artifacts', {})
    if 'deployment.json' not in artifacts or 'source-request.json' not in artifacts:
        raise Refusal('verified application artifact inventory missing')
    check_hashes({str(_member(application, name)): artifacts[name] for name in ('deployment.json', 'source-request.json')})
    plan = _json(application / 'deployment.json')
    if plan.get('schema') != 'merlin.onchip_deployment.v3' or plan.get('mode') != 'bench_staged':
        raise Refusal('only bench staged onchip plans are composed')
    request = _json(application / 'source-request.json')
    types = {node['id']: node['type'] for node in request['nodes']}
    size = lambda value: prod(types[value]['shape']) * np.dtype(DTYPES[types[value]['dtype']]).itemsize
    outputs = [row['value'] for row in plan['application_outputs']]
    if outputs != request['outputs']:
        raise Refusal('staged application outputs differ from the source outputs')
    rows = [dict(index=i, id=value, role='public_output', type=types[value], bytes=size(value))
            for i, value in enumerate(outputs)]
    record, _ = _reference(context, dict(outputs=rows))
    expected = _archive(context['reference'] / 'expected.npz')
    capture = out / 'capture'
    root, nodes, _ = capture_sources(capture, receipt_sha256=digest(capture / 'capture_receipt.json'))
    captured = capture_inputs(root, nodes)
    sources, source_ids = list(captured.values()), list(captured)

    bundle = out / 'bundle'
    bundle.mkdir(exist_ok=False)
    hashes = dict(context.get('objects', {}).get('hashes', {}))
    members = []

    def write(name, data):
        path = _member(bundle, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
        members.append(name)
        return name

    def member(name):
        # Only compiler-authenticated members; each is copied once.
        if 'application/' + name in members:
            return 'application/' + name
        if name not in artifacts:
            raise Refusal('staged member is not an authenticated application artifact: ' + name)
        data = _member(application, name).read_bytes()
        if hashlib.sha256(data).hexdigest() != artifacts[name]:
            raise Refusal('application changed while binding')
        return write('application/' + name, data)

    def span(source):
        return {key: source[key] for key in ('offset', 'bytes', 'count', 'stride')}

    values, first = {}, None
    for step in plan['steps']:
        for load in step['loads']:
            source = load['source']
            if source['kind'] == 'parameter':
                continue
            if source['kind'] not in ('application_input', 'step_output'):
                raise Refusal('unsupported staged load source: ' + str(source['kind']))
            row = values.setdefault(source['value'], dict(id=source['value'], bytes=size(source['value'])))
            if source['kind'] == 'application_input' and 'file' not in row:
                if not 0 <= source['index'] < len(sources):
                    raise Refusal('staged application input index is outside the source inputs')
                tensor = sources[source['index']]
                if (tensor.dtype != types[row['id']]['dtype'] or list(tensor.raw.shape) != types[row['id']]['shape']
                        or tensor.raw.nbytes != row['bytes']):
                    raise Refusal('source input type or size differs from the staged value: ' + row['id'])
                row['file'] = write('inputs/%s.bin' % row['id'], tensor.raw.tobytes(order='C'))
                if first is None:
                    first, first_index = row, source['index']
        for read in step['reads']:
            values.setdefault(read['destination']['value'], dict(id=read['destination']['value'],
                                                                 bytes=size(read['destination']['value'])))
    loaded = [(load['source']['index'], load['source']['value']) for step in plan['steps'] for load in step['loads']
              if load['source']['kind'] == 'application_input']
    if first is None or not {index for index, _ in loaded} <= set(range(len(sources))):
        raise Refusal('source and staged application input indices differ')
    if {index for index, _ in loaded} != set(range(len(sources))):
        _unconsumed_inputs(nodes, source_ids, request, loaded)
    for index, value in enumerate(outputs):
        if value not in values:
            raise Refusal('staged application output is never written: ' + value)
        if not np.any(expected[f'output_{index}']):
            raise Refusal('all-zero application output reference refuses coverage: ' + value)
        values[value]['reference'] = write('reference/%s.bin' % value, expected[f'output_{index}'].tobytes(order='C'))
    # Only application outputs are referenced, so the negative must reach an output. A trained model can absorb a
    # fixed mutation (a saturated corner pixel), so the independent reference decides: candidates in a fixed order
    # (see _negative_candidates) are evaluated on the source inputs and the first that changes an application
    # output is the one input change executed. Temporary rule; replacement: referenced step boundaries, where any
    # one-bit flip is seen at its first consumer.
    data = bytearray(_member(bundle, first['file']).read_bytes())
    width = np.dtype(DTYPES[types[first['id']]['dtype']]).itemsize
    mutation = _choose_negative(context, sources, first_index, first['id'], bytes(data), width, expected, hashes)
    offset = mutation['byte_offset']
    data[offset] ^= mutation['xor_mask']
    negative = dict(value=first['id'], file=write('negative/%s.bin' % first['id'], bytes(data)))

    regions, blobs = [], {}
    for step in plan['steps']:
        package = step['package']
        root = 'steps/%s/' % package
        deployment = _json(_member(bundle, member(root + 'deployment.json')))
        abi = _json(_member(application, root + 'handoff/abi.json'))
        region = dict(id='step_%03d_%s' % (step['step'], package), kind='application', window=deployment['window'],
                      deployment='application/' + root + 'deployment.json', launches=step['launches'],
                      images=[dict(program=member(row['path']),
                                   plan=member(str(Path(row['path']).parent / 'execution_plan.json')))
                              for row in deployment['provider']['images']],
                      sources=[member(root + 'handoff/' + name)
                               for name in (abi['source'], abi['header'], 'deployment_adapter.h')],
                      inputs=[], weights=[], outputs=[])
        for load in step['loads']:
            source, row = load['source'], dict(id=load['port'], address=load['address'], bytes=load['bytes'])
            if source['kind'] == 'parameter':
                if source['path'] not in blobs:
                    blobs[source['path']] = _member(bundle, member(source['path'])).read_bytes()
                if hashlib.sha256(composed.gather(blobs[source['path']], span(source), load['bytes'])).hexdigest() != source['sha256']:
                    raise Refusal('staged parameter range differs from its digest: ' + load['port'])
                region['weights'].append(dict(row, file=member(source['path']), range=span(source)))
            else:
                region['inputs'].append(dict(row, source=dict(value=source['value'], range=span(source))))
        for read in step['reads']:
            region['outputs'].append(dict(id=read['port'], address=read['address'], bytes=read['bytes'],
                                          value=dict(id=read['destination']['value'], range=span(read['destination']))))
        regions.append(region)
    deployment_outputs = []
    for index, item in enumerate(plan['application_outputs']):
        producer = regions[item['producer_steps'][-1]]
        match = [row for row in producer['outputs'] if row['value']['id'] == item['value']]
        if len(match) != 1 or match[0]['bytes'] != values[item['value']]['bytes']:
            raise Refusal('application output is not one whole step output: ' + item['value'])
        deployment_outputs.append(dict(index=index, id=item['value'], address=match[0]['address'],
                                       bytes=match[0]['bytes'], source=dict(region=producer['id'], output=match[0]['id'])))
    manifest = dict(runner=dict(schema='atlas.kit.v1', record_tag=RECORD_TAG),
                    composed=dict(values=list(values.values()), regions=regions, negative=negative,
                                  timeout_seconds=STEP_TIMEOUT_SECONDS),
                    deployment=dict(outputs=deployment_outputs),
                    staged=dict(application_manifest_sha256=digest(application / 'manifest.json'),
                                reference_sha256=digest(context['reference'] / 'reference.json'), negative=mutation))
    write('manifest.json', (json.dumps(manifest, indent=1) + '\n').encode())
    (bundle / 'SHA256SUMS').write_text(''.join(hashes[str(bundle / name)] + '  ' + name + '\n' for name in sorted(members)))
    hashes[str(bundle / 'SHA256SUMS')] = digest(bundle / 'SHA256SUMS')
    try:
        composed.load_plan(bundle)
    except (ValueError, KeyError) as error:
        raise Refusal('composed plan refused: ' + str(error)) from error
    for name in ('reference.json', 'expected.npz'):
        hashes[str(context['reference'] / name)] = digest(context['reference'] / name)
    for name in ('capture_receipt.json', 'frontend-trace.json', 'inputs.npz', 'weights.safetensors'):
        hashes[str(capture / name)] = digest(capture / name)
    rules = out / 'target-rules.json'
    policy = context.get('policy') or out / 'precision-policy.json'
    hashes.update({str(rules): digest(rules), str(policy): digest(policy)})
    context.update(bundle=bundle, bound_hashes=hashes)


def _unconsumed_inputs(nodes, source_ids, request, loaded):
    """A source input the staged plan never loads must be unconsumed in both the captured graph and the compiler's request.

    `loaded` lists (source input index, staged value) for every staged load. Captured placeholders pair
    in order with the request's input order, and every load must name its index's own request argument.
    An unloaded input then has no consumer in either graph, so no output depends on it.
    """
    def strings(item, found):
        # Dict keys count as well as values, since a section such as input_storages is keyed by value id.
        if isinstance(item, dict):
            for key, value in item.items():
                strings(key, found)
                strings(value, found)
        elif isinstance(item, (list, tuple)):
            for value in item:
                strings(value, found)
        elif isinstance(item, str):
            found.add(item)
        return found

    differ = 'source and staged application input indices differ'
    # A non-str or repeated id cannot be matched against uses, so the rule refuses rather than guess.
    ids = [node.get('id') for node in nodes]
    request_ids = [node.get('id') for node in request['nodes']]
    if (any(not isinstance(node_id, str) for node_id in ids + request_ids)
            or len(set(ids)) != len(ids) or len(set(request_ids)) != len(request_ids)):
        raise Refusal(differ)
    placeholders = [node['id'] for node in nodes if node.get('op') == 'placeholder']
    order = request.get('input_order', [])
    if len(placeholders) != len(order) or not set(source_ids) <= set(placeholders):
        raise Refusal(differ)
    argument = dict(zip(placeholders, order))
    if any(argument[source_ids[index]] != value for index, value in loaded):
        raise Refusal(differ)
    # Any string outside a node's own id may name a value, so every other field of every node counts as a use,
    # and so does every top-level request field except the input order and the inputs' own storages.
    request_inputs = {node['id'] for node in request['nodes'] if node.get('op') == 'input'}
    request_uses = strings([{key: value for key, value in node.items() if key != 'id'} for node in request['nodes']],
                           strings({key: value for key, value in request.items()
                                    if key not in ('nodes', 'input_order', 'input_storages')}, set()))
    by_id = {node['id']: node for node in nodes}
    used = {index for index, _ in loaded}
    for index, node_id in enumerate(source_ids):
        if index in used:
            continue
        name = argument[node_id]
        # A reference names the node, one of its result value ids, or a node_id-prefixed value.
        results = by_id[node_id].get('results') or []
        if (not isinstance(name, str) or not isinstance(results, list)
                or any(not isinstance(result, dict) or not isinstance(result.get('id'), str) for result in results)):
            raise Refusal(differ)
        names = {node_id} | {result['id'] for result in results}
        # Every field of every other captured node, keys included, counts as a use.
        captured_uses = strings([{key: value for key, value in node.items() if key != 'id'}
                                 for node in nodes if node['id'] != node_id], set())
        if (name not in request_inputs or name in request_uses
                or any(use in names or use.startswith(node_id + ':') for use in captured_uses)):
            raise Refusal(differ)

def _qualify(bundle, logs, *, backend='sim', transport_factory=None):
    """The runner's step_inputs.qualify verdicts, with the positive and negative simulated at once.

    Each call is its own session with its own journal directory over the read-only bundle,
    so running them together changes only when they finish. The positive's failure is
    reported first, as in the sequential runner.
    """
    from concurrent.futures import ThreadPoolExecutor
    from .chiprunner import composed

    logs = safe_path(logs)

    def negative():
        try:
            composed.execute(bundle, logs / 'negative', backend=backend, mutation=True,
                             transport_factory=transport_factory)
        except composed.BoundaryError:
            return 'expected numerical failure'
        raise ValueError('input mutation did not change an authenticated output')
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(negative)
        positive = composed.execute(bundle, logs / 'positive', backend=backend,
                                    transport_factory=transport_factory)
        return {'positive': positive, 'negative': pending.result()}


def exact_staged(context):
    """Run the composed plan and its declared negative in simulation, then retain both journals."""
    from .chiprunner import composed
    from .deployment_kit import owned_tools

    out, bundle = context['out'], context['bundle']
    hashes = dict(context['bound_hashes'])
    hashes.update({str(path): digest(path) for path in owned_tools().values()})
    check_hashes(hashes)
    receipt = out / 'composed-check.json'
    if receipt.exists():
        raise Refusal('composed EXACT receipt must be fresh')
    logs = out / 'exact'
    try:
        _qualify(bundle, logs)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        raise Refusal('composed EXACT refused: ' + str(error) + '; see ' + str(logs)) from error
    check_hashes(hashes)
    manifest = composed.load_plan(bundle)[0]
    plan = manifest['composed']
    positive, negative = (_json(logs / name / 'journal.json') for name in ('positive', 'negative'))
    rejected = negative.get('rejected', {})
    if (len(positive['completed']) != len(plan['regions']) or positive.get('active') is not None
            or positive['identity']['values'].get(plan['negative']['value']) == plan['negative']['file']
            or negative['identity']['values'].get(plan['negative']['value']) != plan['negative']['file']
            or not rejected.get('errors') or any(not error.startswith('boundary mismatch: ')
                                                 for error in rejected['errors'])):
        raise Refusal('composed journals lack a complete positive and a numerically rejected negative')
    for name in ('positive', 'negative'):
        hashes[str(logs / name / 'journal.json')] = digest(logs / name / 'journal.json')
    identity = {key: value for key, value in positive['identity'].items() if key not in ('inventory', 'values')}
    _write(receipt, dict(
        schema='atlas.composed_check.v1', status='PASS', bundle_inventory=str(bundle / 'SHA256SUMS'),
        bundle_inventory_sha256=digest(bundle / 'SHA256SUMS'), runner=identity,
        positive=dict(result='EXACT', regions=len(plan['regions']), completed=len(positive['completed']),
                      referenced_values=sum('reference' in row for row in plan['values']),
                      journal_sha256=digest(logs / 'positive' / 'journal.json')),
        negative=dict(result='MISMATCH', value=plan['negative']['value'], region=rejected['id'],
                      errors=rejected['errors'], journal_sha256=digest(logs / 'negative' / 'journal.json')),
        scope='simulation: each step application natively with every launch in the pinned functional model '
              '(numerics="rtl"); no linked ELF, RTL, or silicon execution'))
    hashes[str(receipt)] = digest(receipt)
    gate = ComposedGate(receipt=receipt, hashes=hashes)
    gate.verify()
    return gate
