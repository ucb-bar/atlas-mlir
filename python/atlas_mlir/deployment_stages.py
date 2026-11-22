"""Adapters for authenticated deployment stages, without transport access."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .deploy import Refusal, digest, safe_path


def _host_math(out):
    """Build the x86 evaluator of the supplier that host regions embed, for the reference."""
    from merlin.runtime import host_math_supplier as host_math

    root = Path(host_math.__file__).resolve().parent
    with (out / 'host-math.log').open('w') as log:
        result = subprocess.run([sys.executable, str(root / 'build.py'), '--target', 'x86',
                                 '--out', str(out / 'host-math')], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise Refusal('host-math evaluator build failed; see ' + str(out / 'host-math.log'))
    return dict(binding_path=str(root / '__init__.py'), build_dir=str(out / 'host-math' / 'x86'))


def _evaluate(out, destination, log, inputs=None):
    """Run the source evaluator on the deployment's capture and selected placement; refuse unless EXPECTED and nonzero."""
    capture = out / 'capture'
    application = out / 'compiled' / 'application'
    command = [sys.executable, '-m', 'atlas_mlir_native.reference_eval', str(capture),
               '--capture', '--receipt-sha256', digest(capture / 'capture_receipt.json'),
               '--selected-application', str(application),
               '--application-manifest-sha256', digest(application / 'manifest.json'),
               '--target-rules', str(out / 'target-rules.json'), '--out', str(destination)]
    if inputs is not None:
        command += ['--inputs', str(inputs)]
    with log.open('w') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        receipt = destination / 'reference.json'
        record = json.loads(receipt.read_text()) if receipt.is_file() else {}
        raise Refusal('independent reference refused: ' + str(record.get('reason', 'evaluator failed'))
                      + '; see ' + str(log))
    record = json.loads((destination / 'reference.json').read_text())
    if (record.get('status') != 'EXPECTED' or record.get('ordinary_framework_kernels_called') is not False
            or record.get('compiler_arithmetic_or_emitted_program_inputs') is not False
            or not any(row.get('nonzero', 0) for row in record.get('outputs', {}).values())
            or digest(destination / 'expected.npz') != record.get('expected_sha256')):
        raise Refusal('independent reference identity or nonzero coverage missing')


def reference(context):
    """Execute the source evaluator using authenticated placement metadata only."""
    out = context['out']
    selected = dict(npu_model_revision='b97b2f5a97e432141c4ef8504bec396651d77a8f',
                    numerics='rtl', k_order='ascending-k32', host_order='lexicographic-f32')
    regions = json.loads((out / 'compiled' / 'application' / 'compiled-regions.json').read_text())
    if any(region.get('description', {}).get('host_math_symbols') for region in regions):
        selected['host_math'] = _host_math(out)
    (out / 'target-rules.json').write_text(json.dumps(selected, indent=2) + '\n')
    _evaluate(out, out / 'reference', out / 'reference.log')
    context['reference'] = out / 'reference'


def new_inputs(out, inputs, destination):
    """`atlasc reference`: evaluate a packaged deployment's source on new inputs, exactly as deploy evaluated its own.

    The result is the reference `atlasc run OUT --inputs FILE --reference DIR` accepts for these inputs.
    """
    from atlas_mlir_native.reference_eval.source_io import capture_inputs, capture_sources
    from .chiprunner import step_inputs
    from .deployment_run import _supplied_reference, kit_tensors
    out, inputs, destination = safe_path(out), safe_path(inputs), safe_path(destination)
    if json.loads((out / 'deploy-status.json').read_text()).get('status') != 'PACKAGED':
        raise Refusal('deployment incomplete; a packaged deployment is required')
    if destination.exists() or destination.is_relative_to(out):
        raise Refusal('reference output must be fresh and outside OUT')
    # input_NNN names the source's NNN-th user input here and the deployment's input NNN in atlasc run; the packaged
    # inputs prove both orders agree for this deployment. A single-ELF kit and a composed or staged plan alike.
    try:
        deployment, packaged, _ = kit_tensors(out / 'kit')
    except (OSError, ValueError, KeyError) as error:
        raise Refusal(str(error)) from error
    capture = out / 'capture'
    _, nodes, _ = capture_sources(capture, receipt_sha256=digest(capture / 'capture_receipt.json'))
    source = [value.raw.tobytes() for value in capture_inputs(capture, nodes).values()]
    if not packaged or source != packaged:
        raise Refusal('deployment inputs are not the source user inputs in order')
    staged = step_inputs.load_inputs(deployment, inputs)
    _evaluate(out, destination, destination.with_name(destination.name + '.log'), inputs)
    _supplied_reference(out / 'kit', deployment, staged, destination)  # atlasc run must accept it for these inputs
    return destination
