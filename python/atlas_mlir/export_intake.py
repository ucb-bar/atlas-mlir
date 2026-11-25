"""Saved export intake shared by deployment and its capture subprocess."""
from __future__ import annotations

import json
import os
from pathlib import Path


def admit_input_dtype(dtype):
    """Admit one fixture input in its own declared dtype; capture never casts it."""
    if dtype.startswith('F8'):
        raise ValueError('float8 fixture input conversion is unqualified')


def wrapper_prefix(fx_sites, export_sites):
    """The single module prefix p (empty or ending in '.') with every FX site == p + an export site."""
    candidates = None
    for name in fx_sites:
        found = {name[:len(name) - len(site)] for site in export_sites
                 if name.endswith(site) and (len(name) == len(site) or name[len(name) - len(site) - 1] == '.')}
        candidates = found if candidates is None else candidates & found
    if not candidates or len(candidates) != 1:
        raise ValueError(f'FX native sites do not determine one export wrapper prefix: {sorted(candidates or [])}')
    return candidates.pop()


def get_model_and_inputs():
    """Load the authenticated FX program and original positional fixture inputs."""
    from atlas_mlir_native.export_loader import load_export
    from .torchao import activate
    import torch
    from m2m.capture.external_quantization import ExternalQuantizationConfig, apply_external_quantization
    from .deploy import digest

    bundle = load_export(os.environ['ATLAS_DEPLOY_EXPORT'])
    activate()
    # Saved native-FP8 programs name the exporter's custom operators.
    import exporter.atlas_fx_ops  # noqa: F401
    bundle.verify_unchanged()
    program = torch.export.load(bundle.graph_path)
    inputs = []
    for record in bundle.fixtures[0].inputs:
        admit_input_dtype(record.dtype)
        tensor = torch.from_numpy(record.to_numpy().copy())
        if record.dtype == 'BF16':
            tensor = tensor.view(torch.bfloat16)
        inputs.append(tensor)
    args, kwargs = torch.utils._pytree.tree_unflatten(inputs, program.call_spec.in_spec)
    if kwargs or any(not isinstance(item, torch.Tensor) for item in args):
        raise ValueError('capture requires flat positional tensor inputs')
    program._atlas_saved_program_path = bundle.graph_path
    program._atlas_saved_program_sha256 = digest(bundle.graph_path)
    program._atlas_export_dir = bundle.export_dir
    program._atlas_export_prefix = wrapper_prefix(
        bundle.metadata['fx_manifest'].get('native_fp8', {}).get('sites', []), bundle.metadata['policy']['sites'])
    program = apply_external_quantization(program, tuple(args), ExternalQuantizationConfig(
        'atlas_fp8', bundle.export_dir / 'manifest.json', bundle.export_dir / 'policy.json',
        digest(bundle.export_dir / 'manifest.json'), digest(bundle.export_dir / 'policy.json')))
    bundle.verify_unchanged()
    # Preserve the saved program and its node identity through the external
    # quantization capture route, which already accepts ExportedProgram.
    return program, tuple(args)


def host_math_contract(out):
    """Build the RV64 host-math archive and declare it as the target unary-math library."""
    from merlin.runtime import host_math_supplier as host_math
    from merlin.runtime.host_math_supplier.build import build
    from merlin.compiler.lower.c.unary_math import OPERATIONS
    from .deploy import digest

    build('rv64', out, host_math.POLICY_PATH)
    functions = sorted({function for _, function in OPERATIONS.values()})
    return dict(schema='merlin.compiled_unary_math.v1', policy='selected-host-math-rne-finite-v1',
                isa='rv64gc', abi='lp64d', effects='enabled-fpu-rne-unobserved-flags-and-errno-v1',
                operations=sorted(OPERATIONS), linkage=dict(
                    schema='merlin.host_linkage_contract.v1', supplier='selected_math_archive',
                    archive_sha256=digest(out / 'rv64' / 'libmerlin-host-math.a'),
                    symbols=sorted(['merlin_host_check_fp_env', *map(host_math.symbol, functions)])))


def check_board_capacity(weights, io_bytes):
    """Early lower bound before the compile: every captured weight and the fixture's inputs and outputs are resident
    together in the declared DDR. The linked ELF decides the complete fit."""
    from atlas_mlir_native.soc import SOC
    capacity = SOC.memory_region('ddr').capacity_bytes
    with Path(weights).open('rb') as stream:
        header = json.loads(stream.read(int.from_bytes(stream.read(8), 'little')))
    resident = sum(row['data_offsets'][1] - row['data_offsets'][0] for name, row in header.items() if name != '__metadata__')
    if resident + io_bytes > capacity:
        raise ValueError(f'DDR_CAPACITY_EXCEEDED: {resident + io_bytes} bytes > {capacity} bytes')


def prepare(export, out, *, board_memory=False):
    """Authenticate, capture with atlascc, and give the build arguments; atlascc build generates the policy.

    board_memory places the deployment in the declared board DDR instead of the on-chip window.
    """
    from atlas_mlir_native.export_loader import load_export
    from .driver import capture
    from .deploy import digest

    bundle = load_export(export)
    for item in bundle.fixtures[0].inputs:
        admit_input_dtype(item.dtype)
    # Boundary storage, never framework golden outputs. Merlin staging places
    # inputs and outputs per region and refuses an unfit region by source name.
    minimum = sum(item.nbytes for item in bundle.fixtures[0].inputs)
    minimum += sum(item.nbytes for item in bundle.fixtures[0].outputs)
    (out / 'export-intake.json').write_text(json.dumps(dict(
        metadata=bundle.metadata, inventory=bundle.inventory,
        input_output_lower_bound_bytes=minimum), indent=2) + '\n')
    loader = out / 'export-loader.py'
    loader.write_text('from atlas_mlir.export_intake import get_model_and_inputs\n'
                      'from atlas_mlir_native.capture import normalize_capture\n')
    env = os.environ.copy()
    env['ATLAS_DEPLOY_EXPORT'] = str(bundle.root)
    captured = capture(loader, out=out / 'capture', dtype='bf16',
        contract=bundle.export_dir / 'manifest.json',
        quant_policy=bundle.export_dir / 'policy.json',
        expected_contract_sha256=digest(bundle.export_dir / 'manifest.json'),
        expected_policy_sha256=digest(bundle.export_dir / 'policy.json'), env=env)
    bundle.verify_unchanged()
    if board_memory:
        check_board_capacity(captured / 'weights.safetensors', minimum)
    from atlas_mlir_native.provider_deployment import WINDOW_BYTES
    from atlas_mlir_native.soc import SOC

    memory = SOC.memory_region('ddr').capacity_bytes if board_memory else WINDOW_BYTES
    arguments = ['--capture', captured, '--export', bundle.export_dir,
        '--application-deployment', 'board' if board_memory else 'onchip',
        '--application-name', 'capsule', '--application-memory-bytes',
        str(memory),
        '--application-parameters', 'external', '-o', out / 'compiled']
    snapshot = os.environ.get('MERLIN_NATIVE_SNAPSHOT')
    # An installed snapshot is built for the on-chip window; atlascc build makes the board's from the pinned Merlin.
    if snapshot and not board_memory:
        arguments += ['--snapshot', Path(snapshot)]
    return arguments
