"""Exercise an installed target outside the checkout, including Merlin's real CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from atlas_mlir import toolchain


def run(*args):
    result = subprocess.run(list(map(str, args)), text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stdout}\n{result.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--quantized', action='store_true')
    args = parser.parse_args()
    assert not os.environ.get('ATLAS_OOT_BIN_DIR')
    assert not os.environ.get('PYTHONPATH')
    assert 'site-packages' in str(toolchain.bin_dir())
    assert 'site-packages' in str(toolchain.data_root())
    run(sys.executable, '-m', 'atlas_mlir', 'doctor', *(['--capture'] if args.capture else []))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run(sys.executable, '-m', 'atlas_mlir', 'demo', '-o', root / 'demo')
        assert (root / 'demo/compiled/program.bin').stat().st_size > 0
        if not args.capture:
            return
        loader = root / 'loader.py'
        loader.write_text('import torch\nclass Model(torch.nn.Module):\n'
                          '    def forward(self,a,b): return a @ b\n'
                          'def get_model_and_inputs():\n'
                          '    return Model().eval(), (torch.ones(32,32),torch.ones(32,32))\n')
        policy = root / 'precision.py'
        # Explicit test domain; these are the provider's existing policy names.
        policy.write_text('from atlas_mlir_native.binding import TARGET_IDENTITY\n'
                          'from atlas_mlir_native.application_precision import SCALED_CONTRACTION_POLICY\n'
                          'from atlas_mlir_native.row_softmax import SOFTMAX_POLICY\n'
                          'from atlas_mlir_native.pointwise import POINTWISE_POLICY\n'
                          'def get_precision_policy(source_sha256):\n'
                          '    return dict(schema="atlas.diagnostic_application_precision.v4",\n'
                          '        source_sha256=source_sha256, target_identity=TARGET_IDENTITY,\n'
                          '        contraction=dict(policy=SCALED_CONTRACTION_POLICY, operand_scale_exponents=[4,4]),\n'
                          '        row_softmax=SOFTMAX_POLICY, host_support=["transpose","reshape","copy"],\n'
                          '        epilogues=["relu"], pointwise=POINTWISE_POLICY)\n')
        out = root / 'build'
        run(sys.executable, '-m', 'atlas_mlir', 'build', loader, '--dtype', 'fp32',
            '--precision-policy', policy, '--application-memory-bytes', '1048576', '-o', out)
        report = json.loads((out / 'report.json').read_text())
        assert report['status'] == 'verified'
        assert report['route'] == 'merlin-model2mlir-application'
        assert list((out / 'application').rglob('program.bin'))
        manifest = json.loads((out / 'application/manifest.json').read_text())
        assert (out / 'application/c-package' / manifest['application_interface']['source']).is_file()
        assert not (out / 'compile-inputs/inputs.npz').exists()
        if args.quantized:
            check_quantized(root, out / 'snapshot')
    print('Installed wheel: capture, staging, application emission and replay passed (execution not run)')


def check_quantized(root, snapshot):
    from atlas_mlir.torchao import activate
    activate()
    import torch
    import algos
    import quant_export
    class Fixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)
        def forward(self, value):
            return self.proj(value)
    model = Fixture().eval()
    algo = algos.make('granularity:w=channel,a=token,rule=nc')
    options = {'crosscheck': 'off', 'mxu': 'ipt', 'mxu_impl': 'torch'}
    report = algos.quantize(model, 'ref_conv', [], options, algo=algo)
    export = root / 'export'
    quant_export.export(model, export, mode='ref_conv', report=report, algo=algo,
                        algo_spec='granularity:w=channel,a=token,rule=nc', options=options, skip=[])
    # Only this test's own in-memory model is pickled. The loader returns a live
    # module, not an ExportedProgram. No conversion is allowed during capture.
    model._atlas_export_dir = export
    model.proj.weight = torch.nn.Parameter(
        torch.empty((32, 32), dtype=torch.bfloat16, device="meta"), requires_grad=False)
    torch.save(model.proj, root / 'fixture.pt')
    loader = root / 'quantized.py'
    loader.write_text('import torch\nfrom atlas_mlir_native.capture import normalize_capture, prepare_model\nfrom pathlib import Path\nfrom unittest.mock import patch\n'
        'class Model(torch.nn.Module):\n'
        '    def __init__(self, proj):\n'
        '        super().__init__(); self.proj = proj\n'
        '    def forward(self, value): return self.proj(value)\n'
        'def get_model_and_inputs():\n'
        '    for name in ("load_quantized.apply_export", "algos.base.weight_cast", "quant_modes.quantize"):\n'
        '        patch(name, side_effect=AssertionError("weight conversion during capture")).start()\n'
        '    model = Model(torch.load(Path(__file__).with_name("fixture.pt"), weights_only=False)).eval()\n'
        '    model._atlas_export_dir = Path(__file__).with_name("export")\n'
        '    return prepare_model(model, (torch.ones(32,32,dtype=torch.bfloat16),),\n'
        '        model._atlas_export_dir / "manifest.json", model._atlas_export_dir / "policy.json",\n'
        f'        expected_contract_sha256={hashlib.sha256((export / "manifest.json").read_bytes()).hexdigest()!r},\n'
        f'        expected_policy_sha256={hashlib.sha256((export / "policy.json").read_bytes()).hexdigest()!r})\n')
    policy = root / 'quantized_policy.py'
    policy.write_text('from atlas_mlir_native.binding import TARGET_IDENTITY\n'
        'from atlas_mlir_native.application_precision import CONTRACTION_POLICY\n'
        'from atlas_mlir_native.row_softmax import SOFTMAX_POLICY\n'
        'from atlas_mlir_native.already_quantized import POLICY\n'
        'def get_precision_policy(source_sha256):\n'
        '    return dict(schema="atlas.diagnostic_application_precision.v1",source_sha256=source_sha256,\n'
        '        target_identity=TARGET_IDENTITY,contraction=CONTRACTION_POLICY,row_softmax=SOFTMAX_POLICY,\n'
        '        host_support=["transpose","reshape","copy"], already_quantized=POLICY)\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.iterdir() if p.is_file()}
    out = root / 'quantized-build'
    run(sys.executable, '-m', 'atlas_mlir', 'build', loader, '--dtype', 'bf16', '--contract', export / 'manifest.json',
        '--quant-policy', export / 'policy.json', '--expected-contract-sha256', hashes['manifest.json'],
        '--expected-policy-sha256', hashes['policy.json'], '--precision-policy', policy,
        '--application-memory-bytes', '1048576', '--snapshot', snapshot, '-o', out)
    assert json.loads((out / 'report.json').read_text())['status'] == 'verified'
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in export.iterdir() if p.is_file()}
    assert list((out / 'application').rglob('program.bin'))


if __name__ == '__main__':
    main()
