"""Deployment intake binds the export wrapper prefix and admits inputs by declared dtype."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from atlas_mlir import deploy
from atlas_mlir.export_intake import host_math_contract, wrapper_prefix


def test_unprefixed_and_prefixed_sites_select_one_prefix():
    assert wrapper_prefix(["linear"], {"linear": {}}) == ""
    assert wrapper_prefix(["model.lm_head", "model.model.layers.0.q"], {"lm_head": {}, "model.layers.0.q": {}}) == "model."


@pytest.mark.parametrize("fx_sites,export_sites", [
    ([], {"a": {}}),
    (["x.a", "y.b"], {"a": {}, "b": {}}),
    (["xa"], {"a": {}}),
    (["a.b.c"], {"b.c": {}, "c": {}}),
])
def test_missing_or_ambiguous_prefix_refuses(fx_sites, export_sites):
    with pytest.raises(ValueError, match="one export wrapper prefix"):
        wrapper_prefix(fx_sites, export_sites)


def test_host_math_contract_declares_the_built_rv64_archive(tmp_path):
    from merlin.compiler.lower.c.unary_math import validate_policy
    from atlas_mlir.deploy import digest

    contract = host_math_contract(tmp_path)
    assert validate_policy(contract) == contract
    assert contract["linkage"]["archive_sha256"] == digest(tmp_path / "rv64" / "libmerlin-host-math.a")


class Reached(Exception):
    pass


def bundle(root, dtypes, nbytes):
    tensor = lambda dtype: SimpleNamespace(dtype=dtype, nbytes=nbytes)
    fixture = SimpleNamespace(inputs=[tensor(d) for d in dtypes], outputs=[tensor('F32')])
    return SimpleNamespace(fixtures=[fixture], metadata={}, inventory={}, root=root, export_dir=root)


def admit(root, dtypes, nbytes=64):
    """Run the real admission stage up to capture, which the test replaces."""
    for name in ('manifest.json', 'policy.json'):
        (root / name).write_text('{}')
    with patch('atlas_mlir_native.export_loader.load_export', return_value=bundle(root, dtypes, nbytes)), \
            patch('atlas_mlir.driver.capture', side_effect=Reached) as capture:
        try:
            deploy.Stages(jtag=True).admit(root, root)
        except Reached:
            pass
    return capture


@pytest.mark.parametrize('dtypes', [['BF16', 'F32'], ['F32'], ['BF16', 'F32', 'I64', 'BOOL']])
def test_mixed_declared_dtypes_reach_capture(tmp_path, dtypes):
    assert admit(tmp_path, dtypes).call_count == 1
    assert (tmp_path / 'export-intake.json').is_file()


@pytest.mark.parametrize('dtype', ['F8_E4M3', 'F8_E5M2'])
def test_float8_input_refuses_before_capture(tmp_path, dtype):
    with pytest.raises(deploy.Refusal, match='float8 fixture input conversion is unqualified'):
        admit(tmp_path, ['BF16', dtype])
    assert not (tmp_path / 'export-intake.json').exists()


def test_boundary_storage_over_the_window_reaches_capture(tmp_path):
    import json

    assert admit(tmp_path, ['BF16', 'F32'], nbytes=786432 // 2).call_count == 1
    record = json.loads((tmp_path / 'export-intake.json').read_text())
    assert record['input_output_lower_bound_bytes'] == 3 * 786432 // 2


def test_user_without_passwd_entry_refuses_before_capture(tmp_path):
    from atlas_mlir import driver

    loader = tmp_path / 'loader.py'
    loader.write_text('')
    missing = KeyError('getpwuid(): uid not found')
    with patch('pwd.getpwuid', side_effect=missing), patch.object(driver, '_run') as run:
        with pytest.raises(ValueError, match='no passwd entry and USER is unset.*useradd.*set USER'):
            driver.capture(loader, out=tmp_path / 'refused', env={'HOME': str(tmp_path)})
        assert run.call_count == 0
        driver.capture(loader, out=tmp_path / 'named', env={'HOME': str(tmp_path), 'USER': 'tester'})
        assert run.call_count == 1


def test_board_capacity_bound_counts_every_captured_weight_and_the_fixture_io(tmp_path):
    import json
    from dataclasses import replace
    from atlas_mlir.export_intake import check_board_capacity
    from atlas_mlir_native import soc

    capacity = soc.SOC.memory_region('ddr').capacity_bytes
    header = json.dumps({'__metadata__': {}, 'a': dict(dtype='U8', shape=[2], data_offsets=[0, capacity // 2]),
                         'b': dict(dtype='U8', shape=[2], data_offsets=[capacity // 2, capacity - 64])}).encode()
    weights = tmp_path / 'weights.safetensors'
    weights.write_bytes(len(header).to_bytes(8, 'little') + header)
    check_board_capacity(weights, 64)
    with pytest.raises(ValueError, match=f'DDR_CAPACITY_EXCEEDED: {capacity + 1} bytes > {capacity} bytes'):
        check_board_capacity(weights, 65)
    # The bound is the declared DDR region, not a copied constant.
    smaller = replace(soc.SOC, memory_regions=tuple(replace(region, capacity_bytes=capacity // 2)
                      if region.id == 'ddr' else region for region in soc.SOC.memory_regions))
    with patch.object(soc, 'SOC', smaller), pytest.raises(ValueError, match='DDR_CAPACITY_EXCEEDED'):
        check_board_capacity(weights, 64)
