import hashlib
import json
import struct

import numpy as np
import pytest

from atlas_mlir_native.reference_eval.source_io import (
    capture_inputs, capture_sources, checked_path, safetensor_constants,
)


def test_paths_reject_before_following_symlink(tmp_path):
    destination = tmp_path / 'uninspected' / 'absent'
    link = tmp_path / 'source'
    link.symlink_to(destination)
    with pytest.raises(ValueError, match='symlink'):
        checked_path(link)


def test_capture_receipt_requires_trusted_digest(tmp_path):
    (tmp_path / 'capture_receipt.json').write_text('{}')
    with pytest.raises(ValueError, match='trusted digest'):
        capture_sources(tmp_path, receipt_sha256='0' * 64)


def test_capture_checks_source_hashes(tmp_path):
    inventory = {}
    for name, data in [('frontend-trace.json', b'{"graphs":{"quantized":{"nodes":[]}}}'),
                       ('weights.safetensors', b'weights'), ('inputs.npz', b'inputs')]:
        (tmp_path / name).write_bytes(data)
        inventory[name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    receipt = tmp_path / 'capture_receipt.json'
    receipt.write_text(json.dumps(dict(artifacts=inventory)))
    trusted = hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert capture_sources(tmp_path, receipt_sha256=trusted)[1] == []
    (tmp_path / 'inputs.npz').write_bytes(b'mutant')
    with pytest.raises(ValueError, match='artifact differs: inputs.npz'):
        capture_sources(tmp_path, receipt_sha256=trusted)


def tensor_file(path, entries, data):
    header = json.dumps(entries).encode()
    path.write_bytes(struct.pack('<Q', len(header)) + header + data)


def test_raw_bf16_and_f32_preserve_words(tmp_path):
    path = tmp_path / 'weights.safetensors'
    tensor_file(path, {'a': dict(dtype='BF16', shape=[2], data_offsets=[0,4]),
                       'b': dict(dtype='F32', shape=[1], data_offsets=[4,8])},
                bytes.fromhex('803f00800000803f'))
    result = safetensor_constants(path)
    assert result['a'].raw.tolist() == [0x3f80, 0x8000]
    assert result['b'].raw.tolist() == [0x3f800000]


def test_overlapping_tensor_storage_refused(tmp_path):
    path = tmp_path / 'weights.safetensors'
    tensor_file(path, {n: dict(dtype='BF16', shape=[2], data_offsets=[0,4]) for n in ['a','b']}, b'1234')
    with pytest.raises(ValueError, match='overlapping'):
        safetensor_constants(path)


def test_parent_traversal_refused(tmp_path):
    with pytest.raises(ValueError, match="parent traversal"):
        checked_path(tmp_path / ".." / "absent")


def test_directory_symlink_refused_before_child_access(tmp_path):
    link = tmp_path / "linked-directory"
    link.symlink_to(tmp_path / "uninspected", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        checked_path(link / "child")


def test_supplied_inputs_replace_the_capture_inputs_in_source_order(tmp_path):
    nodes = [dict(id='w', op='placeholder', input_kind='PARAMETER', results=[dict(dtype='bfloat16', shape=[2])]),
             dict(id='x', op='placeholder', input_kind='USER_INPUT', results=[dict(dtype='bfloat16', shape=[2])])]
    np.savez(tmp_path / 'inputs.npz', in0=np.array([1, 2], dtype=np.float32))
    assert capture_inputs(tmp_path, nodes)['x'].raw.tolist() == [0x3f80, 0x4000]
    supplied = tmp_path / 'new.npz'
    np.savez(supplied, input_000=np.array([0x4040, 0xc000], dtype=np.uint16))
    assert capture_inputs(tmp_path, nodes, supplied)['x'].raw.tolist() == [0x4040, 0xc000]
    for members in (dict(in0=np.uint16([0, 0])), dict(input_000=np.uint16([0, 0]), input_001=np.uint16([0, 0]))):
        np.savez(supplied, **members)
        with pytest.raises(ValueError, match='exactly the source user inputs'):
            capture_inputs(tmp_path, nodes, supplied)
    np.savez(supplied, input_000=np.uint16([0, 0, 0]))
    with pytest.raises(ValueError, match='shape differs'):
        capture_inputs(tmp_path, nodes, supplied)
