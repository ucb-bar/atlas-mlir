"""Validate per-call input tensors against the deployment's declared shapes, dtypes, and fixed byte ranges."""
import math

if __package__:
    from .kit_manifest import safe_path, sha
else:
    from kit_manifest import safe_path, sha

# Storage dtypes. BF16 tensors use uint16 storage bits.
DTYPES = {'bf16': 'uint16', 'f32': 'float32', 'i64': 'int64', 'i32': 'int32',
          'i8': 'int8', 'u8': 'uint8', 'bool': 'bool'}


def load_inputs(deployment, path):
    """Read one .npz of input_000, input_001, ...; each array must match its declared row exactly."""
    import numpy as np
    rows = deployment.get('inputs')
    if not isinstance(rows, list) or not rows:
        raise ValueError('deployment has no declared tensor inputs')
    archive = np.load(safe_path(path), allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError('inputs must be one .npz archive of input_NNN arrays')
    with archive:
        arrays = {name: archive[name] for name in archive.files}
    result = []
    names, indices, intervals = set(), set(), []
    window = deployment['window']
    for row in rows:
        index, name = row['index'], row['id']
        if type(index) is not int or index < 0 or index in indices or name in names:
            raise ValueError('duplicate or invalid input identity')
        indices.add(index); names.add(name)
        kind = row['type']['dtype']; shape = row['type']['shape']
        if kind not in DTYPES or not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('unsupported input dtype or shape')
        dtype = np.dtype(DTYPES[kind]).newbyteorder('<')
        start, size = row['address'], row['bytes']
        if type(start) is not int or type(size) is not int or size <= 0 or math.prod(shape) * dtype.itemsize != size:
            raise ValueError('input shape and byte range disagree')
        if not window['address'] <= start < start + size <= window['address'] + window['bytes']:
            raise ValueError('input exceeds deployment window')
        if any(start < end and begin < start + size for begin, end in intervals):
            raise ValueError('overlapping input ranges')
        intervals.append((start, start + size))
        member = f'input_{index:03d}'
        array = arrays.get(member)
        if array is None:
            raise ValueError('unexpected or missing tensor input: ' + member)
        if list(array.shape) != shape or array.dtype != dtype:
            raise ValueError(f'input shape or dtype mismatch: {member} (declared {kind} {shape})')
        data = array.tobytes(order='C')
        result.append(dict(row=row, data=data, sha256=sha(data)))
    if set(arrays) != {f'input_{index:03d}' for index in indices}:
        raise ValueError('unexpected or missing tensor input')
    return result


def window_inputs(deployment, window):
    """The packaged inputs, read from the bound window image."""
    base, result = deployment['window']['address'], []
    for row in deployment.get('inputs', []):
        data = window[row['address'] - base:row['address'] - base + row['bytes']]
        if row['address'] < base or len(data) != row['bytes']:
            raise ValueError('packaged input is outside the window image: ' + row['id'])
        result.append(dict(row=row, data=data, sha256=sha(data)))
    return result


def qualify(bundle, logs, *, backend='sim', transport_factory=None):
    """Composed positive run and input-changing negative, each on a fresh session.

    A predicted numerical failure requires a completed changed output. An
    arbitrary runtime or transport exception cannot satisfy this gate.
    """
    if __package__:
        from . import composed
    else:
        import composed
    logs = safe_path(logs)
    positive = composed.execute(bundle, logs/'positive', backend=backend,
                                transport_factory=transport_factory)
    try:
        composed.execute(bundle, logs/'negative', backend=backend,
                         mutation=True, transport_factory=transport_factory)
    except composed.BoundaryError:
        return {'positive': positive, 'negative': 'expected numerical failure'}
    raise ValueError('input mutation did not change an authenticated output')
