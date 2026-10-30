"""Offline ResNet50 precision experiments using provider host-plan artifacts.

Requires optional numpy, torch, and torchvision packages. Inputs must already be
preprocessed float32 NCHW images; this tool never downloads models or datasets.
"""

import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torchvision

from .. import arithmetic


def add_arguments(parser):
    parser.add_argument('--artifact-dir', type=Path, required=True,
                        help='Provider directory containing host-plan.json')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='local torchvision ResNet50 state dictionary')
    parser.add_argument('--input', type=Path, required=True,
                        help='Preprocessed f32 binary, .npy, or .npz inputs')
    parser.add_argument('--reference', type=Path,
                        help='optional provider arithmetic reference receipt')
    parser.add_argument('--labels', type=Path,
                        help='JSON labels list, or object with samples')
    parser.add_argument('--input-index', type=int,
                        help='evaluate only this index in an input array')


def _sha(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().contiguous().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _read(path, shape, dtype=np.float32):
    return torch.from_numpy(np.fromfile(Path(path).resolve(), dtype=dtype)
                            .reshape(shape).copy())


class Engine:
    """Load a model and bind checkpoint weights to the provider encoding."""

    def __init__(self, artifact_dir, checkpoint):
        self.artifact_dir = Path(artifact_dir).resolve()
        plan_path = self.artifact_dir / 'host-plan.json'
        self.plan = json.loads(plan_path.read_text())
        self.nodes = {node['node']: node for node in self.plan['nodes']}
        model = torchvision.models.resnet50(weights=None).eval()
        checkpoint = Path(checkpoint).resolve()
        model.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                        weights_only=True))
        self.checkpoint_sha256 = _sha(checkpoint.read_bytes())
        self.graph = torch.fx.symbolic_trace(model)
        self.weights = {}
        self.batchnorm = {}
        for node in self.plan['nodes']:
            if node['kind'] != 'call_module':
                continue
            module = self.graph.get_submodule(node['target'])
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                if isinstance(module, torch.nn.Conv2d):
                    right = module.weight.detach().flatten(1).T.contiguous()
                else:
                    right = module.weight.detach().T.contiguous()
                self.weights[node['node']] = {'source': right}
            elif isinstance(module, torch.nn.BatchNorm2d):
                hp = node['host_params']
                self.batchnorm[node['node']] = _read(
                    self.artifact(hp['scale_then_bias']['path']),
                    (2, hp['channels']))
        if len(self.weights) != 54:
            raise ValueError('expected all 54 ResNet50 contractions')

    def artifact(self, path):
        """Prefer colocated payloads to allow provider relocation."""
        candidate = (self.artifact_dir / Path(path).name).resolve()
        return candidate if candidate.is_file() else Path(path).resolve()

    def weight(self, site, axis, two_term):
        cached = self.weights[site]
        if two_term:
            key = 'two_' + axis
            if key not in cached:
                cached[key] = arithmetic.two_term(
                    cached['source'], low_axis=axis)
            (high, exponent), low = cached[key]
            return high, exponent, low
        if axis not in cached:
            cached[axis] = arithmetic.pack(cached['source'], axis=axis)
        high, exponent = cached[axis]
        return high, exponent, None


class Runner(torch.fx.Interpreter):
    """Keep host arithmetic fixed while replacing the 54 contractions."""

    def __init__(self, engine, args, reference=None, native=False):
        super().__init__(engine.graph)
        self.engine, self.args = engine, args
        self.reference, self.native = reference, native
        self.gate = bool(reference) and not native and all(
            arithmetic.settings(args, site) == ('tensor', 'tensor', 0, False)
            for site in engine.weights)
        self.current = ''
        self.kernels = []
        self.node_checks = []

    def run_node(self, node):
        self.current = node.name
        result = super().run_node(node)
        if isinstance(result, torch.Tensor):
            if not torch.isfinite(result).all():
                raise ValueError(f'nonfinite output at {node.name}')
            if self.gate:
                ref = next(x for x in self.reference['nodes']
                           if x['node'] == node.name)
                exact = _sha(result) == ref['output']['sha256']
                self.node_checks.append({'node': node.name, 'exact': exact})
                # The archived host pool uses a distinct FP32 division order.
                # Its intermediate difference is tolerated only at these nodes;
                # every contraction and all final logits must remain exact.
                if not exact and node.name not in ('avgpool', 'flatten'):
                    raise ValueError(f'reference node differs at {node.name}')
        return result

    def call_module(self, target, args, kwargs):
        module = self.fetch_attr(target)
        if self.native:
            return module(*args, **kwargs)
        node = self.engine.nodes.get(self.current)
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if isinstance(module, torch.nn.Conv2d):
                left = torch.nn.functional.unfold(
                    args[0], module.kernel_size, dilation=module.dilation,
                    padding=module.padding,
                    stride=module.stride)[0].T.contiguous()
            else:
                left = args[0].reshape(-1, module.in_features).contiguous()
            a_axis, b_axis, chunk, two_term = arithmetic.settings(
                self.args, self.current)
            a, ae = arithmetic.pack(left, axis=a_axis)
            b, be, low = self.engine.weight(self.current, b_axis, two_term)
            # Exact device postscale of one full-K chain equals the FP64 merge.
            value = arithmetic.contraction(
                a, b, ae, be, k_chunk=chunk, low=low)
            m, k = left.shape
            n = b.shape[1]
            mp, kp, npad = node['padded_mkn']
            chunks = math.ceil(kp / chunk) if chunk else 1
            branches = 2 if two_term else 1
            chains = chunks * branches
            tiles = (mp // 32) * (npad // 32)
            record = dict(node=self.current, site=target, mkn=[m, k, n],
                          activation_scale=a_axis, weight_scale=b_axis,
                          k_chunk=chunk, two_term=two_term,
                          independent_chains=chains,
                          baseline_k32_instructions=tiles * (kp // 32),
                          k32_instructions=branches * tiles * (kp // 32),
                          padded_weight_bytes=branches * kp * npad,
                          raw_partial_bytes=2048 * chains * tiles)
            if self.gate:
                ref = self.reference['kernels'][len(self.kernels)]
                checks = dict(
                    scale_exact=(int(torch.as_tensor(ae).item())
                                 == ref['a_scale_exponent']),
                    contraction_exact=(
                        _sha(value) == ref['target_output']['sha256']))
                if not all(checks.values()):
                    raise ValueError(
                        f'reference contraction differs at '
                        f'{self.current}: {checks}')
                record.update(checks)
            self.kernels.append(record)
            if isinstance(module, torch.nn.Conv2d):
                return value.T.reshape(node['output_shape']).contiguous()
            result = value.reshape(node['output_shape'])
            return result + module.bias if module.bias is not None else result
        if isinstance(module, torch.nn.BatchNorm2d):
            params = self.engine.batchnorm[self.current]
            return (args[0] * params[0].reshape(1, -1, 1, 1)
                    + params[1].reshape(1, -1, 1, 1))
        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            data = args[0].flatten(2)
            total = torch.zeros(data.shape[:2], dtype=torch.float32)
            for i in range(data.shape[2]):
                total = total + data[:, :, i]
            return (total / data.shape[2]).reshape(*data.shape[:2], 1, 1)
        return module(*args, **kwargs)


def evaluate(engine, image, args, reference=None, label=None):
    """Return serializable metrics, logits, and optional exact gates."""
    if image.shape != (1, 3, 224, 224) or image.dtype != torch.float32:
        raise ValueError('each input must be float32 [1, 3, 224, 224]')
    if reference:
        placeholder = next(x for x in reference['nodes']
                           if x['kind'] == 'placeholder')
        if _sha(image) != placeholder['output']['sha256']:
            raise ValueError('reference receipt belongs to a different input')
    started = time.monotonic()
    with torch.inference_mode():
        native = Runner(engine, args, native=True).run(image).flatten()
        runner = Runner(engine, args, reference=reference)
        logits = runner.run(image).flatten()
    if len(runner.kernels) != 54 or logits.numel() != 1000:
        raise ValueError('incomplete ResNet50 forward')
    top = torch.argsort(logits, descending=True)[:5].tolist()
    ntop = torch.argsort(native, descending=True)[:5].tolist()
    record = dict(input_sha256=_sha(image), logits_sha256=_sha(logits),
                  native_logits_sha256=_sha(native), top5=top, native_top5=ntop,
                  top1_agreement=top[0] == ntop[0],
                  top5_set_agreement=set(top) == set(ntop),
                  top5_ordered_agreement=top == ntop,
                  top5_overlap_fraction=len(set(top) & set(ntop)) / 5,
                  relative_l2=float(
                      torch.linalg.vector_norm(
                          logits.double() - native.double())
                      / (torch.linalg.vector_norm(native.double()) + 1e-30)),
                  logits=logits.tolist(), native_logits=native.tolist(),
                  kernels=runner.kernels,
                  elapsed_seconds=time.monotonic() - started)
    if label is not None:
        record.update(label=int(label), top1_correct=top[0] == int(label),
                      native_top1_correct=ntop[0] == int(label))
    if runner.gate:
        if record['logits_sha256'] != reference['final_logits']['sha256']:
            raise ValueError('final logits differ from reference')
        record['baseline_gate'] = dict(
            status='EXACT_REFERENCE', contractions_exact=54, scales_exact=54,
            logits_exact=1000, node_checks=runner.node_checks)
    else:
        record['baseline_gate'] = dict(status='NOT_EXTERNALLY_GATED')
    return record


def run(args):
    engine = Engine(args.artifact_dir, args.checkpoint)
    path = Path(args.input).resolve()
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as archive:
            key = 'images' if 'images' in archive.files else 'inputs'
            images = archive[key].copy()
    elif path.suffix == '.npy':
        images = np.load(path, allow_pickle=False)
    else:
        images = np.fromfile(path, dtype=np.float32).reshape(1, 3, 224, 224)
    if images.dtype != np.float32:
        raise ValueError('input arrays must be float32')
    if images.ndim == 3:
        images = images[None]
    if images.ndim == 4:
        images = images[:, None]
    if images.ndim != 5 or images.shape[1:] != (1, 3, 224, 224):
        raise ValueError('expected [N, 3, 224, 224] or [N, 1, 3, 224, 224]')
    reference = (json.loads(Path(args.reference).resolve().read_text())
                 if args.reference else None)
    labels = None
    if args.labels:
        labels = json.loads(Path(args.labels).resolve().read_text())
        if isinstance(labels, dict):
            labels = labels['samples']
        labels = [x['label'] if isinstance(x, dict) else x for x in labels]
        if len(labels) != len(images):
            raise ValueError('labels and input array lengths differ')
    indices = ([args.input_index] if args.input_index is not None
               else range(len(images)))
    records = []
    for index in indices:
        if not 0 <= index < len(images):
            raise ValueError('input index out of range')
        record = evaluate(engine, torch.from_numpy(images[index].copy()), args,
                          reference, labels[index] if labels else None)
        record['index'] = index
        records.append(record)
    if getattr(args, 'output_dir', None):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / 'logits.npz', **{
            f'{kind}_{record["index"]}': np.asarray(
                record[key], dtype=np.float32)
            for record in records
            for kind, key in [('mode', 'logits'), ('native', 'native_logits')]})
    return dict(model='resnet50', precision_mode=args.precision_mode,
                scope=('offline CPU arithmetic emulation; '
                       'no hardware execution or cycle claim'),
                checkpoint_sha256=engine.checkpoint_sha256,
                versions=dict(torch=torch.__version__,
                              torchvision=torchvision.__version__,
                              numpy=np.__version__), records=records)
