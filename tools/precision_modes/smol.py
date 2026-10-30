"""Offline SmolVLA projections with the shared precision-mode arithmetic.

Requires a local LeRobot installation (or an explicit source overlay), policy
checkpoint, backbone, and prepared observation archive. Host operations retain
the model's native Torch behavior; only Linear and patch Conv2d are replaced.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import arithmetic


def array(value):
    value = value.detach().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().copy()
    return value.numpy().copy()


def metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("Metric tensor shapes differ")
    def values(value):
        if value.dtype == np.uint16:
            tensor = torch.from_numpy(value.copy()).view(torch.bfloat16)
            return tensor.float().numpy()
        return value
    a, b = (values(x).astype(np.float64) for x in (actual, reference))
    error = a - b
    numerator = float(np.linalg.norm(error))
    denominator = float(np.linalg.norm(b))
    relative = (numerator / denominator if denominator
                else (0.0 if numerator == 0 else None))
    return dict(max_abs=float(np.abs(error).max()),
                mean_abs=float(np.abs(error).mean()), relative_l2=relative,
                count=int(actual.size),
                bytes_exact=actual.tobytes() == reference.tobytes())


def load_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def check_reference(trajectory, path):
    reference = load_archive(path)
    if set(reference) != set(trajectory):
        raise ValueError("Captured trajectory fields differ")
    checks = {name: (trajectory[name].shape == value.shape
                     and trajectory[name].dtype == value.dtype
                     and trajectory[name].tobytes() == value.tobytes())
              for name, value in reference.items()}
    if not all(checks.values()):
        failed = ", ".join(k for k, v in checks.items() if not v)
        raise ValueError("Captured trajectory differs: " + failed)
    return dict(status="PASS", fields=len(checks), bytes_exact=True)


class Engine:
    """Load once and evaluate multiple recipes against one fresh native run."""

    def __init__(self, args):
        for name in ("checkpoint", "backbone"):
            if not Path(getattr(args, name)).is_dir():
                raise ValueError("Expected a local " + name + " directory")
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(
            str(args.checkpoint), local_files_only=True)
        if (config.action_feature is None
                or tuple(config.action_feature.shape) != (6,)):
            raise ValueError("Smol driver requires six SO100 action dimensions")
        config.device = "cpu"
        config.vlm_model_name = str(args.backbone)
        torch.manual_seed(0)
        self.policy = SmolVLAPolicy.from_pretrained(
            str(args.checkpoint), config=config, local_files_only=True,
            strict=True).eval().requires_grad_(False)
        self.model = self.policy.model
        self.inputs = {
            name: torch.from_numpy(value)
            for name, value in load_archive(args.observation).items()}
        required = {"images", "image_masks", "language_tokens",
                    "language_mask", "state", "noise"}
        if not required.issubset(self.inputs):
            missing = ", ".join(sorted(required - self.inputs.keys()))
            raise ValueError("Missing observation fields: " + missing)
        self.linear = arithmetic.Linear(args)
        self.native = None

    def projection(self, name, module, incoming, args):
        if isinstance(module, torch.nn.Conv2d):
            # The policy's only convolution is the unpadded, nonoverlapping
            # vision patch embedding. Reject other convolutions explicitly.
            if (module.groups != 1 or module.padding not in ((0, 0), "valid")
                    or module.dilation != (1, 1)
                    or module.stride != module.kernel_size):
                raise ValueError("Unsupported SmolVLA convolution: " + name)
            batch, _, height, width = incoming.shape
            kh, kw = module.kernel_size
            patches = F.unfold(incoming, module.kernel_size,
                               stride=module.stride)
            matrix = patches.transpose(1, 2).reshape(
                -1, module.weight[0].numel())
            weight = module.weight.flatten(1).t().contiguous()
            shape = (batch, height // kh, width // kw, module.out_channels)
        else:
            matrix = incoming.reshape(-1, module.in_features)
            weight = module.weight.t().contiguous()
            shape = (*incoming.shape[:-1], module.out_features)
        out = self.linear(name, matrix, lambda: weight)
        out = out.to(incoming.dtype).reshape(shape)
        if module.bias is not None:
            out = out + module.bias
        if isinstance(module, torch.nn.Conv2d):
            out = out.permute(0, 3, 1, 2).contiguous()
        return out

    def sample(self, args=None):
        model, inputs = self.model, self.inputs
        trajectory, handles = {}, []
        calls = 0
        originals = {name: getattr(model, name)
                     for name in ("embed_prefix", "denoise_step")}
        original_forward = model.vlm_with_expert.forward
        step = -1

        def embed(*positional, **kwargs):
            result = originals["embed_prefix"](*positional, **kwargs)
            keys = ("prefix_embeddings", "prefix_pad_mask",
                    "prefix_attention_mask")
            for key, value in zip(keys, result):
                trajectory[key] = array(value)
            return result

        def forward(*positional, **kwargs):
            result = original_forward(*positional, **kwargs)
            if kwargs.get("fill_kv_cache"):
                for field in ("key_states", "value_states"):
                    key = "prefix_" + field.replace("states", "cache")
                    trajectory[key] = array(torch.stack([
                        result[1][i][field] for i in range(len(result[1]))]))
            return result

        def denoise(*positional, **kwargs):
            nonlocal step
            step += 1
            stem = f"step_{step:02d}"
            trajectory[stem + "_input_state"] = array(kwargs["x_t"])
            trajectory[stem + "_timestep"] = array(kwargs["timestep"])
            result = originals["denoise_step"](*positional, **kwargs)
            trajectory[stem + "_velocity"] = array(result)
            return result

        def hook(name):
            def replace(module, arguments, output):
                nonlocal calls
                calls += 1
                return self.projection(name, module, arguments[0], args)
            return replace

        model.embed_prefix, model.denoise_step = embed, denoise
        model.vlm_with_expert.forward = forward
        rng = torch.random.get_rng_state().clone()
        try:
            if args is not None:
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                        handles.append(module.register_forward_hook(hook(name)))
            with torch.inference_mode():
                final = model.sample_actions(
                    [inputs["images"][:, i]
                     for i in range(inputs["images"].shape[1])],
                    [inputs["image_masks"][:, i]
                     for i in range(inputs["image_masks"].shape[1])],
                    inputs["language_tokens"], inputs["language_mask"],
                    inputs["state"],
                    noise=inputs["noise"])
        finally:
            for handle in handles:
                handle.remove()
            del model.embed_prefix, model.denoise_step
            del model.vlm_with_expert.forward
        if not torch.equal(rng, torch.random.get_rng_state()):
            raise ValueError("Sampling consumed RNG despite explicit noise")
        if final.ndim != 3 or final.shape[1] != 50 or final.shape[2] < 6:
            raise ValueError("Expected fifty SO100 timesteps with six actions")
        trajectory["final_full_state"] = array(final)
        trajectory["final_actions"] = array(final[:, :, :6])
        for index in range(step + 1):
            key = f"step_{index:02d}_output_state"
            source = (f"step_{index+1:02d}_input_state" if index < step
                      else "final_full_state")
            trajectory[key] = trajectory[source].copy()
        return trajectory, calls

    def evaluate(self, args):
        started = time.monotonic()
        if self.native is None:
            self.native, _ = self.sample()
        trajectory, calls = self.sample(args)
        gate = check_reference(trajectory, args.reference) if args.reference else None
        actions = trajectory["final_actions"]
        return dict(model="smol", projection_calls=calls,
                    action_metrics=metrics(actions,
                                           self.native["final_actions"]),
                    action_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),
                    trajectory_metrics={name: metrics(value, self.native[name])
                                        for name, value in trajectory.items()
                                        if value.dtype != np.bool_},
                    captured_reference=gate,
                    elapsed_seconds=time.monotonic() - started,
                    arithmetic=("ordered BF16 MAC; ascending FP64 scale "
                                "and merge; direct BF16 store"))


def run(args):
    return Engine(args).evaluate(args)
