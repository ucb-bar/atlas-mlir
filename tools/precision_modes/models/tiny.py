"""Offline TinyLlama recipe emulator with local checkpoint comparisons.

The capture supplies column shards, rotary tables, and the default prompt.
``--inputs`` accepts {"prompts": [{"id": "name", "token_ids": [...]}]}.
Token IDs are used verbatim; tokenization and downloads are never performed.
"""

import gc
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from .. import arithmetic
from . import tiny_host as host

SITES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def add_arguments(parser):
    parser.add_argument("--capture", required=True, type=Path,
                        help="Tiny arithmetic capture directory with "
                             "receipt.json")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Local TinyLlama config.json and "
                             "model.safetensors")
    parser.add_argument("--inputs", type=Path,
                        help="JSON prompts with id and token_ids; omit for "
                             "capture prompt")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--native-precision", default="both",
                        choices=("float32", "bfloat16", "both", "none"),
                        help="Local Torch reference precision(s)")


def digest(tensor):
    return hashlib.sha256(
        tensor.contiguous().float().numpy().tobytes()).hexdigest()


def metric(actual, expected):
    delta = actual.double() - expected.double()
    return {"rmse": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(
                delta.norm() / expected.double().norm().clamp_min(1e-30))}


def summarize(logits):
    top = torch.argsort(logits, descending=True, stable=True)[:5]
    return {"token": int(logits.argmax()), "logits_sha256": digest(logits),
            "top5_ids": top.tolist(), "top5_logits": logits[top].tolist()}


def save_logits(args, prompt_index, filename, logits):
    directory = getattr(args, "output_dir", None)
    if directory is None:
        return None
    path = Path(directory) / f"prompt-{prompt_index:03d}" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(logits.contiguous().float().numpy().tobytes())
    return str(path)


class Engine:
    def __init__(self, args):
        self.args = args
        self.capture = args.capture
        self.receipt = json.loads((self.capture / "receipt.json").read_text())
        self.calls = {(c["step"], c["site"]): c for c in self.receipt["calls"]}
        self.cos = self.load(self.receipt["host_parameters"]["rotary_cosine"])
        self.sin = self.load(self.receipt["host_parameters"]["rotary_sine"])
        self.weights = safe_open(str(args.checkpoint / "model.safetensors"),
                                 framework="pt", device="cpu")
        config = json.loads((args.checkpoint / "config.json").read_text())
        self.layers = config["num_hidden_layers"]
        if (config["hidden_size"], config["num_attention_heads"],
                config["num_key_value_heads"]) != (2048, 32, 4):
            raise ValueError("Tiny host recipe requires hidden size 2048, "
                             "32 query heads, and 4 KV heads")
        self.embedding = self.weights.get_tensor("model.embed_tokens.weight")
        self.packed = {}
        self.norms = {}
        self.panel_calls = 0

    def load(self, item):
        raw = (self.capture / item["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Capture payload digest mismatch: " + item["file"])
        return torch.frombuffer(bytearray(raw), dtype=torch.uint16).reshape(
            item["shape"]).view(torch.bfloat16).float()

    def weight(self, site):
        return self.weights.get_tensor(site + ".weight").float()

    def project(self, x, site, step):
        a_axis, b_axis, chunk, two_term = arithmetic.settings(self.args, site)
        call = self.calls[0 if step == 0 else 1, site]
        a, ea = arithmetic.pack(x, axis=a_axis)
        key = (site, b_axis, two_term)
        if key not in self.packed:
            weight = self.weight(site).T.contiguous()
            panels = []
            for panel in call["panels"]:
                lo, hi = panel["columns"]
                original = weight[:, lo:hi]
                low = None
                if two_term:
                    (b, eb), low = arithmetic.two_term(original,
                                                       low_axis=b_axis)
                else:
                    b, eb = arithmetic.pack(original, axis=b_axis)
                panels.append((b, eb, low))
            self.packed[key] = panels
        outputs = []
        for b, eb, low in self.packed[key]:
            y = arithmetic.contraction(a, b, ea, eb, k_chunk=chunk, low=low)
            outputs.append(torch.as_tensor(y).float())
            self.panel_calls += 1
        return torch.cat(outputs, dim=1)

    def norm(self, x, site):
        if site not in self.norms:
            self.norms[site] = self.weight(site)
        return host.norm(x, self.norms[site])

    def forward(self, ids, cache, step):
        offset = 0 if cache[0] is None else cache[0][0].shape[1]
        if offset + len(ids) > self.cos.shape[0]:
            raise ValueError("Sequence exceeds the capture rotary table")
        x = self.embedding[torch.tensor(ids, dtype=torch.int64)].float()
        for layer in range(self.layers):
            prefix = f"model.layers.{layer}."
            entry = x
            n = self.norm(x, prefix + "input_layernorm")
            q, k, v = [self.project(n, prefix + s, step) for s in SITES[:3]]
            attended, cache[layer] = host.attention(q, k, v, self.cos, self.sin,
                                                   cache[layer])
            residual = host.bf(entry + self.project(
                attended, prefix + SITES[3], step))
            n = self.norm(residual, prefix + "post_attention_layernorm")
            gate, up = [self.project(n, prefix + s, step) for s in SITES[4:6]]
            x = host.bf(residual + self.project(host.silu_mul(gate, up),
                                                prefix + SITES[6], step))
        return self.project(self.norm(x, "model.norm"), "lm_head", step)[
            -1].contiguous(), cache


def native_model(checkpoint, precision):
    from transformers.models.llama import modeling_llama

    config = modeling_llama.LlamaConfig.from_pretrained(str(checkpoint),
                                                       local_files_only=True)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = modeling_llama.LlamaForCausalLM(config)
    model.model.rotary_emb = modeling_llama.LlamaRotaryEmbedding(config=config,
                                                               device="cpu")
    parameters = dict(model.named_parameters())
    with safe_open(str(checkpoint / "model.safetensors"), framework="pt",
                   device="cpu") as handle:
        if set(parameters) != set(handle.keys()):
            raise ValueError("Checkpoint parameters differ from the "
                             "TinyLlama configuration")
        for name, expected in parameters.items():
            value = handle.get_tensor(name).to(precision)
            if value.shape != expected.shape:
                raise ValueError("Checkpoint shape mismatch: " + name)
            parent, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent), attr,
                    torch.nn.Parameter(value, requires_grad=False))
    if precision == torch.float32:
        model.float()
    return model.eval()


def native_forward(model, ids, steps, forced=None):
    from transformers import cache_utils

    cache = cache_utils.DynamicCache(config=model.config)
    tokens, logits = [], []
    for step in range(steps):
        current = ids if step == 0 else [
            tokens[-1] if forced is None else forced[step - 1]]
        output = model(input_ids=torch.tensor([current], dtype=torch.int64),
                       past_key_values=cache, use_cache=True)
        value = output.logits[0, -1].float().contiguous().clone()
        tokens.append(int(value.argmax()))
        logits.append(value)
    return tokens, logits


def run(args):
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    with torch.inference_mode():
        return _run(args)


def _run(args):
    engine = Engine(args)
    if args.inputs:
        data = json.loads(args.inputs.read_text())
        prompts = data["prompts"] if isinstance(data, dict) else data
    else:
        prompts = [{"id": "capture", "token_ids":
                    engine.receipt["target_trace"][0]["input_tokens"][0]}]
    if not prompts:
        raise ValueError("Input prompt set is empty")
    if len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("Prompt IDs must be unique")
    for prompt in prompts:
        ids = prompt["token_ids"]
        if not ids or not all(isinstance(i, int)
                              and 0 <= i < engine.embedding.shape[0]
                              for i in ids):
            raise ValueError("Prompt token_ids must contain valid "
                             "vocabulary IDs")
        if len(ids) + args.steps - 1 > engine.cos.shape[0]:
            raise ValueError("Sequence exceeds the capture rotary table")
    native = {}
    precisions = ("float32", "bfloat16") if args.native_precision == "both" \
        else (() if args.native_precision == "none"
              else (args.native_precision,))
    for precision in precisions:
        model = native_model(args.checkpoint, getattr(torch, precision))
        for index, prompt in enumerate(prompts):
            tokens, logits = native_forward(model, prompt["token_ids"],
                                             args.steps)
            native.setdefault(prompt["id"], {})[precision] = (tokens, logits)
            for step, value in enumerate(logits):
                save_logits(args, index,
                            f"native-{precision}-step-{step}.f32.bin", value)
        del model
        gc.collect()
    result = {"model": "tiny", "precision_mode": args.precision_mode,
              "steps": args.steps, "prompts": [],
              "recipe": {"sites": getattr(args, "sites", "all"),
                         "finer_scales": getattr(args, "finer_scales", False),
                         "activation_scale": getattr(args, "activation_scale",
                                                     None),
                         "weight_scale": getattr(args, "weight_scale", None),
                         "k_chunk": getattr(args, "k_chunk", 128)},
              "torch_version": torch.__version__,
              "arithmetic": "offline CPU emulation; no target timing"}
    capture_gate = (not args.inputs and args.precision_mode == "default"
                    and getattr(args, "activation_scale", None) is None
                    and getattr(args, "weight_scale", None) is None
                    and not getattr(args, "finer_scales", False))
    for index, prompt in enumerate(prompts):
        ids = prompt["token_ids"]
        references = native.get(prompt["id"], {})
        row = {"id": prompt["id"], "tokens": [], "steps": [], "native": {}}
        for precision, (tokens, logits) in references.items():
            row["native"][precision] = {"tokens": tokens,
                                       "steps": [summarize(x) for x in logits]}
        free_cache, forced_cache = ([None] * engine.layers,
                                    [None] * engine.layers)
        reference = references.get("float32", references.get("bfloat16"))
        for step in range(args.steps):
            shared = reference is None or row["tokens"] == reference[0][:step]
            current = ids if step == 0 else [row["tokens"][-1]]
            logits, free_cache = engine.forward(current, list(free_cache), step)
            if shared:
                forced_logits, forced_cache = logits, list(free_cache)
            else:
                forced_logits, forced_cache = engine.forward(
                    [reference[0][step - 1]], list(forced_cache), step)
            item = {"step": step, **summarize(logits)}
            item["logits_file"] = save_logits(
                args, index, f"free-step-{step}.f32.bin", logits)
            row["tokens"].append(item["token"])
            if reference is not None:
                item["forced_native_token"] = int(forced_logits.argmax())
                item["forced_top1_match"] = (item["forced_native_token"]
                                             == reference[0][step])
                item["forced_logit_error"] = metric(forced_logits,
                                                    reference[1][step])
                item["forced_logits_file"] = save_logits(
                    args, index, f"forced-step-{step}.f32.bin", forced_logits)
            if capture_gate and step < len(engine.receipt["target_trace"]):
                target = engine.receipt["target_trace"][step]
                item["capture_exact"] = (
                    item["logits_sha256"] == target["logits"]["sha256"]
                    and item["token"] == target["next_token"])
                if not item["capture_exact"]:
                    raise ValueError("Default Tiny capture mismatch at "
                                     f"step {step}")
            row["steps"].append(item)
        if reference is not None:
            matches = [a == b for a, b in zip(row["tokens"], reference[0])]
            row["matching_prefix"] = next(
                (i for i, x in enumerate(matches) if not x), args.steps)
            row["all_match"] = all(matches)
            row["forced_top1_matches"] = sum(
                x["forced_top1_match"] for x in row["steps"])
        result["prompts"].append(row)
    result["panel_calls"] = engine.panel_calls
    result["capture_exact"] = all(
        x.get("capture_exact", False)
        for p in result["prompts"] for x in p["steps"])
    return result
