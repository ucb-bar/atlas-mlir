"""BF16 LayerNorm, GELU, negation, and patch convolution from torch.export through model2MLIR, Merlin families, and checked host C."""
import ctypes
import itertools
import shutil
import subprocess
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
import torch
from m2m.ir.import_fx import FXImporter
from merlin.compiler.lower.c.pointwise_host import emit_host_pointwise_c
from merlin.compiler.partition.semantic_regions import extract_semantic_region
from merlin.compiler.quantization.policy_generation import PolicyGenerationError, generate_policy
from merlin.compiler.semantic.linalg.linalg_bridge import translate_linalg_text
from merlin.compiler.semantic.source_preparation import prepare_source

from atlas_mlir_native.compiler_interface import describe_compiler, describe_numerical_families
from atlas_mlir_native.generated_export_policy import EXPORT_FAMILIES
from atlas_mlir_native.reference_eval.evaluator import Tensor
from atlas_mlir_native.reference_eval.host import _composite

_PI = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798")


class Block(torch.nn.Module):
    def __init__(self, width, affine):
        super().__init__()
        self.norm = torch.nn.LayerNorm(width, eps=1e-6, elementwise_affine=affine, dtype=torch.bfloat16)

    def forward(self, x):
        y = self.norm(x)
        return torch.nn.functional.gelu(y, approximate="tanh"), torch.nn.functional.gelu(x)


def _graph(module, example):
    text = str(FXImporter().import_graph(torch.export.export(module, (example,))))
    request = translate_linalg_text(text, entry="forward", target_identity="declared-target",
                                    lowering_policy="diagnostic").request
    return prepare_source(request)[0]


def _compile(tmp_path, region, name):
    record, files = emit_host_pointwise_c(region, name)
    directory = tmp_path / name
    directory.mkdir()
    for filename, text in files.items():
        (directory / filename).write_text(text)
    compiler = shutil.which("cc")
    assert compiler, "generated host C validation requires a local C compiler"
    library = directory / "region.so"
    subprocess.run(["nice", "-n", "10", compiler, "-std=gnu11", "-O2", "-fPIC", "-shared", "-fno-fast-math",
                    "-ffp-contract=off", str(directory / (name + ".c")), "-o", str(library), "-lm"],
                   check=True, capture_output=True)
    function = getattr(ctypes.CDLL(str(library)), name)
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    function.restype = ctypes.c_int
    return record, function


def _run(function, tensors, shape):
    buffers = [t.contiguous().view(torch.int16).clone() for t in tensors]
    output = torch.full(shape, 0x5555, dtype=torch.int16)
    pointers = (ctypes.c_void_p * len(buffers))(*(b.data_ptr() for b in buffers))
    status = function(None, pointers, (ctypes.c_void_p * 1)(output.data_ptr()))
    return status, output.view(torch.bfloat16)


def _tanh(x):
    if abs(x) > 20:
        return Decimal(1 if x > 0 else -1)
    e = (2 * abs(x)).exp()
    return (e - 1) / (e + 1) * (1 if x > 0 else -1)


def _erf(x):
    if abs(x) > 6:
        return Decimal(1 if x > 0 else -1)
    term, total, n = x, x, 0
    while abs(term) > Decimal(10) ** -105 or n < 2:
        n += 1
        term = -term * x * x / n
        total += term / (2 * n + 1)
    return total * 2 / _PI.sqrt()


def _rounded(function, values):
    """Correctly rounded binary32 results, element by element: 110-digit Decimal, then one exact rounding."""
    out = []
    for value in values.flatten().tolist():
        if value == 0:
            out.append(value)
            continue
        with localcontext() as context:
            context.prec = 110
            exact = Fraction(function(Decimal(value)))
        nearest = float(np.float32(float(exact)))
        # Pick the binary32 neighbour closest to the exact value, ties to even.
        candidates = [np.float32(nearest), np.nextafter(np.float32(nearest), np.float32(np.inf)),
                      np.nextafter(np.float32(nearest), np.float32(-np.inf))]
        out.append(float(min(candidates, key=lambda c: (abs(Fraction(float(c)) - exact),
                                                         int(np.array(c).view(np.uint32)) & 1))))
    return torch.tensor(out, dtype=torch.float32).reshape(values.shape)


def _gelu(x, approximate):
    """Source order in torch binary32 elementwise operations; correctly rounded tanh/erf."""
    x = x.float()
    if approximate == "tanh":
        cube = (x * x) * x
        inner = torch.tensor(0.7978845608028654, dtype=torch.float32) * (x + torch.tensor(0.044715, dtype=torch.float32) * cube)
        t = _rounded(_tanh, inner)
    else:
        t = _rounded(_erf, x * torch.tensor(0.7071067811865476, dtype=torch.float32))
    return ((x * 0.5) * (1 + t)).to(torch.bfloat16)


def _eager_layer_norm(x, weight, bias, eps):
    """torch's own eager CPU kernel as the pinned wheel dispatches it on this host."""
    assert torch.backends.cpu.get_cpu_capability() in {"AVX2", "AVX512"}
    return torch.nn.functional.layer_norm(x, x.shape[-1:], weight, bias, eps)


def _bf16_tensor(value):
    return Tensor(value.contiguous().view(torch.int16).numpy().view(np.uint16), "bf16")


@pytest.mark.parametrize("affine", [True, False])
def test_exported_layer_norm_and_gelu_select_host_families_and_match_torch_order(tmp_path, affine):
    torch.manual_seed(20261120)
    shape = (2, 24, 64)
    module = Block(shape[-1], affine)
    if affine:
        with torch.no_grad():
            module.norm.weight.copy_(torch.randn(shape[-1]).to(torch.bfloat16))
            module.norm.bias.copy_(torch.randn(shape[-1]).to(torch.bfloat16))
    # Mean 100 with spread 0.5 makes the moment algorithm visible in the result.
    x = (torch.randn(shape) * 0.5 + 100).to(torch.bfloat16)
    graph = _graph(module, x)
    pure = [node for node in graph.nodes if node.effect == "pure"]
    assert [node.op for node in pure] == ["bf16_layer_norm", "bf16_gelu", "bf16_gelu"]
    policy = generate_policy(graph, describe_compiler(), describe_numerical_families(),
                             original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                             selected_family_ids=EXPORT_FAMILIES)
    assert [row["family"] for row in policy["selected_families"]] == [
        "atlas.host-layer-norm", "atlas.host-gelu", "atlas.host-gelu"]
    assert all(row["precision_change"] is None for row in policy["selected_families"])
    parameters = {node.id: value for node, value in zip(
        (n for n in graph.nodes if n.effect == "input"),
        (*((module.norm.weight, module.norm.bias) if affine else ()), x))}
    norm, tanh_gelu, erf_gelu = pure
    weight, bias = (module.norm.weight.detach(), module.norm.bias.detach()) if affine else (None, None)
    expected_norm = _eager_layer_norm(x, weight, bias, 1e-6)
    reference = _composite("aten.layer_norm.default", [_bf16_tensor(x), [shape[-1]],
                           *((_bf16_tensor(v) for v in (weight, bias)) if affine else (None, None)), 1e-6], {}, {})
    assert np.array_equal(reference.raw, expected_norm.view(torch.int16).numpy().view(np.uint16))
    expected = {norm.id: expected_norm, tanh_gelu.id: _gelu(expected_norm, "tanh"), erf_gelu.id: _gelu(x, "none")}
    values = dict(parameters)
    for node in pure:
        region = extract_semantic_region(graph, (node.id,))
        record, function = _compile(tmp_path, region, "host_" + node.id)
        status, actual = _run(function, [values[key] for key in record["inputs"]], shape)
        assert status == 0
        assert torch.count_nonzero(actual.float()) > actual.numel() // 2
        assert torch.equal(actual.view(torch.int16), expected[node.id].view(torch.int16)), node.op
        values[node.id] = actual
        # A non-finite input refuses before any store.
        poisoned = values[record["inputs"][0]].clone()
        poisoned.view(-1)[5] = float("inf")
        status, untouched = _run(function, [poisoned, *(values[key] for key in record["inputs"][1:])], shape)
        assert status == -2 and torch.all(untouched.view(torch.int16) == 0x5555)


@pytest.mark.parametrize("width", [1100, 4097])
def test_wide_layer_norm_matches_torch_eager_through_tail_and_cascade(tmp_path, width):
    """Mean 100, spread 0.5: 1,100 and 4,097 columns reach cascade depths 3 and 4, and 4,097 has a tail."""
    torch.manual_seed(width)
    norm = torch.nn.LayerNorm(width, eps=1e-6, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width).to(torch.bfloat16))
        norm.bias.copy_(torch.randn(width).to(torch.bfloat16))
    x = (torch.randn(3, width) * 0.5 + 100).to(torch.bfloat16)
    graph = _graph(norm, x)
    node, = (n for n in graph.nodes if n.effect == "pure")
    assert node.op == "bf16_layer_norm"
    expected = _eager_layer_norm(x, norm.weight.detach(), norm.bias.detach(), 1e-6)
    reference = _composite("aten.layer_norm.default", [_bf16_tensor(x), [width], _bf16_tensor(norm.weight.detach()),
                           _bf16_tensor(norm.bias.detach()), 1e-6], {}, {})
    assert np.array_equal(reference.raw, expected.view(torch.int16).numpy().view(np.uint16))
    record, function = _compile(tmp_path, extract_semantic_region(graph, (node.id,)), "wide_norm")
    values = {"arg0": norm.weight.detach(), "arg1": norm.bias.detach(), "arg2": x}
    status, actual = _run(function, [values[key] for key in record["inputs"]], x.shape)
    assert status == 0 and torch.count_nonzero(actual.float()) == actual.numel()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("shape,k,roles", [((4, 8), 1, ()), ((4, 8), 1, ("weight", "bias")), ((1, 8), 1, ()),
                                           ((2, 4, 4), 2, ("weight",)), ((2, 4, 4), 1, ())])
def test_layer_norm_window_splits_keep_whole_rows(tmp_path, shape, k, roles):
    """No planned piece normalizes part of a row, and whole-row pieces reproduce the unsplit output."""
    from math import prod

    from merlin.compiler.build import staging
    from merlin.compiler.semantic.linalg.source_compositions import row_reduction_maps
    from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType

    rank, rows = len(shape), prod(shape[:-k])
    typ = TensorType(shape, "bf16", "raw-bit-pattern")
    terms = tuple(SemanticNode(role, "input", (), TensorType(shape[-k:], "bf16", "raw-bit-pattern"), effect="input")
                  for role in roles)
    node = SemanticNode("r", "bf16_layer_norm", ("x", *roles), typ,
                        attrs=(("affine_operands", ",".join(roles)), ("epsilon_f32_bits", 0x358637BD),
                               ("input_domain", "finite-bf16"), ("normalized_rank", k),
                               ("numeric_semantics", "torch-2.11-cpu-fma-bf16-layer-norm-v1")),
                        index_maps=row_reduction_maps(rank, k, len(roles)))
    inputs = (SemanticNode("x", "input", (), typ, effect="input"), *terms)
    region = KernelRequest((*inputs, node), ("r",), ("ram",), tuple((item.id, "ram") for item in inputs), "fixture")
    assert staging.split_rule(region) is None
    assert not {split.axis for split in staging.band_rules(region)} & set(range(rank - k, rank))
    total = prod(shape)
    admitted = [count for count in range(2, total + 1) if not total % count and staging.leading_rows(region, count)]
    assert all(not rows % count for count in admitted)
    torch.manual_seed(rows + k)
    x = torch.randn(shape).to(torch.bfloat16)
    values = {"x": x, **{role: torch.randn(shape[-k:]).to(torch.bfloat16) for role in roles}}
    record, function = _compile(tmp_path, region, "whole_norm")
    status, whole = _run(function, [values[key] for key in record["inputs"]], shape)
    assert status == 0
    if rows == 1:
        assert not admitted
        return
    assert admitted == [count for count in range(2, rows + 1) if not rows % count]
    assert staging.leading_rows(region, rows) == {"r": "split", "x": "split", **{role: "whole" for role in roles}}
    piece = staging.row_slice(region, rows)
    record, function = _compile(tmp_path, piece, "row_norm")
    flat = x.reshape(rows, *shape[-k:])
    parts = []
    for row in flat:
        status, part = _run(function, [row[None] if key == "x" else values[key] for key in record["inputs"]],
                            (1, *shape[-k:]) if rank - k == 1 else (*(1,) * (rank - k), *shape[-k:]))
        assert status == 0
        parts.append(part.reshape(shape[-k:]))
    assert torch.equal(torch.stack(parts).view(torch.int16).reshape(shape), whole.view(torch.int16))


# A width-264, mean-1,000 row whose output fusing the AddMomentsVec m1 or tail m2 update changes (BF16, big-endian hex).
_REGRESSION_ROW = bytes.fromhex(
    "447C447B447A447A447B447A4479447B447B447A447A447A447A447A447A44794478447B4479447B447944794478447A"
    "447A447B447A447B447B447A447A4479447C447A447B447A447A447A447A4479447A447A447C447A447A447A447A447B"
    "44794479447B447B447A4479447A447A447A447A4479447C4478447C44794478447A44794479447C447B447944774479"
    "447A447A4479447B447B447B447A447A447A447A447A447B447A447A4478447A4479447A4479447C447A447944774479"
    "447A447B447A447C447C4479447B447A4479447C447C447A4479447B447B447B447A44794479447A447C447D447A447A"
    "447B447A44794479447A447A44784479447B447A447A447A447A447A447B44794478447B447B447A447A447A447A447A"
    "447B447944794479447A4478447A447A447B447A447A447A4479447A447B447B4478447B447B447A447B4479447C447B"
    "4479447A447B447B4479447A447B4478447A447B447A447A447A4479447A447A447A447A447B447A44794479447B4479"
    "4478447B447A447A447A447A447B447A447A447A4479447A447A4479447A447B447A447B447A447A4479447A447B447A"
    "447A447A447B447A4479447B44794479447B447A447A447C447A4479447B447B447B447B447B447B4477447B447B4479"
    "447A447A447A447A447B44794479447B4478447B447944784479447B44794478447944794479447A447A4479447C447B")


def _many_row_cases():
    """(seed, rows, widths, (mean, spread) pairs, affine): each set exposes moment or output sites the others miss."""
    return ((11, 6000, (264, 520, 1100, 4104), ((1000, 4),), False),  # one to 256 vectors, cascade depths 0 to 4
            (12, 4000, (24, 40), ((100, 0.5), (1000, 2)), False),  # tail-dominated rows: the scalar AddMoments steps
            (13, 2000, (2000,), ((100, 0.5),), False),  # a partial last chunk: a combine weight that is not a power of 2
            (14, 2000, (1100,), ((1000, 4), (100, 0.5)), True))  # random BF16 weight and bias: the outer output step


def test_layer_norm_matches_torch_eager_on_many_rows(tmp_path):
    """The reference and emitted C equal eager on the regression row; the C on 14,000 rows of four sets, the reference on 500."""
    row = torch.from_numpy(np.frombuffer(_REGRESSION_ROW, ">u2").astype(np.int16)).view(torch.bfloat16).reshape(1, -1)
    expected = _eager_layer_norm(row, None, None, 1e-6)
    reference = _composite("aten.layer_norm.default", [_bf16_tensor(row), [row.shape[1]], None, None, 1e-6], {}, {})
    assert np.array_equal(reference.raw, expected.view(torch.int16).numpy().view(np.uint16))
    functions = {}
    for _, _, widths, _, affine in _many_row_cases():
        for width in widths:
            norm = torch.nn.LayerNorm(width, eps=1e-6, elementwise_affine=affine, dtype=torch.bfloat16)
            graph = _graph(norm, torch.zeros(1, width, dtype=torch.bfloat16))
            node, = (n for n in graph.nodes if n.effect == "pure")
            functions[width, affine] = _compile(tmp_path, extract_semantic_region(graph, (node.id,)),
                                                f"norm_{width}_{int(affine)}")
    record, function = functions[264, False]
    status, actual = _run(function, [row], row.shape)
    assert status == 0 and torch.count_nonzero(actual.float()) == actual.numel()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    for seed, count, widths, shapes, affine in _many_row_cases():
        torch.manual_seed(seed)
        differing = 0
        for index in range(count):
            width = widths[index % len(widths)]
            mean, spread = shapes[index // len(widths) % len(shapes)]
            x = (torch.randn(1, width) * spread + mean).to(torch.bfloat16)
            weight, bias = (torch.randn(width).to(torch.bfloat16) for _ in range(2)) if affine else (None, None)
            record, function = functions[width, affine]
            values = {"arg0": weight, "arg1": bias, "arg2": x} if affine else {"arg0": x}
            status, actual = _run(function, [values[key] for key in record["inputs"]], x.shape)
            assert status == 0
            expected = _eager_layer_norm(x, weight, bias, 1e-6).view(torch.int16)
            differing += int((actual.view(torch.int16) != expected).sum())
            if affine and index < 500:  # the reference's outer output step
                reference = _composite("aten.layer_norm.default", [_bf16_tensor(x), [width], _bf16_tensor(weight),
                                       _bf16_tensor(bias), 1e-6], {}, {})
                differing += int((reference.raw != expected.numpy().view(np.uint16)).sum())
        assert differing == 0, seed


def test_layer_norm_row_moments_match_torch_eager_binary32_statistics():
    """BF16 input with binary32 unit affine makes eager return its binary32 mean and rstd, which expose every moment site."""
    from atlas_mlir_native.reference_eval.host import _torch_bf16_row_moments

    torch.manual_seed(1000)
    rows = 0
    # Mean 1,000 exposes the Welford and scalar AddMoments steps; mean 3, spread 7 the AddMomentsVec combinations.
    for (mean_in, spread), width in itertools.product(((1000, 4), (3, 7)), (24, 40, 264, 520, 1100, 1360, 2000, 3000, 4104)):
        x = (torch.randn(32, width) * spread + mean_in).to(torch.bfloat16)
        assert torch.backends.cpu.get_cpu_capability() in {"AVX2", "AVX512"}
        _, mean, rstd = torch.native_layer_norm(x, (width,), torch.ones(width), torch.zeros(width), 1e-6)
        assert mean.dtype == rstd.dtype == torch.float32
        for i, row in enumerate(x.float().numpy()):
            m1, variance = _torch_bf16_row_moments(row)
            inverse = np.float32(np.float32(1) / np.sqrt(np.float32(variance + np.float32(1e-6))))
            assert np.float32(m1).view(np.uint32) == mean[i, 0].numpy().view(np.uint32), (mean_in, width, i)
            assert inverse.view(np.uint32) == rstd[i, 0].numpy().view(np.uint32), (mean_in, width, i)
            rows += 1
    assert rows == 576


def test_unsupported_variants_have_no_family():
    """F32 sources and BF16 sources without one final BF16 rounding keep ordinary refused nodes."""

    class Float32Norm(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.layer_norm(value, (8,), eps=1e-5)

    class Float32Gelu(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.gelu(value, approximate="tanh")

    class WideNorm(torch.nn.Module):
        def forward(self, value):
            return torch.nn.functional.layer_norm(value.float(), (8,), eps=1e-5)

    cases = ((Float32Norm(), torch.float32), (Float32Gelu(), torch.float32), (WideNorm(), torch.bfloat16))
    for module, dtype in cases:
        graph = _graph(module, torch.randn(2, 3, 8).to(dtype))
        assert not {"bf16_layer_norm", "bf16_gelu"} & {node.op for node in graph.nodes}
        with pytest.raises(PolicyGenerationError):
            generate_policy(graph, describe_compiler(), describe_numerical_families(),
                            original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                            selected_family_ids=EXPORT_FAMILIES)


def test_exported_select_chunk_and_negation_use_layout_and_sign_flip_families(tmp_path):
    class Rotate(torch.nn.Module):
        def forward(self, value):
            first, second = value[:, 1].chunk(2, dim=-1)
            return torch.cat((-second, first), dim=-1)

    x = torch.randn(2, 3, 8).to(torch.bfloat16)
    graph = _graph(Rotate(), x)
    policy = generate_policy(graph, describe_compiler(), describe_numerical_families(),
                             original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                             selected_family_ids=EXPORT_FAMILIES)
    families = {graph.node(row["node"]).op: row["family"] for row in policy["selected_families"]}
    assert families.pop("bf16_neg") == "atlas.host-sign-flip"
    assert families and set(families.values()) == {"atlas.layout"}
    negation = next(node for node in graph.nodes if node.op == "bf16_neg")
    region = extract_semantic_region(graph, (negation.id,))
    record, function = _compile(tmp_path, region, "negation")
    value = x[:, 1, 4:].contiguous()
    status, actual = _run(function, [value], value.shape)
    assert status == 0 and torch.equal(actual.view(torch.int16), (-value).view(torch.int16))


@pytest.mark.parametrize("geometry", [
    dict(kernel_size=4, stride=4), dict(kernel_size=3, stride=1), dict(kernel_size=3, dilation=2),
    dict(kernel_size=(2, 3), stride=(2, 1)), dict(kernel_size=2, stride=2, bias=False)])
def test_exported_convolution_selects_host_contraction_and_matches_ascending_k_order(tmp_path, geometry):
    torch.manual_seed(7)
    conv = torch.nn.Conv2d(3, 16, dtype=torch.bfloat16, **geometry)
    x = torch.randn(1, 3, 16, 16).to(torch.bfloat16)
    graph = _graph(conv, x)
    node, = (n for n in graph.nodes if n.effect == "pure")
    assert node.op == "bf16_affine_contraction"
    policy = generate_policy(graph, describe_compiler(), describe_numerical_families(),
                             original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                             selected_family_ids=EXPORT_FAMILIES)
    assert [row["family"] for row in policy["selected_families"]] == ["atlas.host-bf16-contraction"]
    region = extract_semantic_region(graph, (node.id,))
    record, function = _compile(tmp_path, region, "patch")
    values = {"arg0": conv.weight, "arg1": conv.bias, "arg2": x} if conv.bias is not None else {"arg0": conv.weight, "arg1": x}
    shape = tuple(conv(x).shape)
    status, actual = _run(function, [values[key] for key in record["inputs"]], shape)
    assert status == 0
    # Reference: unfold with torch, then binary32 products and sums in ascending K order.
    columns = torch.nn.functional.unfold(x.float(), conv.kernel_size, dilation=conv.dilation, stride=conv.stride)[0]
    weights = conv.weight.float().reshape(16, -1)
    total = torch.zeros(16, columns.shape[1])
    for k in range(columns.shape[0]):
        total = total + weights[:, k:k + 1] * columns[k:k + 1]
    if conv.bias is not None:
        total = total + conv.bias.float()[:, None]
    expected = total.to(torch.bfloat16).reshape(shape)
    assert torch.count_nonzero(expected.float()) == expected.numel()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_unadmitted_convolutions_stay_decomposed_and_have_no_family():
    """Batch two, padding, and groups have no row-major unfolded form; policy generation refuses them."""
    cases = ((torch.nn.Conv2d(3, 16, 4, stride=4, dtype=torch.bfloat16), (2, 3, 16, 16)),
             (torch.nn.Conv2d(3, 16, 3, padding=1, dtype=torch.bfloat16), (1, 3, 16, 16)),
             (torch.nn.Conv2d(3, 6, 3, groups=3, dtype=torch.bfloat16), (1, 3, 16, 16)))
    for conv, shape in cases:
        graph = _graph(conv, torch.randn(shape).to(torch.bfloat16))
        assert "bf16_affine_contraction" not in {n.op for n in graph.nodes}
        with pytest.raises(PolicyGenerationError):
            generate_policy(graph, describe_compiler(), describe_numerical_families(),
                            original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                            selected_family_ids=EXPORT_FAMILIES)


def _f32_region(operation, shape):
    from merlin.compiler.semantic.linalg.linalg_recognition import _f32_reduce_maps
    from merlin.compiler.semantic.model import ConstantBinding, KernelRequest, SemanticNode, TensorType, pointwise_identity_maps

    typ = TensorType(shape, "f32", "raw-bit-pattern")
    x = SemanticNode("x", "input", (), typ, effect="input")
    rank, constants = len(shape), ()
    if operation == "sum":
        out = TensorType(shape[:-1], "f32", "raw-bit-pattern")
        seed = SemanticNode("seed", "constant", (), out, effect="constant")
        node = SemanticNode("r", "source_f32_reduce_add_unqualified", ("x", "seed"), out,
                            attrs=(("axis", rank - 1), ("reduction_order", "unresolved"),
                                   ("numeric_semantics", "source-f32-add-unqualified-v1"),
                                   ("scalar_add_order", "input-accumulator"), ("source_operation", "aten.mean.dim")),
                            index_maps=_f32_reduce_maps(rank, (rank - 1,)))
        constants = (ConstantBinding("seed", "f32-le-raw", (b"\0" * (4 * int(np.prod(shape[:-1])))).hex()),)
        nodes = (x, seed, node)
    else:
        op, attrs = (("f32_silu", (("input_domain", "finite-f32"), ("numeric_semantics", "torch-2.11-cpu-f32-silu-v1")))
                     if operation == "silu" else
                     ("source_f32_rsqrt_unqualified", (("numeric_semantics", "source-f32-rsqrt-unqualified-v1"),
                                                       ("source_operation", "aten.rsqrt.default"))))
        node = SemanticNode("r", op, ("x",), typ, attrs=attrs, index_maps=pointwise_identity_maps(rank, 1))
        nodes = (x, node)
    return KernelRequest(nodes, ("r",), ("ram",), (("x", "ram"),), "fixture", constants=constants)


def _f32_values(operation, shape, seed):
    rng = np.random.default_rng([20261121, seed, *shape])
    if operation == "rsqrt":
        return (np.abs(rng.standard_normal(shape)) * 10.0 ** rng.uniform(-30, 30, shape)).astype(np.float32)
    if operation == "sum":
        return (rng.standard_normal(shape) * 10.0 ** rng.uniform(-3, 3, shape)).astype(np.float32)
    if shape[-1] == 1 << 20:
        # Where the exponential dominates the denominator, so its last bit reaches the quotient.
        return rng.uniform(-30, 5, shape).astype(np.float32)
    # Both exponential branches, the rounding-sensitive middle, and wide magnitudes.
    return (rng.standard_normal(shape) * rng.choice([1e-3, 1.0, 8.0, 40.0, 120.0], shape)).astype(np.float32)


@pytest.mark.parametrize("operation,shape", [
    ("silu", (1, 50, 720)), ("silu", (1, 50, 2048)), ("silu", (1, 4416)), ("silu", (1, 1 << 20)), ("rsqrt", (1, 50, 1)), ("rsqrt", (3, 64)),
    ("sum", (1, 50, 720)), ("sum", (1, 50, 2048)), ("sum", (1, 4416)), ("sum", (1, 15, 227)), ("sum", (3, 7)),
    ("sum", (4, 8)), ("sum", (4, 13)), ("sum", (2, 1025)), ("sum", (2, 4097)), ("sum", (2, 70000)),
])
def test_f32_host_families_match_torch_eager(tmp_path, operation, shape):
    """Emitted C and the independent reference equal unforced torch eager bit for bit on nonzero data."""
    from atlas_mlir_native.reference_eval import host

    assert torch.backends.cpu.get_cpu_capability() in {"AVX2", "AVX512"}
    record, function = _compile(tmp_path, _f32_region(operation, shape), "f32_" + operation)
    assert record["inputs"] == ["x"]
    out_shape = shape[:-1] if operation == "sum" else shape
    eager = {"silu": torch.nn.functional.silu, "rsqrt": torch.rsqrt, "sum": lambda t: t.sum(-1)}[operation]
    rules = {"host_order": {"default": "lexicographic-f32", "aten.sum.dim_IntList": host.TORCH_F32_SUM}}

    def call(values):
        output = np.full(out_shape, 0x55555555, np.uint32)
        status = function(None, (ctypes.c_void_p * 1)(values.ctypes.data_as(ctypes.c_void_p)),
                          (ctypes.c_void_p * 1)(output.ctypes.data_as(ctypes.c_void_p)))
        return status, output

    for seed in range(6):
        values = _f32_values(operation, shape, seed)
        expected = eager(torch.from_numpy(values.copy())).numpy().view(np.uint32)
        assert np.count_nonzero(expected & 0x7fffffff) > expected.size // 2
        status, actual = call(values)
        assert status == 0 and np.array_equal(actual, expected)
        tensor = Tensor(values.view(np.uint32).copy(), "f32")
        if operation == "sum":
            reference = host.call("aten.sum.dim_IntList", [tensor, [-1], False], {}, rules)
        else:
            reference = host.call(f"aten.{operation}.default", [tensor], {}, {})
        assert np.array_equal(reference.raw, expected)
    poisoned = values.copy()
    poisoned.flat[poisoned.size // 2] = np.inf
    status, untouched = call(poisoned)
    assert status == -2 and np.all(untouched == 0x55555555)


@pytest.mark.parametrize("shape,admitted", [((1, 32767), True), ((1, 32768), False), ((1, 40000), False),
                                            ((1, 1, 70000), False), ((2, 70000), True)])
def test_f32_sum_refuses_one_row_past_the_thread_split(tmp_path, shape, admitted):
    """The tag, the emitter, and the reference share the bound; admitted shapes equal eager at 1, 2, and 8 threads."""
    from dataclasses import replace

    from merlin.compiler.semantic.linalg.source_compositions import collapse_source_compositions
    from atlas_mlir_native.reference_eval import host

    region = _f32_region("sum", shape)
    bare = tuple(replace(node, attrs=tuple(row for row in node.attrs if row[0] != "source_operation"))
                 if node.id == "r" else node for node in region.nodes)
    tagged = {node.id: node for node in collapse_source_compositions(bare, ("r",), region.constants,
                                                                     sources={"r": "aten.mean.dim"})[0]}["r"]
    assert ("source_operation" in dict(tagged.attrs)) is admitted
    values = _f32_values("sum", shape, 0)
    tensor = Tensor(values.view(np.uint32).copy(), "f32")
    rules = {"host_order": {"default": "lexicographic-f32", "aten.mean.dim": host.TORCH_F32_SUM}}
    if not admitted:
        with pytest.raises(ValueError, match="32,768"):
            _compile(tmp_path, region, "f32_sum")
        with pytest.raises(ValueError, match="32,768"):
            host.call("aten.mean.dim", [tensor, [-1], False], {}, rules)
        return
    reference = host.call("aten.mean.dim", [tensor, [-1], False], {}, rules)
    threads = torch.get_num_threads()
    try:
        for count in (1, 2, 8):
            torch.set_num_threads(count)
            assert np.array_equal(reference.raw, torch.from_numpy(values.copy()).mean(-1).numpy().view(np.uint32))
    finally:
        torch.set_num_threads(threads)


def test_f32_sum_is_exact_only_against_its_declared_order():
    """A transposed operand keeps its strides in eager, whose mean then leaves the declared contiguous order."""
    from atlas_mlir_native.reference_eval import host

    shape = (4, 50, 960)
    values = _f32_values("sum", shape, 0)
    rules = {"host_order": {"default": "lexicographic-f32", "aten.mean.dim": host.TORCH_F32_SUM}}
    declared = host.call("aten.mean.dim", [Tensor(values.view(np.uint32).copy(), "f32"), [-1], False], {}, rules).raw
    contiguous = torch.from_numpy(values.copy())
    transposed = contiguous.transpose(1, 2).contiguous().transpose(1, 2)
    assert torch.equal(transposed, contiguous) and not transposed.is_contiguous()
    assert np.array_equal(contiguous.mean(-1).numpy().view(np.uint32), declared)
    assert np.count_nonzero(transposed.mean(-1).numpy().view(np.uint32) != declared) > 0


def _rounded_exp(x):
    """Correctly rounded binary32 exp: binary64 exp where it decides the rounding, else 60-digit Decimal."""
    wide = np.exp(x.astype(np.float64))
    result = wide.astype(np.float32)
    below, above = (np.nextafter(result, np.float32(limit)).astype(np.float64) for limit in (-np.inf, np.inf))
    midpoints = np.stack([(result + below) / 2, (result + above) / 2])
    near = np.isfinite(wide) & (np.abs(midpoints - wide).min(axis=0) <= np.abs(wide) * 2.0 ** -40)
    with localcontext() as context:
        context.prec = 60
        for index in zip(*np.nonzero(near)):
            exact = Decimal(float(x[index])).exp()
            low, high = (Decimal(float(m)) for m in midpoints[(slice(None), *index)])
            result[index] = below[index] if exact < low else above[index] if exact > high else result[index]
    return result


def _declared_softmax(values):
    """host-softmax-f32-left-fold-div-rne-v1 over the last axis, one binary32 operation at a time."""
    rows = values.reshape(-1, values.shape[-1])
    maximum = np.full(rows.shape[0], -np.inf, np.float32)
    for column in rows.T:
        maximum = np.where(column > maximum, column, maximum)
    e = _rounded_exp(np.subtract(rows, maximum[:, None]))
    total = np.zeros(rows.shape[0], np.float32)
    for column in e.T:
        total = np.add(total, column)
    return np.divide(e, total[:, None]).reshape(values.shape)


@pytest.mark.parametrize("shape", [(1, 15, 50, 227), (1, 15, 50, 177), (1, 8, 24, 24), (3, 1), (2, 1025)])
def test_f32_softmax_matches_its_declared_order(tmp_path, shape):
    """Emitted C and the reference evaluator equal an independent left-fold softmax with a proven rounded exp."""
    import json
    import os
    from pathlib import Path

    from merlin.compiler.build import staging
    from merlin.compiler.semantic.linalg.source_compositions import softmax_maps
    from merlin.compiler.semantic.model import KernelRequest, SemanticNode, TensorType
    from atlas_mlir_native.reference_eval import host

    rules = json.loads(Path(os.environ["ATLAS_REFERENCE_TEST_RULES"]).read_text())
    typ = TensorType(shape, "f32", "raw-bit-pattern")
    node = SemanticNode("r", "source_f32_softmax_unqualified", ("x",), typ,
                        attrs=(("axis", len(shape) - 1), ("numeric_semantics", "source-f32-softmax-unqualified-v1"),
                               ("reduction_order", "unresolved")), index_maps=softmax_maps(len(shape)))
    region = KernelRequest((SemanticNode("x", "input", (), typ, effect="input"), node), ("r",), ("ram",),
                           (("x", "ram"),), "fixture")
    # A window split may cut whole rows, never the reduced axis.
    assert staging.split_rule(region) is None
    assert all(split.axis != len(shape) - 1 for split in staging.band_rules(region))
    read = softmax_maps(len(shape))[0]
    assert [read.apply((*(0,) * len(shape), k))[-1] for k in range(shape[-1])] == list(range(shape[-1]))
    rows, total = int(np.prod(shape[:-1])), int(np.prod(shape))
    admitted = [count for count in range(2, total + 1) if not total % count and staging.leading_rows(region, count)]
    # Every admitted count cuts between rows; which divisors are reachable depends on the leading dimensions.
    assert all(not rows % count for count in admitted) and bool(admitted) == (rows > 1)
    if shape[0] > 1:
        assert staging.leading_rows(region, shape[0]) == {"r": "split", "x": "split"}
    record, function = _compile(tmp_path, region, "f32_softmax")
    assert record["numeric_semantics"] == "host-softmax-f32-left-fold-div-rne-v1"
    assert record["host_math_symbols"] == ["merlin_host_expf"]
    for seed in range(3):
        rng = np.random.default_rng([20261121, seed, *shape])
        values = (rng.standard_normal(shape) * rng.choice([1e-3, 1.0, 8.0, 30.0], shape)).astype(np.float32)
        # Additive-mask magnitudes: the most negative BF16 value and -inf, never a whole row.
        masked = rng.random(shape) < 0.2
        masked[..., 0] = False
        values[masked] = rng.choice(np.float32([-3.3895313892515355e38, -np.inf]), np.count_nonzero(masked))
        expected = _declared_softmax(values).view(np.uint32)
        assert np.count_nonzero(expected) > expected.size // 2
        output = np.full(shape, 0x55555555, np.uint32)
        status = function(None, (ctypes.c_void_p * 1)(values.ctypes.data_as(ctypes.c_void_p)),
                          (ctypes.c_void_p * 1)(output.ctypes.data_as(ctypes.c_void_p)))
        assert status == 0 and np.array_equal(output, expected)
        reference = host.call("aten._softmax.default", [Tensor(values.view(np.uint32).copy(), "f32"), -1, False],
                              {}, rules)
        assert np.array_equal(reference.raw, expected)

def _captured_graph(tmp_path, module, example):
    from m2m.capture.bundle import write_bundle

    write_bundle(module.eval(), (example,), tmp_path, capture_regions=False, capture_trace=True)
    text = (tmp_path / "model.mlir").read_text()
    return prepare_source(translate_linalg_text(text, entry="forward", target_identity="declared-target",
                                                lowering_policy="diagnostic").request)[0]


class _Source(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, x):
        return self.function(x)


@pytest.mark.parametrize("form,ops,families", [
    ("silu", ["f32_silu"], ["atlas.host-f32-silu"]),
    ("bf16-silu", ["f32_to_bf16", "bf16_to_f32", "f32_silu", "f32_to_bf16"], None),
    ("x-times-sigmoid", ["source_f32_composite_unqualified", "f32_mul"], None),
    ("rms", None, None),
    ("softmax", ["source_f32_softmax_unqualified"], ["atlas.host-f32-softmax"]),
    ("scaled-masked-softmax", ["f32_mul", "gather", "f32_add", "source_f32_softmax_unqualified"],
     ["atlas.host-f32-mul", "atlas.layout", "atlas.host-f32-add", "atlas.host-f32-softmax"]),
    ("hand-softmax", None, None),
    ("first-axis-softmax", None, None),
    ("f32-layer-norm", None, None),
])
def test_captured_f32_forms_select_only_their_family(tmp_path, form, ops, families):
    function = {
        "silu": torch.nn.functional.silu,
        "bf16-silu": lambda x: torch.nn.functional.silu(x.to(torch.bfloat16)),
        "x-times-sigmoid": lambda x: x * torch.sigmoid(x),
        "rms": lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5),
        "softmax": lambda x: torch.softmax(x, -1),
        "scaled-masked-softmax": lambda x: torch.softmax(x * 0.125 + x[..., :1], -1),
        "hand-softmax": lambda x: (lambda e: e / e.sum(-1, keepdim=True))((x - x.amax(-1, keepdim=True)).exp()),
        "first-axis-softmax": lambda x: torch.softmax(x, 0),
        "f32-layer-norm": lambda x: torch.nn.functional.layer_norm(x, (24,), eps=1e-5),
    }[form]
    graph = _captured_graph(tmp_path, _Source(function), torch.randn(2, 24))
    pure = [node.op for node in graph.nodes if node.effect == "pure"]
    if ops is not None:
        assert pure == ops
    def select():
        return generate_policy(graph, describe_compiler(), describe_numerical_families(),
                               original_source_sha256="a" * 64, preparation_sha256="b" * 64,
                               selected_family_ids=EXPORT_FAMILIES)
    if form in {"x-times-sigmoid", "hand-softmax", "first-axis-softmax", "f32-layer-norm"}:
        # A real product with a sigmoid keeps its own rounding; only a source softmax over the last axis is one
        # node, and f32 layer norm has another kernel.
        with pytest.raises(PolicyGenerationError):
            select()
        if form != "x-times-sigmoid":
            from merlin.compiler.quantization import policy_generation as pg

            rows = [row for row in describe_numerical_families() if row.family == "atlas.host-f32-sum"]
            sums = [item for item in pg.graph_requirements(graph)
                    if item["operation"] == "source_f32_reduce_add_unqualified"]
            assert sums and not any(pg._matches(item, rows[0]) for item in sums)
            if form == "f32-layer-norm":
                roots = [row for row in describe_numerical_families() if row.family == "atlas.host-f32-rsqrt"]
                inverses = [item for item in pg.graph_requirements(graph)
                            if item["operation"] == "source_f32_rsqrt_unqualified"]
                assert inverses and not any(pg._matches(item, roots[0]) for item in inverses)
        return
    selected = [row["family"] for row in select()["selected_families"]]
    assert all(row is None for row in (r["precision_change"] for r in select()["selected_families"]))
    if families is not None:
        assert selected == families
    if form == "bf16-silu":
        assert "atlas.host-f32-silu" in selected
    if form == "rms":
        assert {"atlas.host-f32-sum", "atlas.host-f32-rsqrt"} <= set(selected)
