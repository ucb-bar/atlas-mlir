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
