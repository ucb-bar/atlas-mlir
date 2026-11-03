"""Pure staging for opt-in Smol BF16 bias, GELU, and connector producer recipes.

No existing exporter or capture imports this module. A descriptor names requested
host work; callers may suppress that work only after successful device completion.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

from atlas_epilogue import Epilogue, Step, TileInput
from atlas_program import assembly, dma_load_bytes, _reuse_module
sys.path.insert(0, str(Path(__file__).absolute().parent / "host_iss"))
from chipyard.arena import relocate_assembly

DRAM_BASE = 0x90000000
GUARD_BYTES = 256
_OPERATIONS = ((), ("bias",), ("bias", "gelu_tanh"), ("connector_scale",))


@dataclass(frozen=True)
class FusionProgram:
    source: str
    arena: bytes
    metadata: dict


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def channel_tiles(bias_bf16, n):
    """Repeat immutable little-endian BF16 channels in native pair-half tiles."""
    if type(n) is not int or n <= 0 or n % 32:
        raise ValueError("channel width must be a positive integer multiple of 32")
    if type(bias_bf16) is not bytes or len(bias_bf16) != 2 * n:
        raise ValueError("bias requires exactly n immutable little-endian BF16 words")
    if any((int.from_bytes(bias_bf16[i:i+2], "little") & 0x7f80) == 0x7f80 for i in range(0, len(bias_bf16), 2)):
        raise ValueError("bias must contain finite BF16 values")
    return b"".join(bias_bf16[column+half:column+half+32] * 32
                    for column in range(0, 2*n, 64) for half in (0, 32))


def stage_panel(row, initial, *, operations=(), bias_bf16=None, weight_source=None):
    """Build a source/arena pair from one baseline capture projection descriptor.

    operations is exactly (), ('bias',), ('bias','gelu_tanh'), or
    ('connector_scale',). Bias/GELU apply only to vision BF16 projections, and
    connector scaling applies only to the bias-free modality projection. The
    baseline descriptor remains unchanged. Later baseline-v1 panels support the
    disabled path exactly; their base recipe is not admitted for fusion here.
    """
    if type(initial) is not bytes or type(operations) is not tuple or operations not in _OPERATIONS:
        raise ValueError("require immutable arena bytes and a supported ordered operation tuple")
    dims = row.get("physical_mkn")
    if (not isinstance(dims, (tuple, list)) or len(dims) != 3
            or any(type(d) is not int or d <= 0 or d % 32 for d in dims)):
        raise ValueError("physical M/K/N must be positive integer multiples of 32")
    m, k, n = dims
    output_offset = m*k + k*n + GUARD_BYTES
    output_bytes = 2*m*n
    arena_bytes = output_offset + output_bytes + GUARD_BYTES
    if row.get("output_offset") != output_offset or len(initial) != arena_bytes or arena_bytes > 64*1024*1024:
        raise ValueError("baseline arena geometry differs")
    for name in ("dram_bytes", "total_dram_bytes"):
        if name in row and row[name] != arena_bytes:
            raise ValueError("baseline descriptor extent differs")
    if (initial[output_offset-GUARD_BYTES:output_offset] != b"\xa5"*GUARD_BYTES
            or initial[output_offset:output_offset+output_bytes] != b"\xcd"*output_bytes
            or initial[-GUARD_BYTES:] != b"Z"*GUARD_BYTES):
        raise ValueError("baseline output poison or guards differ")
    exponent = row.get("output_rescale_exponent")
    if type(exponent) is not int or not -126 <= exponent <= 127:
        raise ValueError("require the captured integer output rescale exponent")
    if "a_scale" in row and "b_scale" in row and row["a_scale"] + row["b_scale"] != exponent:
        raise ValueError("captured A/B scales differ from the output rescale literal")
    recipe = row.get("recipe", row.get("program_policy"))
    if recipe not in ("traffic-reuse-v1", "baseline-v1") or row.get("program_policy", recipe) != recipe:
        raise ValueError("unsupported or inconsistent baseline program policy")
    if row.get("dram_base", DRAM_BASE) != DRAM_BASE:
        raise ValueError("Smol fusion requires the low arena base")
    reuse = recipe == "traffic-reuse-v1"
    resident_b, resident_a = k*n <= 262144, k*n > 262144 and m*k <= 262144
    flags = (dict(traffic_reuse=True) if reuse else
             dict(resident_a=resident_a, resident_b=resident_b, pretranspose_resident_b=resident_b))
    dma_flags = {key:value for key,value in flags.items() if key != "pretranspose_resident_b"}
    baseline_dma = dict(load_bytes=dma_load_bytes(m, k, n, **dma_flags), store_bytes=output_bytes)
    if row.get("requested_dma") != baseline_dma:
        raise ValueError("baseline requested DMA differs")
    if ("bias" in operations) != (bias_bf16 is not None):
        raise ValueError("bias operation and immutable bias bytes must be supplied together")
    epilogue = None
    staged = initial
    inputs = []
    extra = b""
    scratch = None
    if operations:
        if not reuse:
            raise ValueError("fusion requires a qualified traffic-reuse recipe")
        module = row.get("module", "")
        if "bias" in operations and ".vision_model." not in module:
            raise ValueError("bias fusion requires a BF16 vision projection")
        if "gelu_tanh" in operations and not module.endswith(".mlp.fc1"):
            raise ValueError("GELU fusion requires the vision fc1 producer")
        if operations == ("connector_scale",) and not module.endswith(".connector.modality_projection.proj"):
            raise ValueError("connector scaling requires the bias-free modality projection")
        scratch = (27, 28) if _reuse_module(m, k, n) == "atlas_program_reuse_resident" else (10, 11)
        steps = []
        if bias_bf16 is not None:
            tiles = channel_tiles(bias_bf16, n)
            bias_offset = len(initial)
            extra = tiles
            staged += tiles + b"Z"*GUARD_BYTES
            steps.append(Step("add", TileInput(bias_offset, "channel")))
            inputs.append(dict(name="bias", offset=bias_offset, size_bytes=len(tiles), dtype="bf16", layout="channel",
                               scale_exponent=0, channel_bf16_sha256=_sha(bias_bf16), sha256=_sha(tiles)))
        steps += [Step(op) for op in operations if op != "bias"]
        epilogue = Epilogue(m, n, output_offset, tuple(steps), scratch_registers=scratch)
    source = assembly(m, k, n, output_offset, exponent, epilogue=epilogue, weight_source=weight_source, **flags)
    if weight_source is not None:
        from smol_weight_relocate import relocate_assembly as bound_relocate
        source = bound_relocate(source, 0x100000000, DRAM_BASE, weight_source=weight_source)
    else:
        source = relocate_assembly(source, 0x100000000, DRAM_BASE)
    requested = dict(load_bytes=dma_load_bytes(m, k, n, epilogue=epilogue, **dma_flags), store_bytes=output_bytes)
    metadata = dict(schema="smol-bf16-fusion-v1", operations=list(operations), host_ops_on_success=list(operations),
                    panel=row.get("index"), call=row.get("call"), module=row.get("module"),
                    logical_row_start=row.get("logical_row_start"),
                    logical_mkn=list(row["logical_mkn"]) if "logical_mkn" in row else None,
                    physical_mkn=list(dims), recipe=recipe, output_rescale_exponent=exponent,
                    output_offset=output_offset, output_bytes=output_bytes, output_dtype="bf16", output_layout="tiles",
                    dram_base=DRAM_BASE, dram_bytes=len(staged), total_dram_bytes=len(staged),
                    extra_offset=len(initial), extra_bytes=len(extra), extra_sha256=_sha(extra),
                    requested_dma=requested, input_regions=inputs,
                    scratch_registers=list(scratch) if scratch else None, source_sha256=_sha(source.encode()),
                    initial_sha256=_sha(staged), baseline_initial_sha256=_sha(initial))
    return FusionProgram(source, staged, metadata)
