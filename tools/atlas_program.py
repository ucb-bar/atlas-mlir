"""Bounded contraction assembly and source-bundle admission."""

from pathlib import Path
from atlas_program_checks import checked_assembly
from atlas_path_guard import checked as safe_path
from atlas_tooling import sha256
from run_model_tile import validate_archive_members
import zipfile


_TRAFFIC_REUSE_MODULES = {
    shape: "atlas_program_reuse_" + recipe
    for recipe, shapes in (
        ("resident", ((128, 768, 768),)),
        ("tiny", (
            (128, 2048, 2048), (128, 2048, 1536), (128, 2048, 256), (128, 2048, 1280),
        )),
        ("stream", ((128, 3072, 768),)),
        ("smol_connector", ((64, 12288, 960),)),
        ("tiny_breadth", (
            (128, 5632, 1024), (32, 2048, 2048), (32, 2048, 1024), (32, 5632, 1024), (32, 2048, 1536),
            (32, 2048, 256), (32, 2048, 1280),
        )),
        ("blocked", (
            (224, 2304, 256), (64, 4608, 512), (800, 1152, 128), (224, 256, 1024), (3136, 576, 64),
            (3136, 64, 256), (224, 1024, 256), (800, 128, 512), (64, 512, 2048), (800, 512, 128), (12544, 160, 64),
            (64, 2048, 512), (64, 1024, 2048), (224, 512, 1024), (224, 1024, 512), (800, 256, 512),
            (3136, 256, 64), (3136, 256, 128), (800, 512, 256), (3136, 64, 64), (64, 320, 320), (128, 320, 320),
            (64, 32, 736), (32, 32, 960), (128, 768, 3072), (64, 736, 2048), (64, 2048, 736), (128, 960, 2560),
            (128, 2560, 960), (64, 960, 736), (64, 736, 960), (64, 736, 320), (64, 960, 2560), (128, 960, 960),
            (64, 2560, 960), (64, 960, 960), (128, 960, 320), (64, 1440, 736), (64, 960, 320), (64, 736, 736),
            (192, 960, 320), (192, 960, 960), (192, 960, 2560), (192, 2560, 960),
        )),
    )
    for shape in shapes
}


def _reuse_enabled(value):
    if type(value) is not bool:
        raise ValueError("traffic_reuse must be a boolean")
    return value


def _reuse_module(m, k, n, *, resident_a=False, resident_b=False, pretranspose_resident_b=False):
    if any(type(d) is not int or d <= 0 or d % 32 for d in (m, k, n)):
        raise ValueError("traffic reuse requires positive integer dimensions divisible by 32")
    if any(flag is not False for flag in (resident_a, resident_b, pretranspose_resident_b)):
        raise ValueError("traffic reuse excludes legacy residency and pretranspose flags")
    try:
        return _TRAFFIC_REUSE_MODULES[(m, k, n)]
    except KeyError:
        raise ValueError("traffic reuse is supported only for the qualified shapes") from None


def dma_load_bytes(m, k, n, *, resident_a=False, resident_b=False, traffic_reuse=False, epilogue=None, direct_input=None):
    """Requested FP8 input bytes for a complete tiled contraction.

    m, k and n must be positive physical dimensions divisible by 32. Callers
    enforce this precondition and resident bank capacity before accounting.
    """
    if direct_input is not None:
        direct_input.validate(m, k)
        if resident_a or resident_b or (traffic_reuse and _reuse_module(m, k, n) != "atlas_program_reuse_blocked"):
            raise ValueError("direct input supports base and blocked recipes without legacy residency")
    if epilogue is not None:
        if (epilogue.m, epilogue.n) != (m, n):
            raise ValueError("epilogue geometry differs from contraction")
        return dma_load_bytes(m, k, n, resident_a=resident_a, resident_b=resident_b,
                              traffic_reuse=traffic_reuse, direct_input=direct_input) + epilogue.dma_load_bytes
    if _reuse_enabled(traffic_reuse):
        module = _reuse_module(m, k, n, resident_a=resident_a, resident_b=resident_b)
        from importlib import import_module
        return import_module(module).dma_load_bytes(m, k, n)
    if resident_a and resident_b:
        raise ValueError("resident A and resident B are mutually exclusive")
    if resident_b:
        return m * k * (n // 32) + k * n
    if resident_a:
        return m * k + k * n * (m // 32)
    return 2 * m * k * (n // 32)


@checked_assembly(dispatch=True)
def assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False, pretranspose_resident_b=False,
    resident_a=False, traffic_reuse=False, epilogue=None, direct_input=None, weight_source=None):
    if weight_source is not None:
        weight_source.validate(k, n)
    if direct_input is not None:
        direct_input.validate(m, k, output_offset=output_offset, output_bytes=epilogue.output_bytes if epilogue else 2 * m * n)
        if resident_a or resident_b or (traffic_reuse and _reuse_module(m, k, n) != "atlas_program_reuse_blocked"):
            raise ValueError("direct input supports base and blocked recipes without legacy residency")
        if direct_input.kernel > 1 and not traffic_reuse:
            raise ValueError("kxk direct input requires the blocked recipe")
    if epilogue is not None:
        if (epilogue.m, epilogue.n, epilogue.output_offset) != (m, n, output_offset):
            raise ValueError("epilogue geometry differs from contraction")
    if _reuse_enabled(traffic_reuse):
        module = _reuse_module(m, k, n, resident_a=resident_a, resident_b=resident_b,
                               pretranspose_resident_b=pretranspose_resident_b)
        if type(output_offset) is not int or output_offset != m * k + k * n + 256:
            raise ValueError("traffic reuse must preserve the original output offset and guard")
        if type(rescale_exponent) is not int or not -126 <= rescale_exponent <= 127:
            raise ValueError("traffic reuse requires a normal finite BF16 integer rescale")
        from importlib import import_module
        extra = {"direct_input": direct_input} if direct_input is not None else {}
        if weight_source is not None:
            if module not in ("atlas_program_reuse_blocked", "atlas_program_reuse_tiny_breadth", "atlas_program_reuse_tiny",
                              "atlas_program_reuse_resident", "atlas_program_reuse_stream", "atlas_program_reuse_smol_connector"):
                raise ValueError("explicit weight source is not supported by this recipe")
            extra["weight_source"] = weight_source
        return import_module(module).assembly(m, k, n, output_offset, rescale_exponent, epilogue=epilogue, **extra)
    if resident_a and resident_b:
        raise ValueError("resident A and resident B are mutually exclusive")
    if pretranspose_resident_b and not resident_b:
        raise ValueError("--pretranspose-resident-b requires --resident-b")
    if resident_a and m * k > 256 * 1024:
        raise ValueError("resident A must fit one 256-KiB VMEM bank")
    if resident_b and k * n > 256 * 1024:
        raise ValueError("resident B must fit one 256-KiB VMEM bank")
    stride = k * 32
    lines = ["li x1, " + ("0x10000" if resident_a else "0x800"), "li x2, " + ("0x10000" if resident_b else "0x900"),
        "li x3, 0xa00", "li x5, 1024", "li x8, 2048", "li x9, 0xb00", "li x10, " + ("0x10000" if resident_a else "0"),
        f"li x12, {output_offset}", f"li x13, {m // 32}", f"li x15, {weight_source.low if weight_source else m * k}", f"li x16, {stride}", f"li x17, {n // 32}",
        "li x30, 1", "dma.config x30, ch0"]
    if weight_source is not None:
        lines += weight_source.setup(6, (2,))
    if resident_a:
        lines += ["add x19, x1, x0", "li x20, 0", f"li x21, {m * k // 1024}", f"li x18, {stride // 4}", "resident_a_loop:",
            "dma.load x19, x20, x5, ch0", "dma.wait ch0", "addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
            "beq x21, x0, resident_a_done", "nop", "jal x0, resident_a_loop", "nop", "resident_a_done:"]
    if resident_b:
        # DMA/VLOAD VMEM operands count four-byte words; DRAM operands count bytes.
        lines += ["add x19, x2, x0", "add x20, x15, x0", f"li x21, {k * n // 1024}",
            f"li x18, {stride // 4}", "resident_b_loop:", *( ["dma.load x19, x20, x5, ch2", "dma.wait ch2"] if weight_source else ["dma.load x19, x20, x5, ch0", "dma.wait ch0"])]
        if pretranspose_resident_b:
            # Cache the raw byte transpose in the same bank-1 tile, in-program.
            lines += ["vload m1, 0(x19)", "delay 34", "vtrpose.xlu m1, m1", "delay 66", "vstore m1, 0(x19)", "delay 34"]
        lines += ["addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1", "beq x21, x0, resident_b_done", "nop",
            "jal x0, resident_b_loop", "nop", "resident_b_done:"]
    lines += ["rows_loop:", "add x11, " + ("x2" if resident_b else "x15") + ", x0", "add x14, x17, x0", "columns_loop:",
        "add x4, x10, x0", "add x6, x11, x0"]
    for tile in range(k // 32):
        if not resident_a:
            lines += (direct_input.gather(1, 4, f"direct_a_{tile}") if direct_input is not None else
                      ["dma.load x1, x4, x5, ch0", "dma.wait ch0"])
        if not resident_b:
            lines += (["dma.load x2, x6, x5, ch2", "dma.wait ch2"] if weight_source else
                      ["dma.load x2, x6, x5, ch0", "dma.wait ch0"])
        lines += ["vload m0, 0(" + ("x4" if resident_a else "x1") + ")", "delay 34",
            "vload m1, 0(" + ("x6" if resident_b else "x2") + ")", "delay 34"]
        if not pretranspose_resident_b:
            lines += ["vtrpose.xlu m1, m1", "delay 66"]
        lines += ["vmatpush.weight.mxu0 w0, m1", "delay 32",
            ("vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0") + " acc0, m0, w0", "delay 96",
            "addi x4, x4, " + ("256" if resident_a else "1024"), "addi x6, x6, " + ("256" if resident_b else "1024")]
    if rescale_exponent is not None:
        if not -126 <= rescale_exponent <= 127:
            raise ValueError("VPU rescale factor must be normal finite BF16")
    from atlas_program_reuse_common import store_output
    lines += store_output(0, 12, rescale_exponent, epilogue, initialize_rescale=True)
    lines += ["add x12, x12, x8", "add x11, x11, " + ("x18" if resident_b else "x16"), "addi x14, x14, -1",
        "beq x14, x0, columns_done", "nop", "jal x0, columns_loop", "nop", "columns_done:",
        "add x10, x10, " + ("x18" if resident_a else "x16"), "addi x13, x13, -1", "beq x13, x0, finished", "nop",
        "jal x0, rows_loop", "nop", "finished:", "nop", "ecall"]
    return "\n".join(lines) + "\n"


def preflight_bundle(args):
    """Admit the NPZ central directory before any input digest or content read."""
    if args.smoke:
        if args.input or args.manifest:
            raise ValueError("--smoke excludes --input/--manifest")
        return
    if not args.input or not args.manifest:
        raise ValueError("require --input and --manifest, or --smoke")
    with zipfile.ZipFile(safe_path(args.input)) as archive:
        validate_archive_members(archive.infolist())


def code_sources(*, traffic_reuse=False):
    names = ("atlas_program.py", "atlas_program_checks.py", "atlas_epilogue.py", "atlas_rne_add.py", "atlas_resnet_input.py",
             "atlas_program_reuse_common.py", "run_model_tile.py", "atlas_tile_arithmetic.py", "atlas_weight_source.py")
    if _reuse_enabled(traffic_reuse):
        names += tuple(module + ".py" for module in dict.fromkeys(_TRAFFIC_REUSE_MODULES.values()))
    return {name: sha256(Path(__file__).with_name(name)) for name in names}


def legacy_epilogue_assembly(m, k, n, output_offset, rescale_exponent, *, epilogue=None, direct_input=None, weight_source=None):
    """Reproduce C's archived reuse programs with their original hook registers.

    This compatibility entry point is only for the non-Tiny C schedules. New
    launches use assembly(), whose recipe-selected scratch survives the low
    arena relocators. The archived x30/x31 programs predate that admission.
    """
    import re
    from importlib import import_module
    from atlas_epilogue import Epilogue
    from atlas_program_checks import check_matrix_registers, check_scalar_scratch

    module = _reuse_module(m, k, n)
    if module not in {"atlas_program_reuse_blocked", "atlas_program_reuse_resident",
                      "atlas_program_reuse_stream", "atlas_program_reuse_smol_connector"}:
        raise ValueError("legacy epilogue emission excludes Tiny schedules")
    if epilogue is not None and (type(epilogue) is not Epilogue
            or epilogue.scratch_registers != (30, 31)
            or any(step.op not in {"mul", "add", "relu"} for step in epilogue.steps)
            or epilogue.output_row_stride is not None or epilogue.output_column_tile):
        raise ValueError("legacy epilogue emission requires C's original hook contract")
    source = assembly(m, k, n, output_offset, rescale_exponent, traffic_reuse=True,
                      epilogue=epilogue, direct_input=direct_input, weight_source=weight_source)
    if epilogue is None:
        return source
    pair = (27, 28) if module == "atlas_program_reuse_resident" else (10, 11)
    # Gather temporaries are dead before the hook, as in checked_assembly.
    extra = {} if weight_source is None else {"weight_source": weight_source}
    baseline = import_module(module).assembly(m, k, n, output_offset, rescale_exponent, **extra)
    check_scalar_scratch(baseline, pair)
    check_scalar_scratch(baseline, (30, 31))
    registers = {str(pair[0]): "30", str(pair[1]): "31"}
    source = re.sub(r"\bx(\d+)\b", lambda match: "x" + registers.get(match[1], match[1]), source)
    check_matrix_registers(source)
    return source
