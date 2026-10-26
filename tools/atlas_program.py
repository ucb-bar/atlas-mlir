"""Bounded contraction assembly and source-bundle admission."""

from pathlib import Path
from atlas_path_guard import checked as safe_path
from atlas_tooling import sha256
from run_model_tile import validate_archive_members
import zipfile


def dma_load_bytes(m, k, n, *, resident_a=False, resident_b=False):
    """Requested FP8 input bytes for a complete tiled contraction.

    Callers admit physical dimensions and resident bank capacity separately.
    """
    if resident_a and resident_b:
        raise ValueError("resident A and resident B are mutually exclusive")
    if resident_b:
        return m * k * (n // 32) + k * n
    if resident_a:
        return m * k + k * n * (m // 32)
    return 2 * m * k * (n // 32)


def assembly(m, k, n, output_offset, rescale_exponent=None, *, resident_b=False, pretranspose_resident_b=False,
    resident_a=False):
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
        f"li x12, {output_offset}", f"li x13, {m // 32}", f"li x15, {m * k}", f"li x16, {stride}", f"li x17, {n // 32}",
        "li x30, 1", "dma.config x30, ch0"]
    if resident_a:
        lines += ["add x19, x1, x0", "li x20, 0", f"li x21, {m * k // 1024}", f"li x18, {stride // 4}", "resident_a_loop:",
            "dma.load x19, x20, x5, ch0", "dma.wait ch0", "addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1",
            "beq x21, x0, resident_a_done", "nop", "jal x0, resident_a_loop", "nop", "resident_a_done:"]
    if resident_b:
        # DMA/VLOAD VMEM operands count four-byte words; DRAM operands count bytes.
        lines += ["add x19, x2, x0", "add x20, x15, x0", f"li x21, {k * n // 1024}",
            f"li x18, {stride // 4}", "resident_b_loop:", "dma.load x19, x20, x5, ch0", "dma.wait ch0"]
        if pretranspose_resident_b:
            # Cache the raw byte transpose in the same bank-1 tile, in-program.
            lines += ["vload m1, 0(x19)", "delay 34", "vtrpose.xlu m1, m1", "delay 66", "vstore m1, 0(x19)", "delay 34"]
        lines += ["addi x19, x19, 256", "addi x20, x20, 1024", "addi x21, x21, -1", "beq x21, x0, resident_b_done", "nop",
            "jal x0, resident_b_loop", "nop", "resident_b_done:"]
    lines += ["rows_loop:", "add x11, " + ("x2" if resident_b else "x15") + ", x0", "add x14, x17, x0", "columns_loop:",
        "add x4, x10, x0", "add x6, x11, x0"]
    for tile in range(k // 32):
        if not resident_a:
            lines += ["dma.load x1, x4, x5, ch0", "dma.wait ch0"]
        if not resident_b:
            lines += ["dma.load x2, x6, x5, ch0", "dma.wait ch0"]
        lines += ["vload m0, 0(" + ("x4" if resident_a else "x1") + ")", "delay 34",
            "vload m1, 0(" + ("x6" if resident_b else "x2") + ")", "delay 34"]
        if not pretranspose_resident_b:
            lines += ["vtrpose.xlu m1, m1", "delay 66"]
        lines += ["vmatpush.weight.mxu0 w0, m1", "delay 32",
            ("vmatmul.mxu0" if tile == 0 else "vmatmul.acc.mxu0") + " acc0, m0, w0", "delay 96",
            "addi x4, x4, " + ("256" if resident_a else "1024"), "addi x6, x6, " + ("256" if resident_b else "1024")]
    lines += ["vmatpop.bf16.acc.mxu0 m2, acc0", "delay 34"]
    if rescale_exponent is not None:
        if not -126 <= rescale_exponent <= 127:
            raise ValueError("VPU rescale factor must be normal finite BF16")
        lines += [f"vli.all m4, {(rescale_exponent + 127) << 7}", "delay 66", "vmul.bf16 m2, m2, m4", "delay 66"]
    lines += ["vstore m2, 0(x3)", "delay 34", "vstore m3, 0(x9)", "delay 34", "dma.store x12, x3, x8, ch0", "dma.wait ch0",
        "add x12, x12, x8", "add x11, x11, " + ("x18" if resident_b else "x16"), "addi x14, x14, -1",
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


def code_sources():
    return {name: sha256(Path(__file__).with_name(name)) for name in ("atlas_program.py", "run_model_tile.py",
        "atlas_tile_arithmetic.py")}
