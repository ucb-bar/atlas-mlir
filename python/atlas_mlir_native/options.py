"""Atlas-owned code generation choices, presets, and board memory assumptions."""

from dataclasses import asdict, dataclass, field, fields, replace
from merlin.target.soc import SoCDescription
from .soc import SOC
from merlin.compiler.semantic.codegen import (
    IO_MEMORY_OPTIONS, CodegenOptions as GenericCodegenOptions, codegen_options, materialization_cache,
    choose_reduction_unroll, current_codegen_options as generic_current_options,
    options_from_args as generic_options_from_args,
)

NATIVE_PIPELINES = ("mandatory", "optimized")
STAGING_STORAGE = ("dram", "vmem", "vmem-inputs")
LOOPED_OPERAND_REUSE = ("ordered", "resident-a")
QUANTIZED_OPERAND_REUSE = (
    "ordered", "row-cache", "prefix-cache", "wide-cache", "paired-cache", "panel-cache", "batch-panel-cache",
)


@dataclass(frozen=True)
class CodegenOptions(GenericCodegenOptions):
    application_deployment: str | None = None
    native_pipeline: str = "mandatory"
    adjacent_patterns: str = ""
    exponent_page_cache: bool = False
    staging_storage: str = "dram"
    looped_operand_reuse: str = "ordered"
    quantized_operand_reuse: str = "ordered"
    exact_binary_lane_groups: bool = False
    exact_binary_vector_preflight: bool = False
    exact_cast_vector_preflight: bool = False
    static_quantizer_sizing: bool = False
    exact_binary_routes: bool = False
    capture_postspill: bool = False
    kernel_library: bool = False
    device_kv_quantizer: bool = False
    # The deploy's host clock when it differs from the declared one, so the provider publishes the clock the image runs.
    host_clock_hz: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.application_deployment not in (None, "onchip"):
            raise ValueError("application deployment must be onchip or unset")
        for name, choices in (
            ("native_pipeline", NATIVE_PIPELINES),
            ("staging_storage", STAGING_STORAGE),
            ("looped_operand_reuse", LOOPED_OPERAND_REUSE),
            ("quantized_operand_reuse", QUANTIZED_OPERAND_REUSE),
        ):
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {', '.join(choices)}")
        if type(self.adjacent_patterns) is not str:
            raise ValueError("adjacent_patterns must be a string")
        for name in (
            "exponent_page_cache", "exact_binary_lane_groups", "exact_binary_vector_preflight",
            "exact_cast_vector_preflight", "static_quantizer_sizing", "exact_binary_routes",
            "capture_postspill", "kernel_library",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be Boolean")
        if type(self.device_kv_quantizer) is not bool:
            raise ValueError("device_kv_quantizer must be Boolean")
        if self.host_clock_hz is not None and (type(self.host_clock_hz) is not int or self.host_clock_hz <= 0):
            raise ValueError("host_clock_hz must be a positive integer")

    def identity_record(self):
        record = super().identity_record()
        if self.application_deployment is None:
            del record["application_deployment"]
        for name in (
            "native_pipeline", "adjacent_patterns", "exponent_page_cache", "staging_storage",
            "looped_operand_reuse", "quantized_operand_reuse", "exact_binary_lane_groups",
            "exact_binary_vector_preflight", "exact_cast_vector_preflight", "static_quantizer_sizing",
            "exact_binary_routes", "capture_postspill", "kernel_library",
        ):
            if getattr(self, name) == getattr(type(self), name):
                del record[name]
        if not self.device_kv_quantizer:
            del record["device_kv_quantizer"]
        if self.host_clock_hz is None:
            del record["host_clock_hz"]
        return record


_DEFAULT_OPTIONS = CodegenOptions()

def current_codegen_options():
    selected = generic_current_options()
    if isinstance(selected, CodegenOptions):
        return selected
    if selected == GenericCodegenOptions():
        return _DEFAULT_OPTIONS
    return CodegenOptions(**asdict(selected))


def options_from_args(args):
    from copy import copy
    args = copy(args)
    args._codegen_options_type = CodegenOptions
    return generic_options_from_args(args)


DISPATCH_REDUCTION = dict(
    adjacent_patterns="adjacent_square_reduce,adjacent_reduction_scale,adjacent_pointwise_chain,pointwise_narrowing,quantized_narrowing,quantized_epilogue,expanded_silu,quantized_operand_permutation,float_input_layout,quantizer_input_permutation,pointwise_input_widening",
    quantized_operand_reuse="batch-panel-cache", exponent_page_cache=True,
    staging_storage="vmem-inputs", native_pipeline="optimized", static_quantizer_sizing=True,
    exact_binary_routes=True, exact_binary_lane_groups=True, exact_binary_vector_preflight=True,
    exact_cast_vector_preflight=True, exact_float_unroll=True,
    execution_instruction_limit=128000000, device_kv_quantizer=True,
    io_memory_bytes=SOC.memory_region("atlas_data").capacity_bytes,
)


@dataclass(frozen=True)
class BoardDescription:
    soc: SoCDescription = SOC
    # Measured host prefix and staging demand are supplied by the caller.
    host_reserved_bytes: int | None = None
    staging_bytes: int | None = None
    residency: bool = True

    @property
    def ddr_base(self):
        return self.soc.memory_region("ddr").base

    @property
    def ddr_capacity_bytes(self):
        return self.soc.memory_region("ddr").capacity_bytes

    @property
    def dma_base(self):
        return self.soc.memory_region("dma").base

    @property
    def dma_capacity_bytes(self):
        return self.soc.memory_region("dma").capacity_bytes

    def options(self):
        if self.host_reserved_bytes is None:
            raise ValueError("--board sophialake requires --host-memory-reserved-bytes from the measured host footprint")
        if self.staging_bytes is None:
            raise ValueError("--board sophialake requires --io-memory-bytes from the measured staging demand")
        if type(self.staging_bytes) is not int or self.staging_bytes <= 0 or self.staging_bytes % 2048:
            raise ValueError("staging bytes must be positive and tile aligned")
        if (type(self.host_reserved_bytes) is not int or self.host_reserved_bytes <= 0
                or self.host_reserved_bytes % 2048
                or self.host_reserved_bytes + self.staging_bytes >= self.ddr_capacity_bytes):
            raise ValueError("host reservation must be positive, tile aligned, and leave room for staging and residency")
        start = self.ddr_base + self.host_reserved_bytes
        return dict(io_memory_base=start, io_memory_capacity_base=start,
                    io_memory_capacity_bytes=self.ddr_capacity_bytes - self.host_reserved_bytes,
                    io_memory_bytes=self.staging_bytes, application_residency=self.residency,
                    quantized_operand_reuse="wide-cache")


SOPHIALAKE = BoardDescription()


def configure_codegen_parser(parser):
    """Register Atlas-only switches after Merlin's target-neutral arguments."""
    from argparse import SUPPRESS, BooleanOptionalAction
    parser.add_argument("--application-deployment", choices=("onchip",), default=None,
                        help="publish fixed deployment bindings for the Atlas on-chip memory map")
    for name, choices in (
        ("native_pipeline", NATIVE_PIPELINES), ("staging_storage", STAGING_STORAGE),
        ("looped_operand_reuse", LOOPED_OPERAND_REUSE),
        ("quantized_operand_reuse", QUANTIZED_OPERAND_REUSE),
    ):
        parser.add_argument("--" + name.replace("_", "-"), choices=choices, default=None)
    parser.add_argument("--adjacent-patterns", default=None)
    for name in ("exponent_page_cache", "exact_binary_lane_groups", "exact_binary_vector_preflight",
                 "exact_cast_vector_preflight", "static_quantizer_sizing", "exact_binary_routes",
                 "capture_postspill", "kernel_library", "device_kv_quantizer"):
        parser.add_argument("--" + name.replace("_", "-"), action=BooleanOptionalAction, default=None)
    parser.add_argument("--host-clock-hz", type=int, default=None, help=SUPPRESS)
    parser.add_argument("--host-memory-reserved-bytes", type=lambda value: int(value, 0),
                        help="measured SophiaLake host prefix at DDR base; required with --board sophialake")
    # Explicit Boolean overrides must work regardless of their position relative to presets.
    for action in parser._actions:
        if action.dest in ("exact_float_unroll", "application_residency"):
            action.default = None
    parser.add_argument("--no-exact-float-unroll", dest="exact_float_unroll", action="store_false", default=None)
    parser.add_argument("--no-application-residency", dest="application_residency", action="store_false", default=None)
    parser.set_defaults(_codegen_options_type=CodegenOptions)


def expand_codegen_arguments(args):
    if not hasattr(args, "_explicit_codegen_fields"):
        args._explicit_codegen_fields = frozenset(field.name for field in fields(CodegenOptions)
            if getattr(args, field.name, None) is not None)
    recipe, board = getattr(args, "recipe", None), getattr(args, "board", None)
    if recipe not in (None, "dispatch-reduction"):
        raise ValueError("unknown Atlas recipe: " + recipe)
    if board not in (None, "sophialake"):
        raise ValueError("unknown Atlas board: " + board)
    if recipe and board:
        raise ValueError(
            "--recipe dispatch-reduction --board sophialake is unsupported: "
            "known spill rewrite work limit reached with a 16 MiB staging window; "
            "use --board sophialake without the recipe, or --recipe dispatch-reduction "
            "without the board at its 768 KiB staging window"
        )
    selected = dict(DISPATCH_REDUCTION) if recipe else {}
    reservation = getattr(args, "host_memory_reserved_bytes", None)
    if reservation is not None and board is None:
        raise ValueError("host memory reservation requires --board sophialake")
    if board:
        description = replace(SOPHIALAKE, host_reserved_bytes=reservation,
                              staging_bytes=getattr(args, "io_memory_bytes", None))
        selected.update(description.options())
    defaults = CodegenOptions()
    for field in fields(CodegenOptions):
        value = getattr(args, field.name, None)
        if value is None:
            setattr(args, field.name, selected.get(field.name, getattr(defaults, field.name)))
    if board:
        # Overrides remain subject to the physical board and host exclusion range.
        start = description.ddr_base + description.host_reserved_bytes
        end = description.ddr_base + description.ddr_capacity_bytes
        capacity_start, capacity_size = args.io_memory_capacity_base, args.io_memory_capacity_bytes
        if (capacity_start < description.dma_base
                or capacity_start + capacity_size > description.dma_base + description.dma_capacity_bytes):
            raise ValueError("selected capacity leaves the board DMA aperture")
        if capacity_start < start or capacity_start + capacity_size > end:
            raise ValueError("selected capacity overlaps reserved host memory or leaves physical DDR")
        if not capacity_start <= args.io_memory_base < args.io_memory_base + args.io_memory_bytes <= capacity_start + capacity_size:
            raise ValueError("selected staging window leaves usable board capacity")


CodegenOptions.prepare_args = staticmethod(expand_codegen_arguments)
