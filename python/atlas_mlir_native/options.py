"""Atlas-owned code generation choices."""

from dataclasses import asdict, dataclass, field, fields
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
    "ordered", "row-cache", "prefix-cache", "wide-cache", "paired-cache", "panel-cache",
)


@dataclass(frozen=True)
class CodegenOptions(GenericCodegenOptions):
    application_deployment: str | None = None
    native_pipeline: str = "mandatory"
    # Temporary: adjacent_patterns and the nine Boolean fields below have no command-line switch, and deploy never
    # selects them. Replacement: the optimized-pipeline follow-up pull request registers their switches.
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


def configure_codegen_parser(parser):
    """Register Atlas-only switches after Merlin's target-neutral arguments."""
    parser.add_argument("--application-deployment", choices=("onchip",), default=None,
                        help="publish fixed deployment bindings for the Atlas on-chip memory map")
    for name, choices in (
        ("native_pipeline", NATIVE_PIPELINES), ("staging_storage", STAGING_STORAGE),
        ("looped_operand_reuse", LOOPED_OPERAND_REUSE),
        ("quantized_operand_reuse", QUANTIZED_OPERAND_REUSE),
    ):
        parser.add_argument("--" + name.replace("_", "-"), choices=choices, default=None)

    # Unset switches take their CodegenOptions defaults in expand_codegen_arguments.
    for action in parser._actions:
        if action.dest == "application_residency":
            action.default = None
    parser.add_argument("--no-application-residency", dest="application_residency", action="store_false", default=None)
    parser.set_defaults(_codegen_options_type=CodegenOptions)


def expand_codegen_arguments(args):
    defaults = CodegenOptions()
    for field in fields(CodegenOptions):
        value = getattr(args, field.name, None)
        if value is None:
            setattr(args, field.name, getattr(defaults, field.name))


CodegenOptions.prepare_args = staticmethod(expand_codegen_arguments)
