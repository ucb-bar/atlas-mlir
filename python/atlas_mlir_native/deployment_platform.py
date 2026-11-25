"""Atlas's platform files for a placed on-chip deployment: linker script, window assembly, and bound window.

Merlin places the deployment and publishes deployment.json; the adapter adds these members to that publication
(``AtlasApplicationAdapter.deployment_platform_files``), and the deploy binds saved deployments with explicit
invocation bytes through ``bind_deployment_payloads``.
"""

from __future__ import annotations

import json
from hashlib import sha256

from merlin.compiler.build.deployment import provider_sections, section_name
from merlin.compiler.semantic.target_binding import NativeCompilationError

# A saved deployment that predates the provider's window section name passes its own (from its frozen window.S)
# as deployment["window_section"], so its platform files regenerate byte for byte.
README = b"""
Deployment provides chip.ld, window.S and platform-cflags.txt. Apply
platform-cflags.txt to all platform sources, including drivers.
Also provided: harness_sha256.h and platform-requirements.json.
Null platform-owned values must be supplied
by the platform; this record does not configure a platform build.
"""
BIND_README = b"""
Bind explicit
invocation bytes with atlas_mlir_native.deployment_platform.bind_deployment_payloads
to emit window.bin; inputs are caller-owned and are never replaced with
fabricated values. Write the returned members outside the verified application/
directory, in a separate handoff/ directory beside application/.
"""
_REQUIREMENT_CHARACTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789")


def _refuse(reason, message):
    status = "resource_limit" if reason.endswith("_overflow") else "unsupported_semantics"
    raise NativeCompilationError(status, reason + ": " + message)


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < int(positive):
        _refuse("deployment_provider_contract", "invalid provider " + name)
    return value


def _requirement_name(name):
    """An upper-case C macro name: letters, digits, and underscores, not starting with a digit."""
    return (isinstance(name, str) and bool(name) and not name[0].isdigit()
            and set(name) <= _REQUIREMENT_CHARACTERS)


def _spaced(text):
    """Write each linker assignment ``=`` as `` = ``, leaving ``==``, ``<=``, ``>=``, ``!=``, and ``+=`` alone."""
    return "".join(" = " if character == "=" and text[i - 1:i] not in ("<", ">", "=", "!", "+")
                   and text[i + 1:i + 2] != "=" else character for i, character in enumerate(text))


def deployment_platform_files(deployment):
    """The handoff members Merlin's publication adds for a placed deployment, with their README notes."""
    files = platform_files(deployment)
    files["handoff/README.md"] = README + (b"" if deployment.get("paging") else BIND_README)
    return files


def bind_deployment_payloads(deployment, payloads):
    """Materialize a saved deployment with caller-supplied invocation bytes.

    Write returned members outside the verified application/ directory; adding
    window.bin there invalidates its file inventory. Use a separate handoff/
    directory beside the application/ directory for the returned members. Use
    platform-cflags.txt for every platform translation unit, including drivers.
    The final host build must still enforce chip.ld's aggregate section bound.
    """
    return platform_files(deployment, payloads)


def _host_abi(provider):
    """Physical section ceilings and usable aggregate capacity are distinct."""
    host = provider["host_allocation"]
    usable = _integer(host.get("usable_bytes"), "usable host bytes", positive=True)
    clock = provider.get("sys_clk_freq_hz")
    if type(clock) is not int or clock <= 0:
        _refuse("deployment_clock_contract", "system clock Hz must be a positive integer")
    if usable > host["bytes"]:
        _refuse("deployment_host_overflow", "usable host region exceeds physical capacity")
    sizes = {row["name"]: row["bytes"] for row in host["declared_reservations"]}
    names = ("stack", "heap") if host.get("measurement") == "measured_sections" else (
        "code", "data", "context", "stack", "heap")
    if host.get("measurement") == "measured_sections" and set(sizes) != set(names):
        _refuse("deployment_provider_contract", "measured host sections declare only stack and heap")
    for name in names:
        if name not in sizes:
            _refuse("deployment_provider_contract", "linker requires host reservation " + name)
        _integer(sizes[name], "host " + name + " bytes")
        if sizes[name] > usable:
            _refuse("deployment_host_overflow", "host section exceeds usable capacity: " + name)
    # Stack and heap consume fixed space; code/data/context are section ceilings.
    if sizes["stack"] + sizes["heap"] > usable:
        _refuse("deployment_host_overflow", "stack and heap exceed usable host capacity")
    return host, sizes, usable, clock


def _platform_linker(base, window, host_bytes, window_bytes, sizes, host_end, section, image_section,
                     parameter_section=None, *, symbolic=False, resident=None):
    # Measured sections have no fixed code or data ceiling; the aggregate host
    # budget below bounds them together with the declared stack and heap.
    code_ceiling = (f" ASSERT(. - ORIGIN(HOST) <= {sizes['code']}, \"deployment code reservation exceeded\")\n"
                    if "code" in sizes else "")
    data_ceiling = (f" ASSERT(__bss_end - __data_start <= {sizes['data']+sizes['context']}, "
                    "\"deployment data/context reservation exceeded\")\n" if "data" in sizes else "")
    # Read-only parameters end the window data in their own read-only load, with the instruction images after them.
    images_load = "parameters" if parameter_section else "window"
    parameters_load = " parameters PT_LOAD FLAGS(4);" if parameter_section else ""
    text=f'''OUTPUT_ARCH("riscv")
ENTRY(_start)
PHDRS {{ host PT_LOAD FLAGS(7); window PT_LOAD FLAGS(6);{parameters_load} }}
MEMORY {{ HOST (rwx) : ORIGIN = {base}, LENGTH = {host_bytes}
 WINDOW (rw) : ORIGIN = {window}, LENGTH = {window_bytes} }}
SECTIONS {{
 __boot_hart = 0; __stack_size = {sizes['stack']}; __heap_size = {sizes['heap']};
 .text : ALIGN(16) {{ __text_start=.; *(.text.init) *(.text.trap_vector) *(.text .text.*) __text_end=.; }} >HOST :host
 __text_load_start=LOADADDR(.text);
 .rodata : ALIGN(16) {{ *(.rodata .rodata.* .srodata .srodata.*) }} >HOST :host
{code_ceiling} .data : ALIGN(16) {{ __data_start=.; *(.data .data.*) __global_pointer$=.+0x800; *(.sdata .sdata.*) *(.htif) }} >HOST :host
 .tdata : ALIGN(16) {{ __tdata_start=.; *(.tdata .tdata.*) __tdata_end=.; __data_end=.; }} >HOST :host
 __data_load_start=LOADADDR(.data); __tdata_size=SIZEOF(.tdata);
 .bss (NOLOAD) : ALIGN(16) {{ __bss_start=.; __tbss_start=.; *(.tbss .tbss.*) __tbss_end=.; *(.sbss .sbss.* .bss .bss.*) *(COMMON) . = ALIGN(16); __bss_end=.; }} >HOST :host
 __tbss_size=__tbss_end-__tbss_start; __tbss_offset=0;
{data_ceiling} __end=.;
 .heap (NOLOAD) : ALIGN(16) {{ __heap_start=.; . += __heap_size; __heap_end=.; }} >HOST :host
 .stack (NOLOAD) : ALIGN(16) {{ __stack_start=.; . += __stack_size; __stack_end=.; __sp=.; }} >HOST :host
 __stack_shift=LOG2CEIL(__stack_size);
 ASSERT(__stack_end <= {host_end}, "deployment host budget exceeded")
 .window : {{ KEEP(*({section})) }} >WINDOW :window
 /DISCARD/ : {{ *(.eh_frame .eh_frame.* .comment) }}
}}
'''
    if symbolic:
        # EXTERN makes PROVIDE definitions survive even with no object reference.
        # Bare symbols reserve addresses without advancing the section cursor or
        # extending a PT_LOAD zero-fill extent. __end is the allocator's origin.
        text = text.replace("SECTIONS {",
            "EXTERN(__end __heap_start __heap_end __stack_start __stack_end __sp __stack_shift)\nSECTIONS {")
        start = text.index(" __end=.;")
        end = text.index(" .window :", start)
        text = text[:start] + f''' PROVIDE(__heap_start = ABSOLUTE(ALIGN(__bss_end, 16)));
 PROVIDE(__end = __heap_start);
 PROVIDE(__heap_end = __heap_start + __heap_size);
 PROVIDE(__stack_start = ALIGN(__heap_end, 16));
 PROVIDE(__stack_end = __stack_start + __stack_size);
 PROVIDE(__sp = __stack_end);
 PROVIDE(__stack_shift = LOG2CEIL(__stack_size));
 ASSERT(__end == __heap_start, "deployment allocator origin differs")
 ASSERT(__heap_start >= ORIGIN(HOST) && __heap_start >= __bss_end, "deployment heap overlaps linked storage")
 ASSERT(__heap_end >= __heap_start && __heap_end - __heap_start == __heap_size, "deployment heap range differs")
 ASSERT(__stack_start >= __heap_end, "deployment stack overlaps heap")
 ASSERT(__stack_end >= __stack_start && __stack_end - __stack_start == __stack_size, "deployment stack range differs")
 ASSERT((__stack_start & 15) == 0 && (__stack_end & 15) == 0 && __sp == __stack_end, "deployment stack alignment differs")
 ASSERT(__stack_start >= ORIGIN(HOST) && __stack_end <= {host_end}, "deployment host budget exceeded")
 ASSERT(ORIGIN(HOST)+LENGTH(HOST) <= ORIGIN(WINDOW) || ORIGIN(WINDOW)+LENGTH(WINDOW) <= ORIGIN(HOST), "deployment host and window overlap")
''' + text[end:]
        # Instruction images are device payloads: they load with the window and
        # are copied to IMEM from there, so they occupy no host memory.
        text = text.replace(f" .window : {{ KEEP(*({section})) }} >WINDOW :window\n",
            f" .window : {{ KEEP(*({section})) }} >WINDOW :window\n"
            + (f" {parameter_section} : {{ KEEP(*({parameter_section})) }} >WINDOW :parameters\n" if parameter_section else "")
            + f" {image_section} : ALIGN(16) {{ __{image_section[1:]}_start=.; *({image_section}) __{image_section[1:]}_end=.; }} >WINDOW :{images_load}\n")
    if resident is not None:
        # The resident arena is a read-only load of its own after the window, in the resident capacity.
        resident_section, address, size = resident
        text = (text.replace("window PT_LOAD FLAGS(6);", "window PT_LOAD FLAGS(6); resident PT_LOAD FLAGS(4);")
                .replace(f"LENGTH = {window_bytes} }}", f"LENGTH = {window_bytes}\n RESIDENT (r) : ORIGIN = {address}, LENGTH = {size} }}")
                .replace(" /DISCARD/ :", f" {resident_section} : {{ KEEP(*({resident_section})) }} >RESIDENT :resident\n /DISCARD/ :"))
    return _spaced(text).encode()


def _window_source(section, parameter_section=None, parameters=None):
    label = section[1:]
    if parameters is None:
        return (f'.section {section},"aw",@progbits\n.global {label}_start,{label}_end\n'
                f'{label}_start:\n.incbin "window.bin"\n{label}_end:\n').encode()
    offset, size = parameters
    return (f'.section {section},"aw",@progbits\n.global {label}_start,{label}_end\n'
            f'{label}_start:\n.incbin "window.bin",0,{offset}\n'
            f'.section {parameter_section},"a",@progbits\n.incbin "window.bin",{offset},{size}\n{label}_end:\n').encode()


def _resident_load(deployment):
    """The provider's resident section, address, and size when the deployment loads a resident arena, else None."""
    row = deployment.get("resident")
    if not row:
        return None
    section = section_name(deployment["provider"].get("resident_section"), "resident")
    if section in provider_sections(deployment["provider"]) or deployment.get("paging"):
        _refuse("deployment_provider_contract", "the resident section must differ from the other sections")
    return section, row["address"], row["bytes"]


def _read_only_parameters(deployment, parameter_section):
    """The window offset and size of the immutable parameters when the provider loads them read-only, else None."""
    row = deployment.get("parameters")
    if parameter_section is None or not row:
        return None
    if (deployment.get("paging") or deployment.get("host_linker_abi") != "symbolic_reservations_v1"
            or row["address"] + row["bytes"] != deployment["window"]["address"] + deployment["window_required_bytes"]):
        _refuse("deployment_provider_contract", "read-only parameters must end the complete window data")
    return row["address"] - deployment["window"]["address"], row["bytes"]


def platform_files(deployment, payloads=None):
    """Emit linker/assembly and, with explicit invocation bytes, the window.

    payloads maps deployment-relative input and parameter paths to bytes. None
    leaves the invocation unbound; missing inputs are never synthesized. This
    function reads only the packaged SHA helper, never payload paths, and can
    also bind a saved deployment.
    """
    host, sizes, usable, clock = _host_abi(deployment["provider"])
    if host != deployment["host_allocation"]:
        _refuse("deployment_provider_contract", "host allocation differs from provider")
    base = host["address"]
    window = deployment["window"]["address"]
    abi = deployment.get("host_linker_abi")
    if abi not in (None, "symbolic_reservations_v1"):
        _refuse("deployment_provider_contract", "unsupported host linker ABI")
    symbolic = abi == "symbolic_reservations_v1"
    if host.get("measurement") == "measured_sections" and not symbolic:
        _refuse("deployment_provider_contract", "measured host sections require symbolic reservations")
    # Old saved deployments retain their frozen aggregate and exact bytes.
    # New deployments use all provider-declared physically backed host memory.
    host_bytes = host["bytes"] if symbolic else usable
    provider_window, image_section, parameter_section = provider_sections(deployment["provider"])
    section = section_name(deployment.get("window_section", provider_window), "window")
    if symbolic and section != provider_window:
        _refuse("deployment_provider_contract", "new deployments use the provider window section")
    parameters = _read_only_parameters(deployment, parameter_section)
    loaded = parameter_section if parameters else None
    resident = _resident_load(deployment)
    linker = _platform_linker(base, window, host_bytes, deployment["window"]["bytes"],
                              sizes, f"{base + host_bytes:#010x}", section, image_section, loaded, symbolic=symbolic,
                              resident=resident)
    if not symbolic:
        # Reproduce the whitespace of the historical build_kits.py patch.
        linker = linker.replace(f"LENGTH  =  {usable}\n".encode(), f"LENGTH = {usable}\n".encode())
    output = {
        "handoff/chip.ld": linker,
        "handoff/window.S": _window_source(section, loaded, parameters) + (
            f'.section {resident[0]},"a",@progbits\n.incbin "resident.bin"\n'.encode() if resident else b""),
        "handoff/platform-cflags.txt": f"-DSYS_CLK_FREQ={clock}\n".encode(),
    }
    from merlin.common.paths import data_path
    output["handoff/harness_sha256.h"] = data_path("runtime", "baremetal", "spike", "output_sha256.h").read_bytes()
    requirements = {
        "SYS_CLK_FREQ": {"value": clock, "owner": "compiler-required",
                         "source": "deployment.json:provider.sys_clk_freq_hz",
                         "scope": "all platform translation units, including drivers"},
        "APPLICATION_LINKER": {"value": "handoff/chip.ld", "owner": "compiler-required",
                               "source": "generated chip.ld; platform resolves its installed path"},
        "APPLICATION_BUNDLE": {"value": None, "owner": "platform-owned",
                               "source": "platform location of the application and bound handoff"},
    }
    platform_owned = deployment["provider"].get("platform_owned_requirements", [])
    if (not isinstance(platform_owned, list)
            or any(not _requirement_name(name) or name in requirements for name in platform_owned)
            or len(set(platform_owned)) != len(platform_owned)):
        _refuse("deployment_provider_contract", "invalid platform-owned requirement names")
    for name in platform_owned:
        requirements[name] = {"value": None, "owner": "platform-owned",
                              "source": "platform build configuration; not defined by deployment.json"}
    output["handoff/platform-requirements.json"] = (
        json.dumps(requirements, sort_keys=True, indent=2) + "\n").encode()
    if payloads is None:
        return output
    if deployment.get("paging"):
        _refuse("deployment_paging_contract", "paged deployments bind through refill segments, not window.bin")
    required = _integer(deployment["window_required_bytes"], "window required bytes")
    if required > deployment["window"]["bytes"]:
        _refuse("deployment_window_overflow", "payload exceeds deployment window")
    payload = bytearray(required)
    occupied = []

    def write(row, data):
        offset = row["address"] - window
        size = row["bytes"]
        if offset < 0 or size < 0 or offset + size > required:
            _refuse("deployment_window_overflow", "payload range outside window")
        if any(offset < end and start < offset + size for start, end in occupied):
            _refuse("deployment_payload_contract", "overlapping payload ranges")
        occupied.append((offset, offset + size))
        if not isinstance(data, bytes) or len(data) != size:
            _refuse("deployment_payload_contract", "payload byte count differs")
        if "sha256" in row and sha256(data).hexdigest() != row["sha256"]:
            _refuse("deployment_payload_contract", "payload digest differs")
        payload[offset:offset + size] = data

    for row in [*deployment["inputs"], *deployment.get("state_inputs", []), *([deployment["parameters"]] if deployment["parameters"] else [])]:
        if row["path"] not in payloads:
            _refuse("deployment_payload_missing", "explicit invocation bytes required: " + row["path"])
        write(row, payloads[row["path"]])
    for row in deployment["outputs"]:
        write(row, b"\xa5" * row["bytes"])
    output["handoff/window.bin"] = bytes(payload)
    if resident:
        row = deployment["resident"]
        data = payloads.get(row["path"])
        if not isinstance(data, bytes) or len(data) != row["bytes"] or sha256(data).hexdigest() != row["sha256"]:
            _refuse("deployment_payload_contract", "resident arena bytes differ from the deployment")
        output["handoff/resident.bin"] = data
    return output
