"""Link one Atlas host runtime when an application combines several regions."""

from __future__ import annotations

from hashlib import sha256


def program_array(symbol: str, binary: bytes) -> str:
    """Give identical final images one immutable C array in a combined source."""
    if not binary or len(binary) % 4:
        raise ValueError("program must contain complete instruction words")
    digest = sha256(binary).hexdigest()
    name, guard = "atlas_image_" + digest, "ATLAS_IMAGE_" + digest.upper()
    words = ",".join(f"0x{int.from_bytes(binary[i : i + 4], 'little'):08x}u" for i in range(0, len(binary), 4))
    return (
        f"#ifndef {guard}\n#define {guard}\n"
        f"static const uint32_t {name}[] = {{{words}}};\n#endif\n"
        f"#define {symbol} {name}\n"
    )


HOST_RUNTIME_HEADERS = ("atlas_host_layout.h", "atlas_host.h")
RUNTIME_SHARED_FILES = tuple("runtime/" + name for name in (*HOST_RUNTIME_HEADERS, "atlas_host.c"))


def runtime_headers(runtime):
    """Read the public host ABI in the dependency order required by C packaging."""
    headers = {}
    for name in HOST_RUNTIME_HEADERS:
        path = runtime / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(runtime.resolve()):
            raise ValueError(f"selected Atlas runtime header is absent or unsafe: {name}")
        headers[name] = path.read_text()
    return headers


def runtime_members(headers: dict[str, str], source: str, linkage: str):
    """Keep standalone exports unchanged; separate exports share identical files.

    Region headers still contain the guarded public declarations. The application
    links ``runtime/atlas_host.c`` once, alongside all region translation units.
    This selects packaging only, never a different launch protocol or driver.
    """
    if linkage == "embedded":
        return {"atlas_host.c": source}, {}, {}
    if linkage != "separate":
        raise ValueError("runtime linkage must be embedded or separate")
    return (
        {},
        {
            **{"runtime/" + name: text.encode() for name, text in headers.items()},
            "runtime/atlas_host.c": source.encode(),
        },
        {"runtime_linkage": "separate", "runtime_source": "runtime/atlas_host.c"},
    )
