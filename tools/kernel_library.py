"""Import the pinned atlas-kernels submodule's generators by bare module name.

Kernel generators and their shared helpers (``atlas_program``, ``atlas_execution``,
``atlas_path_guard`` and the rest) live in ``third_party/atlas-kernels``. Import
this module before any of them; it is idempotent. ``module_path(name)`` returns
the library file for a module file name such as ``atlas_program.py``, and
``source_path(name)`` resolves a former ``tools/`` file name in this checkout.
"""

from pathlib import Path
import sys

LIBRARY = Path(__file__).absolute().parents[1] / "third_party" / "atlas-kernels"
if not (LIBRARY / "atlas_kernels_path.py").is_file():
    raise ImportError("atlas-kernels submodule is missing; run: git submodule update --init third_party/atlas-kernels")
if str(LIBRARY) not in sys.path:
    sys.path.append(str(LIBRARY))

from atlas_kernels_path import DIRECTORIES, module_path  # noqa: E402,F401

ROOT = LIBRARY.parents[1]


def relative_path(name):
    """Path below the atlas-mlir root of a former ``tools/`` file name, following moves into atlas-kernels."""
    try:
        return module_path(name).relative_to(ROOT)
    except ValueError:
        return Path("tools") / name


def source_path(name):
    """Absolute path in this checkout of a former ``tools/`` file name such as ``atlas_program.py``."""
    return ROOT / relative_path(name)


def moved_path(path):
    """Follow a missing ``<root>/tools/<module>.py`` path into atlas-kernels; other paths are returned unchanged."""
    path = Path(path)
    if path.parent == ROOT / "tools" and not path.exists():
        try:
            return module_path(path.name)
        except ValueError:
            pass
    return path
