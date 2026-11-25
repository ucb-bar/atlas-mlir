"""Bind the checker's modules to the files the gate hashed, whatever else is on the import path.

The checker and the staged step workers import the runtime siblings and Atlas's instruction decoder by name. bind()
loads each from an explicit file before anything imports it by name, and makes `tools` a package of exactly one
directory, so the order of sys.path or a same-named module or package elsewhere cannot substitute another file. A name
already loaded from another file is refused.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

RUNTIME = ('model_runtime', 'functional_runtime')  # in import order: functional_runtime imports model_runtime
INTERFACES = ('check_variant_inventory', 'check_program_decode', 'model_instructions')


def _load(name, path):
    path = Path(path).resolve(strict=True)
    loaded = sys.modules.get(name)
    if loaded is not None:
        origin = getattr(loaded, '__file__', None)
        if origin is None or Path(origin).resolve() != path:
            raise ImportError(f'{name} is already loaded from {origin}, not from {path}')
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def bind(runtime, atlas_root, modules=RUNTIME):
    """Load ATLAS_ROOT/tools' decoder interfaces, then MODULES from the RUNTIME directory, by file."""
    tools = Path(atlas_root).resolve(strict=True) / 'tools'
    package = sys.modules.get('tools')
    if package is None:
        package = types.ModuleType('tools', 'Atlas instruction decoder interfaces, bound to one directory')
        package.__path__, package.__package__ = [str(tools)], 'tools'
        sys.modules['tools'] = package
    elif list(getattr(package, '__path__', ())) != [str(tools)]:
        raise ImportError(f'tools is already loaded from {list(getattr(package, "__path__", ()))}, not from {tools}')
    for name in INTERFACES:
        setattr(package, name, _load(f'tools.{name}', tools / f'{name}.py'))
    for name in modules:
        _load(name, Path(runtime) / f'{name}.py')
