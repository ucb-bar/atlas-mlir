"""Resolve target specs, including generated AtlasCore enumerations."""
from pathlib import Path

import yaml

TARGETS = Path(__file__).resolve().parent / "targets"


def load_spec(path):
    path = Path(path).resolve()
    if path.parent.parent.name == "atlas" and path.parent.parent.parent == TARGETS:
        from .targets.atlas.gen_core import GENERATED, generate
        name = f"{path.parent.name}/{path.name}"
        if name in GENERATED:
            return generate(name)
    return yaml.safe_load(path.read_text())


def available(target, group=""):
    root = TARGETS / target / group
    names = {path.stem for path in root.glob("*.yaml")}
    if target == "atlas":
        from .targets.atlas.gen_core import GENERATED
        names.update(Path(name).stem for name in GENERATED if Path(name).parent.as_posix() == group)
    return sorted(names)
