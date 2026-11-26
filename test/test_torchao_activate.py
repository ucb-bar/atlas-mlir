"""Activated exporter sources keep one module, and one operator registration, per file."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from atlas_mlir import torchao

CHILD = textwrap.dedent("""
    import sys
    import torch
    from atlas_mlir import torchao

    # The synthetic tree is not a pinned git checkout.
    torchao.check = lambda root: None
    torchao.activate()
    import exporter.probe_ops

    class Model(torch.nn.Module):
        def forward(self, x):
            return torch.ops.onefile.double(x)

    x = torch.arange(4, dtype=torch.float32) + 1
    torch.export.save(torch.export.export(Model(), (x,)), sys.argv[1])
    program = torch.export.load(sys.argv[1])
    import probe_ops  # the exporter's top-level spelling, after the program is loaded
    assert probe_ops is exporter.probe_ops
    (target,) = [node.target for node in program.graph.nodes if node.op == "call_function"]
    assert target is torch.ops.onefile.double.default, "loaded node lost its live registration"
    assert torch.equal(program.module()(x), x * 2)
""")

OPS = textwrap.dedent("""
    import torch

    @torch.library.custom_op("onefile::double", mutates_args=())
    def double(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    @double.register_fake
    def _(x):
        return torch.empty_like(x)
""")


def test_top_level_exporter_import_reuses_package_module(tmp_path):
    exporter = tmp_path / "src" / "exporter"
    exporter.mkdir(parents=True)
    (exporter / "__init__.py").write_text("")
    (exporter / "probe_ops.py").write_text(OPS)
    # The child imports the same atlas_mlir as this process.
    package_root = str(Path(torchao.__file__).resolve().parents[1])
    env = dict(os.environ, ATLAS_TORCHAO_ROOT=str(tmp_path),
               PYTHONPATH=os.pathsep.join([package_root, os.environ.get("PYTHONPATH", "")]))
    result = subprocess.run([sys.executable, "-c", CHILD, str(tmp_path / "program.pt2")],
                            env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def _commit_fixture(root, loader, gitmodules, links):
    """Make a one-commit repository with the given reference loader, .gitmodules, and submodule links."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (root / "src" / "reference").mkdir(parents=True)
    (root / "src" / "reference" / "ref_loader.py").write_text(loader)
    (root / ".gitmodules").write_text(gitmodules)
    git("add", ".")
    for path, commit in links.items():
        git("update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "commit", "-q", "-m", "fixture")
    return git("rev-parse", "HEAD")


FIRST, SECOND = "1" * 40, "2" * 40
LOADER = f'PINS = {{\n    "first-ref": "{FIRST}",\n    "second-ref": "{SECOND}",\n}}\nORDER = ("first-ref", "second-ref")\n'
MODULES = "".join(f'[submodule "third_party/{name}"]\n\tpath = third_party/{name}\n\turl = https://example.invalid/{name}.git\n'
                  for name in ("first-ref", "second-ref", "unrelated"))


def test_reference_submodules_come_from_the_pinned_loader(tmp_path, monkeypatch):
    links = {"third_party/first-ref": FIRST, "third_party/second-ref": SECOND, "third_party/unrelated": "3" * 40}
    revision = _commit_fixture(tmp_path, LOADER, MODULES, links)
    monkeypatch.setitem(torchao.PINS, "atlas-torchao", revision)
    assert torchao.reference_submodules(tmp_path) == ("third_party/first-ref", "third_party/second-ref")


def test_reference_submodules_reject_a_link_off_its_declared_commit(tmp_path, monkeypatch):
    links = {"third_party/first-ref": FIRST, "third_party/second-ref": "4" * 40}
    revision = _commit_fixture(tmp_path, LOADER, MODULES, links)
    monkeypatch.setitem(torchao.PINS, "atlas-torchao", revision)
    try:
        torchao.reference_submodules(tmp_path)
    except torchao.ToolchainError as error:
        assert "third_party/second-ref" in str(error)
    else:
        raise AssertionError("a reference link off its declared commit was accepted")


def test_reference_submodules_reject_an_undeclared_submodule(tmp_path, monkeypatch):
    modules = MODULES.replace("third_party/second-ref", "third_party/other")
    links = {"third_party/first-ref": FIRST, "third_party/second-ref": SECOND}
    revision = _commit_fixture(tmp_path, LOADER, modules, links)
    monkeypatch.setitem(torchao.PINS, "atlas-torchao", revision)
    try:
        torchao.reference_submodules(tmp_path)
    except torchao.ToolchainError as error:
        assert "not a declared submodule" in str(error)
    else:
        raise AssertionError("an undeclared reference submodule was accepted")
