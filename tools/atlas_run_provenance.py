"""Content identities for a selection run, with locations kept as evidence.

Capture before preparing/searching, bind generated facts before consuming them,
and complete before publishing selected artifacts. This records the caller's
opaque policy identity; it does not define a search-policy format. No model or
numerical package is imported here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat


SCHEMA = "atlas.run_provenance.v1"


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_copy(value):
    return json.loads(_encoded(value))


def safe_path(value):
    """Reject prohibited names and links before following any filesystem path."""
    path = Path(value)
    def lexical(candidate):
        if any(word in part.lower() for part in candidate.parts for word in ("hammer", "vlsi")):
            raise ValueError("prohibited provenance path")
        if ".." in candidate.parts:
            raise ValueError("parent traversal is unsupported")
    lexical(path)
    path = Path(os.path.abspath(path))
    lexical(path)
    # Root-first inspection must stop at a link, including dangling links.
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ValueError("provenance paths must not contain symlinks")
    return path


def _read(path):
    path = safe_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("provenance input must be a regular file")
    return path, path.read_bytes()


def _content(data):
    return {"sha256": _digest(data), "size_bytes": len(data)}


def _tree(root, suffixes=None):
    root = safe_path(root)
    if not root.is_dir():
        raise ValueError("provenance tree must be a directory")
    rows = []
    def visit(directory):
        with os.scandir(directory) as stream:
            entries = sorted(stream, key=lambda entry: entry.name)
        # Check names before calling any entry method, even is_symlink().
        if any(word in entry.name.lower() for entry in entries for word in ("hammer", "vlsi")):
            raise ValueError("prohibited provenance tree entry")
        for entry in entries:
            if entry.is_symlink():
                raise ValueError("provenance trees must not contain symlinks")
            path = directory / entry.name
            if entry.is_dir(follow_symlinks=False):
                if entry.name != "__pycache__":
                    visit(path)
            elif suffixes is None or path.suffix in suffixes:
                _, data = _read(path)
                rows.append(dict(path=str(path.relative_to(root)), **_content(data)))
    visit(root)
    return {"sha256": _digest(_encoded(rows)), "files": rows}


def selection_facts_identity(facts):
    """Return (semantic digest, verified provider digest) without path aliases.

    Keep the provider's sealed digest as evidence. Only the known configuration
    curve filename is substituted, using the curve bytes already bound by that
    verified fact payload; unrelated strings and content fields remain intact.
    """
    from atlas_derived_contract import semantic_payload, verify_derived

    verified = verify_derived(facts)
    payload = semantic_payload(facts)
    configuration = payload["model"]["configuration"]
    params = configuration.get("dma_memory_params", {})
    if isinstance(params.get("curves"), str):
        digest = payload.get("dma_timing", {}).get("curve_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("curve path normalization requires a verified curve content digest")
        params["curves"] = {"sha256": digest}
    payload["model"]["configuration_sha256"] = _digest(_encoded(configuration))
    return _digest(_encoded(payload)), verified


class RunCapture:
    """A filesystem snapshot; completion fails if any consumed input changes.

    ``files`` and ``trees`` use stable, caller-chosen role names. Tree values
    are a path or ``(path, suffixes)``; ``None`` includes every regular file.
    ``evidence_files`` records raw reports without affecting semantic identity.
    ``semantic`` accepts content identities already established by a provider.
    Configuration path fields must be normalized explicitly by the caller
    (for example by substituting the checked curve content digest).
    """
    def __init__(self, source, *, regions=(), files=None, trees=None,
                 evidence_files=None, configuration=None, search_policy=None,
                 revisions=None, archive_dir=None):
        if search_policy is None:
            raise ValueError("run provenance requires the versioned effective search policy")
        self._files = {}
        self._trees = {}
        self._semantic = {"source": None, "regions": [], "files": {}, "trees": {}}
        self._evidence = {"locations": {}, "revisions": _json_copy(revisions or {}),
                          "files": {}, "regions": []}
        self._archive = safe_path(archive_dir) if archive_dir is not None else None
        self._semantic["source"] = self._capture_file("source", source)
        if configuration is not None:
            self._semantic["configuration"] = _json_copy(configuration)
        self._semantic["search_policy"] = _json_copy(search_policy)
        for index, region_path in enumerate(regions):
            role = f"region:{index}"
            path, raw = _read(region_path)
            self._remember(role, path, raw)
            document = json.loads(raw)
            if not isinstance(document, dict) or type(document.get("address")) is not int or document["address"] < 0:
                raise ValueError("model region requires a nonnegative integer address")
            initial = bytes.fromhex(document["initial_hex"])
            record = {"address": document["address"], "initial": _content(initial)}
            archive = {"json": self._archive_bytes(raw), "initial": self._archive_bytes(initial)}
            if "expected_hex" in document:
                expected = bytes.fromhex(document["expected_hex"])
                record["expected"] = _content(expected)
                archive["expected"] = self._archive_bytes(expected)
            self._semantic["regions"].append(record)
            self._evidence["regions"].append(archive)
        self.bind(files=files, trees=trees, evidence_files=evidence_files)

    def _remember(self, role, path, data):
        if role in self._files:
            raise ValueError("duplicate provenance file role: " + role)
        self._files[role] = (path, _content(data))
        self._evidence["locations"][role] = str(path)

    def _capture_file(self, role, value):
        path, data = _read(value)
        self._remember(role, path, data)
        return _content(data)

    def _archive_bytes(self, data):
        content = _content(data)
        if self._archive is None:
            return dict(content, hex=data.hex())
        directory = safe_path(self._archive / "objects")
        directory.mkdir(parents=True, exist_ok=True)
        path = safe_path(directory / content["sha256"])
        if path.exists():
            if _read(path)[1] != data:
                raise ValueError("provenance archive object differs from its digest")
        else:
            path.write_bytes(data)
        role = "archive:" + content["sha256"]
        if role not in self._files:
            self._remember(role, path, data)
        return dict(content, path=str(path))

    def bind(self, *, files=None, trees=None, evidence_files=None, semantic=None):
        """Snapshot additional inputs before use, including newly generated facts."""
        for role, path in (files or {}).items():
            self._semantic["files"][role] = self._capture_file("file:" + role, path)
        for role, specification in (trees or {}).items():
            if role in self._trees:
                raise ValueError("duplicate provenance tree role: " + role)
            root, suffixes = specification if isinstance(specification, tuple) else (specification, None)
            root = safe_path(root)
            suffixes = tuple(suffixes) if suffixes is not None else None
            snapshot = _tree(root, suffixes)
            self._trees[role] = (root, suffixes, snapshot)
            self._semantic["trees"][role] = snapshot
            self._evidence["locations"]["tree:" + role] = str(root)
        for role, path in (evidence_files or {}).items():
            self._evidence["files"][role] = self._capture_file("evidence:" + role, path)
        for role, value in (semantic or {}).items():
            if role in self._semantic:
                raise ValueError("duplicate semantic provenance field: " + role)
            self._semantic[role] = _json_copy(value)
        return self

    def verify_completion(self):
        """Re-read rosters and bytes, refusing mutation, removal, or added files."""
        for role, (path, original) in self._files.items():
            try:
                current = _content(_read(path)[1])
            except (OSError, ValueError) as error:
                raise ValueError("provenance input unavailable or unsafe at completion: " + role) from error
            if current != original:
                raise ValueError("provenance input changed during selection: " + role)
        for role, (root, suffixes, original) in self._trees.items():
            if _tree(root, suffixes) != original:
                raise ValueError("provenance tree changed during selection: " + role)

    def complete(self, *, scoring_configurations=None, runtime_versions=None):
        """Verify the snapshot and seal path-independent identity plus evidence."""
        self.verify_completion()
        semantic = deepcopy(self._semantic)
        evidence = deepcopy(self._evidence)
        evidence["scoring_configurations"] = _json_copy(scoring_configurations or {})
        profiles = _json_copy(scoring_configurations or {})
        for profile in profiles.values():
            profile.pop("sha256", None)
            # The effective profile configuration and normalized base
            # configuration bind these values without the raw path-based hash.
            profile.get("model_identity", {}).pop("configuration_sha256", None)
        semantic["scoring_configurations"] = profiles
        if runtime_versions is not None:
            semantic["runtime_versions"] = _json_copy(runtime_versions)
        return {"schema": SCHEMA, "semantic": semantic,
                "semantic_sha256": _digest(_encoded(semantic)),
                "evidence": evidence}


def capture_run_provenance(source, **kwargs):
    return RunCapture(source, **kwargs)
