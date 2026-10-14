"""Compare all timing data, normalizing only verified provenance/output paths."""
import hashlib
import json
from pathlib import Path
import re
from .regen_cache import safe_path
SCHEMAS = {
    "merlin.op_timing.v1": "op_timing",
    "rtl_extract.op_pairs.v1": "op_pairs",
    "rtl_extract.op_sequences.v1": "op_sequences",
    "merlin.op_pairs.v1": "op_pairs",
}

# Normalize only firtool hierarchy output attributes.
_HIERARCHY_OUTPUT = re.compile(
    rb'(firrtl\.moduleHierarchyFile\s*=\s*\[#hw\.output_file<)'
    rb'"(?:[^"\\\r\n]*/)?((?:model|top)_module_hierarchy\.json)"'
    rb'(,\s*excludeFromFileList>\])'
)

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def _indexed(records, key, where):
    if not isinstance(records, list):
        raise ValueError(f"{where}: expected a list")
    result = {}
    for record in records:
        identity = key(record)
        if identity in result:
            raise ValueError(f"{where}: duplicate record {identity}")
        result[identity] = record
    return result

def _runs(record):
    """Expand compressed intervals, comparing every gap and its entire result."""
    first, last = record["gaps"]
    if type(first) is not int or type(last) is not int or first > last:
        raise ValueError(f"{record['name']}: invalid gaps")
    results = {}
    for run in record["results"]:
        start, end = run["from"], run["to"]
        if type(start) is not int or type(end) is not int or not first <= start <= end <= last:
            raise ValueError(f"{record['name']}: result outside gaps")
        for gap in range(start, end + 1):
            if str(gap) in results:
                raise ValueError(f"{record['name']}: duplicate result for gap {gap}")
            results[str(gap)] = {k: v for k, v in run.items() if k not in ("from", "to")}
    if len(results) != last - first + 1:
        raise ValueError(f"{record['name']}: missing results for gaps")
    return results

class _Artifacts:
    def __init__(self, root, hw_ir=None, names=()):
        self.root = safe_path(root)
        self.hw_ir = safe_path(hw_ir) if hw_ir is not None else None
        self.hardware_cache = {}
        self.cache = {}
        self.active = set()
        names = tuple(names)
        self.names = {Path(name).name: name for name in names}
        if len(self.names) != len(names):
            raise ValueError('duplicate artifact basenames in comparison mapping')

    def hardware(self, link):
        path = self.hw_ir or Path(link["path"])
        if not path.is_absolute():
            path = self.root / path
        path = safe_path(path)
        if path not in self.hardware_cache:
            raw = path.read_bytes()
            normalized = _HIERARCHY_OUTPUT.sub(
                lambda match: match[1] + b'"' + match[2] + b'"' + match[3], raw
            )
            self.hardware_cache[path] = (hashlib.sha256(raw).hexdigest(), hashlib.sha256(normalized).hexdigest())
        digest, semantic_digest = self.hardware_cache[path]
        if digest != link["sha256"]:
            raise ValueError(f"{path.name}: recorded hw_ir sha256 {link['sha256']} does not match source {digest}")
        return {**link, "path": "<hardware-ir>", "sha256": semantic_digest}

    def path(self, name):
        name = Path(self.names.get(str(name), name))
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"artifact name must be relative: {name}")
        direct = safe_path(self.root / name)
        alternate = safe_path(self.root / "system" / name.name)
        if direct.is_file():
            return direct
        if len(name.parts) == 1 and alternate.is_file():
            return alternate
        raise ValueError(f"missing artifact {name}")

    def load(self, name):
        path = self.path(name)
        if path in self.cache:
            return self.cache[path]
        if path in self.active:
            raise ValueError(f"cyclic artifact provenance: {name}")
        self.active.add(path)
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
            normalized = self.normalize(document)
            self.cache[path] = (hashlib.sha256(raw).hexdigest(), normalized)
            return self.cache[path]
        finally:
            self.active.remove(path)

    def link(self, link):
        if not isinstance(link, dict) or not isinstance(link.get("path"), str) or not isinstance(link.get("sha256"), str):
            raise ValueError("invalid artifact provenance link")
        name = Path(link["path"]).name
        # Pin local dependency bytes, ignoring embedded absolute source paths.
        digest, source = self.load(name)
        if digest != link["sha256"]:
            raise ValueError(f"{name}: recorded sha256 {link['sha256']} does not match source {digest}")
        if "schema" in link and link["schema"] != source["schema"]:
            raise ValueError(f"{name}: provenance schema does not match source")
        return {**link, "path": name, "sha256": hashlib.sha256(_json(source).encode()).hexdigest()}

    def normalize(self, document):
        if not isinstance(document, dict) or document.get("schema") not in SCHEMAS:
            raise ValueError("unknown timing artifact schema")
        result = dict(document)
        hardware = result.get("hw_ir")
        if not isinstance(hardware, dict) or not isinstance(hardware.get("path"), str) or not isinstance(hardware.get("sha256"), str):
            raise ValueError("invalid hw_ir provenance")
        result["hw_ir"] = self.hardware(hardware)
        schema = result["schema"]
        records_name = SCHEMAS[schema]
        records = result[records_name]
        if schema == "merlin.op_pairs.v1":
            key = lambda r: _json([r["first"], r["second"], r["relation"]])
            result["op_timing"] = self.link(result["op_timing"])
            result["sources"] = _indexed([self.link(link) for link in result["sources"]], lambda r: r["path"], "sources")
            result["capacity"] = _indexed(result["capacity"], lambda r: r["engine"], "capacity")
        else:
            key = lambda r: r["name"]
        result[records_name] = _indexed(records, key, records_name)
        if schema.startswith("rtl_extract."):
            for name, record in result[records_name].items():
                record = dict(record)
                record["results"] = _runs(record)
                if "rejected" in record:
                    record["rejected"] = _indexed(record["rejected"], lambda r: str(r["gap"]), f"{name}.rejected")
                result[records_name][name] = record
        return result

def _diff(generated, reference, where, differences):
    if type(generated) is not type(reference):
        differences.append(f"{where}: generated {generated!r}; reference {reference!r}")
    elif isinstance(generated, (dict, list)):
        is_list = isinstance(generated, list)
        left, right = (dict(enumerate(value)) if is_list else value
                       for value in (generated, reference))
        for key in sorted(left.keys() | right.keys()):
            label = f"{where}[{key}]" if is_list else f"{where}.{key}"
            if key not in left:
                differences.append(f"{label}: missing from generated")
            elif key not in right:
                differences.append(f"{label}: absent from reference")
            else:
                _diff(left[key], right[key], label, differences)
    elif generated != reference:
        differences.append(f"{where}: generated {generated!r}; reference {reference!r}")

def compare_artifacts(generated_dir, reference_dir, artifact_names, *,
                      generated_hw_ir=None, reference_hw_ir=None):
    """Compare all named/mapped files and pins, supporting legacy system/ roots."""
    mapping = artifact_names if isinstance(artifact_names, dict) else {name: name for name in artifact_names}
    generated = _Artifacts(generated_dir, generated_hw_ir, mapping.keys())
    reference = _Artifacts(reference_dir, reference_hw_ir, mapping.values())
    differences = []
    for name, reference_name in mapping.items():
        documents = []
        for side, artifacts in (("generated", generated), ("reference", reference)):
            try:
                documents.append(artifacts.load(name if side == 'generated' else reference_name)[1])
            except (ValueError, OSError, KeyError, TypeError) as error:
                differences.append(f"{name}: {side}: {error}")
                documents.append(None)
        if all(document is not None for document in documents):
            _diff(*documents, str(name), differences)
    return differences
