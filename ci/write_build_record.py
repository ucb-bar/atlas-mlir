"""CMake install hook: record the exact installed tools and their shared libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--revision', required=True)
p.add_argument('--dirty', required=True)
p.add_argument('--llvm', required=True)
p.add_argument('--compiler', required=True)
a = p.parse_args()
files = {}
for directory in ('bin', 'lib', 'share'):
    for root, dirs, names in os.walk(a.package / directory, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink())
        for name in sorted(names):
            f = Path(root) / name
            if f.is_file() and not f.is_symlink():
                files[str(f.relative_to(a.package))] = hashlib.sha256(f.read_bytes()).hexdigest()
record = {'schema': 'atlas_mlir.wheel_build.v1', 'revision': a.revision,
          'dirty': a.dirty != '0', 'llvm': a.llvm, 'compiler': a.compiler, 'files': files}
(a.package / 'build-record.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
