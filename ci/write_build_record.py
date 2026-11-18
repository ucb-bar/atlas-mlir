"""CMake install hook: record the exact installed tools and their shared libraries."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--package', type=Path, required=True)
p.add_argument('--revision', required=True)
p.add_argument('--dirty', required=True)
p.add_argument('--llvm', required=True)
p.add_argument('--compiler', required=True)
a = p.parse_args()
files = {str(f.relative_to(a.package)): hashlib.sha256(f.read_bytes()).hexdigest()
         for directory in ('bin', 'lib', 'share')
         for f in sorted((a.package / directory).rglob('*')) if f.is_file()}
record = {'schema': 'atlas_mlir.wheel_build.v1', 'revision': a.revision,
          'dirty': a.dirty != '0', 'llvm': a.llvm, 'compiler': a.compiler, 'files': files}
(a.package / 'build-record.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
