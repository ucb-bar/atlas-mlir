#!/usr/bin/env bash
# Publish the laptop run bundle for one installed atlasc (docs/laptop.md).
# usage: laptop-publish.sh ACTIVATE OUT DEST
#   ACTIVATE  the install's activate file: laptops deploy through it over ssh, and its atlas_mlir is the run bundle
#   OUT       a packaged deployment made with that install, published as the reference kit
#   DEST      a fresh directory to publish into
# NOV_OPENOCD must name a declared Linux OpenOCD build; its whole prefix is published for Linux laptops.
set -euo pipefail
[ $# -eq 3 ] || { sed -n '3,7p' "$0" >&2; exit 2; }
activate=$1 out=$2 dest=$3 python=${PYTHON:-python3}
[ -f "$activate" ] && [ -f "$out/deploy-status.json" ] && [ -d "$out/kit" ] || { echo "laptop-publish: need ACTIVATE and a packaged OUT" >&2; exit 2; }
[ ! -e "$dest" ] || { echo "laptop-publish: $dest exists; publish into a fresh directory" >&2; exit 2; }
[ ! -e "$dest.partial" ] || { echo "laptop-publish: $dest.partial is left from a killed or running publish; once none runs, remove it (chmod -R u+w $dest.partial; rm -rf $dest.partial) and rerun" >&2; exit 2; }
activate=$(cd "$(dirname "$activate")" && pwd -P)/$(basename "$activate")
check=$(mktemp -d)
# A failed or interrupted publish leaves nothing: cleanup first makes read-only copies writable.
trap 'chmod -R u+w "$check" "$dest.partial" 2>/dev/null || true; rm -rf "$check" "$dest.partial"' EXIT
python=$(bash -c '. "$1" >/dev/null && command -v "$2"' _ "$activate" "$python")  # the install's interpreter
package=$(bash -c '. "$1" >/dev/null && "$2" -B -c "import atlas_mlir, os; print(os.path.dirname(atlas_mlir.__file__))"' _ "$activate" "$python")
mkdir -p "$dest.partial"/{python,openocd,reference}
cp -R "$package" "$dest.partial/python/atlas_mlir"
# The package reads its pins when imported: an installed wheel from its dist-info, a source checkout from pyproject.toml.
if [ -f "$package/build-record.json" ]; then
  cp -R "$(bash -c '. "$1" >/dev/null && "$2" -B -c "import importlib.metadata as m; print(next(f for f in m.distribution(\"atlas-mlir\").files if f.name == \"METADATA\").locate().parent)"' _ "$activate" "$python")" "$dest.partial/python/"
else
  cp "$(dirname "$(dirname "$package")")/pyproject.toml" "$dest.partial/pyproject.toml"
fi
cp -R "$out/kit" "$out/deploy-status.json" "$dest.partial/reference/"
printf '%s\n' "$activate" > "$dest.partial/activate"
# The OpenOCD for Linux laptops is a declared build's relocatable prefix (it loads ../lib).
PYTHONPATH="$dest.partial/python" "$python" -B - "${NOV_OPENOCD:?NOV_OPENOCD must name a declared OpenOCD build}" <<'PY'
import hashlib, sys
from atlas_mlir.chiprunner.preflight import openocd_builds
digest = hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest()
if digest not in {row['sha256'] for row in openocd_builds()}:
    sys.exit('laptop-publish: NOV_OPENOCD is not a declared OpenOCD build: sha256 ' + digest)
PY
[ "$(basename "$NOV_OPENOCD")" = openocd ] && [ "$(basename "$(dirname "$NOV_OPENOCD")")" = bin ] \
  || { echo "laptop-publish: NOV_OPENOCD must be PREFIX/bin/openocd" >&2; exit 2; }
cp -R "$(dirname "$(dirname "$NOV_OPENOCD")")" "$dest.partial/openocd/linux-64"
chmod -R u+w "$dest.partial"  # the copies keep a read-only install's modes; the bundle's owner must be able to change it
find "$dest.partial/python" -name __pycache__ -prune -exec rm -rf {} +
# The bundled package alone must accept its reference kit: a dry run on a scratch copy.
cp -R "$dest.partial/reference" "$check/reference"
PYTHONPATH="$dest.partial/python" PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "$python" -m atlas_mlir.cli run "$check/reference" --dry-run
# SHA256SUMS lists every file with its digest and LINKS every link with its target, which must resolve inside its part.
"$python" -B "$(dirname "$0")/laptop-bundle.py" list "$dest.partial"
chmod -R a+rX "$dest.partial"
mv "$dest.partial" "$dest"
echo "published $dest"
