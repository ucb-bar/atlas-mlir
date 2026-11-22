#!/bin/sh
# Set up a laptop to run deployed kits on a board over USB; deploys run on BWRC over ssh (docs/laptop.md).
# usage: laptop-setup.sh HOST:DIR
#   HOST  your ssh destination on BWRC; DIR  the run bundle published there by laptop-publish.sh
# Optional environment: ATLASC_HOME (default ~/.atlasc), ATLASC_PYTHON (default python3), ATLASC_SERIAL (UART device).
# It builds nothing. macOS needs network access once, to install micromamba and the conda-forge OpenOCD.
set -eu
die() { echo "laptop-setup: $*" >&2; exit 1; }
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
case ${1:-} in *:/*) ;; *) sed -n '3,4p' "$0" >&2; exit 2 ;; esac
remote=${1%%:*} dir=${1#*:}
here=$(cd "$(dirname "$0")" && pwd -P)
home=${ATLASC_HOME:-$HOME/.atlasc}
mkdir -p "$home"
home=$(cd "$home" && pwd -P)

# The package reads its pins with tomllib when imported, so the run path needs Python 3.11.
python=$(command -v "${ATLASC_PYTHON:-python3}") || die "python3 not found; install Python 3.11 or newer"
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "$python is older than Python 3.11"
case $(uname -s) in
  Darwin) os=macos ;;
  Linux) if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then os=wsl; else os=linux; fi ;;
  *) die "on Windows, run this inside WSL2" ;;
esac
# Live runs check for other bench processes with pgrep and for the UART's users with fuser (macOS: lsof, built in).
if [ $os != macos ]; then
  for tool in pgrep fuser; do command -v $tool > /dev/null || die "$tool not found; live runs need pgrep and fuser: install procps and psmisc"; done
fi
parts='SHA256SUMS LINKS activate python reference'
[ $os = macos ] || parts="$parts openocd"

echo "Fetching the run bundle from $1"
rm -rf "$home/bundle.new" "$home/bundle.tar"
mkdir "$home/bundle.new"
# A bundle published from a source checkout also holds its pyproject.toml, the package's pin record.
ssh -n "$remote" "exec /bin/bash -c $(q "cd $(q "$dir") && tar -cf - $parts \$(test ! -e pyproject.toml || echo pyproject.toml)")" > "$home/bundle.tar" || die "could not fetch $1"
tar -C "$home/bundle.new" -xf "$home/bundle.tar" || die "could not unpack the bundle fetched from $1"
rm -f "$home/bundle.tar"
# Every fetched entry must be listed: files in SHA256SUMS with their digest, links in LINKS with their target.
# A source checkout's pin record, pyproject.toml, is the one optional entry.
"$python" -B "$here/laptop-bundle.py" check "$home/bundle.new" $parts $(test ! -e "$home/bundle.new/pyproject.toml" || echo pyproject.toml) \
  || die "the fetched bundle does not match its SHA256SUMS and LINKS"
rm -rf "$home/bundle"
mv "$home/bundle.new" "$home/bundle"

# Prints yes if the file is a declared OpenOCD build, else its sha256 (or missing).
declared_build() {
  PYTHONPATH=$home/bundle/python "$python" -B - "$1" <<'EOF'
import hashlib, os, sys
from atlas_mlir.chiprunner.preflight import openocd_builds
digest = hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest() if os.path.isfile(sys.argv[1]) else 'missing'
print('yes' if digest in {row['sha256'] for row in openocd_builds()} else digest)
EOF
}
if [ $os = macos ]; then
  # conda-forge builds OpenOCD 0.12.0 for osx-64 only; Apple silicon runs it under Rosetta.
  if [ "$(uname -m)" = arm64 ] && ! arch -x86_64 /usr/bin/true 2>/dev/null; then
    die "install Rosetta first: softwareupdate --install-rosetta --agree-to-license"
  fi
  openocd=$home/ocd/bin/openocd
  # Only a declared build is kept; a missing, partial, or not yet declared install is made afresh.
  if [ "$(declared_build "$openocd")" != yes ]; then
    rm -rf "$home/ocd"
    # micromamba 1.5.10 for osx-64 from conda-forge, by the digest its anaconda.org file listing records.
    url=https://conda.anaconda.org/conda-forge/osx-64/micromamba-1.5.10-0.tar.bz2
    sha256=3b5dc32ed12d964392a4cf869a3d5dd0aaa7f5fb917a7c0831c35ddd323438ea
    mkdir -p "$home/micromamba"
    curl -fsSL "$url" -o "$home/micromamba.tar.bz2" || die "could not download micromamba"
    [ "$(shasum -a 256 "$home/micromamba.tar.bz2" | cut -c1-64)" = "$sha256" ] || die "micromamba differs from its pinned sha256"
    tar -C "$home/micromamba" -xjf "$home/micromamba.tar.bz2" bin/micromamba || die "could not unpack micromamba"
    rm -f "$home/micromamba.tar.bz2"
    MAMBA_ROOT_PREFIX=$home/micromamba "$home/micromamba/bin/micromamba" create -y -p "$home/ocd" --platform osx-64 \
      --override-channels -c conda-forge 'openocd=0.12.0=h57f6e69_0' || die "could not install OpenOCD"
  fi
else
  openocd=$home/bundle/openocd/linux-64/bin/openocd
fi
declared=$(declared_build "$openocd")

serial=${ATLASC_SERIAL:-}
if [ -z "$serial" ]; then
  # The FTDI dual adapter's second channel (B, USB interface 01) is the UART; the first is JTAG.
  if [ $os = macos ]; then set -- /dev/cu.usbserial-*; [ $# -eq 2 ] && serial=$2
  else set -- /dev/serial/by-id/*FTDI*if01*; [ $# -eq 1 ] && [ -e "$1" ] && serial=$1
  fi
fi

{
  echo "ATLASC_HOME=$(q "$home") ATLASC_PYTHON=$(q "$python") ATLASC_REMOTE=$(q "$remote")"
  echo "ATLASC_ACTIVATE=$(q "$(cat "$home/bundle/activate")") ATLASC_SERIAL=$(q "$serial") NOV_OPENOCD=$(q "$openocd")"
  echo 'export NOV_OPENOCD'
} > "$home/env"
mkdir -p "$home/bin"
cat > "$home/bin/atlasc" <<'EOF'
#!/bin/sh
# atlasc on a laptop, written by laptop-setup.sh: `run` runs here; `deploy` runs on BWRC over ssh and copies OUT back.
set -eu
. "$(cd "$(dirname "$0")/.." && pwd -P)/env"
q() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }
on_bwrc() { ssh "$@" "exec /bin/bash -c $(q "cd \"\$(cd && pwd -P)\" && $command")"; }
usage() { echo 'usage: atlasc deploy EXPORT --jtag [--clock-mhz N] -o OUT | atlasc run OUT [--dry-run] [--negative] [--serial PATH]' >&2; exit 2; }
case ${1:-} in
  run)
    shift
    [ -z "$ATLASC_SERIAL" ] || set -- --serial "$ATLASC_SERIAL" "$@"
    PYTHONPATH=$ATLASC_HOME/bundle/python PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
    export PYTHONPATH PYTHONDONTWRITEBYTECODE PYTHONNOUSERSITE
    exec "$ATLASC_PYTHON" -m atlas_mlir.cli run "$@" ;;
  deploy)
    shift
    source= out= flags=
    while [ $# -gt 0 ]; do
      case $1 in
        -o) [ $# -ge 2 ] || usage; out=$2; shift 2 ;;
        --jtag) flags="$flags --jtag"; shift ;;
        --clock-mhz) [ $# -ge 2 ] || usage; flags="$flags --clock-mhz $(q "$2")"; shift 2 ;;
        -*) echo "atlasc deploy: $1 is not available from a laptop; deploy on BWRC directly" >&2; exit 2 ;;
        *) [ -z "$source" ] || usage; source=$1; shift ;;
      esac
    done
    [ -n "$source" ] && [ -n "$out" ] || usage
    [ -d "$source" ] || { echo "atlasc deploy: EXPORT must be a directory" >&2; exit 2; }
    [ ! -e "$out" ] || { echo "atlasc deploy: OUT must not exist yet" >&2; exit 2; }
    # Remote paths stay relative to the physical home directory: deploy refuses paths through symlinks.
    work=atlasc-remote/$(date -u +%Y%m%dT%H%M%SZ)-$$
    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT
    kept() { echo "atlasc deploy: $1; the remote files stay in $ATLASC_REMOTE:$work" >&2; exit "${2:-1}"; }
    # macOS tar also archives extended attributes, which GNU tar on BWRC reports as unknown header keywords.
    attributes=; [ "$(uname -s)" != Darwin ] || attributes=--no-xattrs
    COPYFILE_DISABLE=1 tar $attributes -C "$source" -cf "$tmp/model.tar" . || { echo "atlasc deploy: could not archive EXPORT" >&2; exit 1; }
    # The upload is not named export: deploy reads a directory named export or fx as part of an export.
    command="mkdir -p $work/model && tar -C $work/model -xf -"
    on_bwrc "$ATLASC_REMOTE" < "$tmp/model.tar" || kept "upload failed"
    rc=0
    # Deploy prints its OUT's path last; that remote line is dropped here, and the copy's path printed below.
    command="cd $work && . $(q "$ATLASC_ACTIVATE") && set -o pipefail && atlasc deploy model$flags -o out | o=\$PWD/out awk '\$0 != ENVIRON[\"o\"] { print; fflush() }'"
    on_bwrc -n "$ATLASC_REMOTE" || rc=$?
    # test -d exits 1 when deploy wrote no OUT; ssh itself fails with 255.
    command="test -d $work/out"
    on_bwrc -n "$ATLASC_REMOTE" && probe=0 || probe=$?
    case $probe in
      0)
        command="tar -C $work/out -cf - ."
        on_bwrc -n "$ATLASC_REMOTE" > "$tmp/out.tar" || kept "could not copy OUT back"
        mkdir "$out" && tar -C "$out" -xf "$tmp/out.tar" || kept "could not unpack OUT into $out"
        [ $rc -ne 0 ] || (cd "$out" && pwd -P) ;;
      1) [ $rc -ne 0 ] || rc=1; echo "atlasc deploy: the deploy wrote no OUT" >&2 ;;
      *) kept "could not check for the remote OUT" ;;
    esac
    command="rm -rf $work"
    on_bwrc -n "$ATLASC_REMOTE" || kept "cleanup failed" $((rc ? rc : 1))
    exit $rc ;;
  *) usage ;;
esac
EOF
chmod +x "$home/bin/atlasc"

echo "Checking the reference kit (dry run, no board)"
"$home/bin/atlasc" run "$home/bundle/reference" --dry-run || die "the reference kit's dry run failed"
echo
echo "Installed atlasc for $os in $home. Add it to your PATH: export PATH=$(q "$home/bin"):\$PATH"
echo "OpenOCD: $openocd"
[ "$declared" = yes ] || echo "This OpenOCD build is not declared yet (sha256 $declared), so live runs refuse; dry runs work. Send that sha256 and the first line of: $(q "$openocd") --version"
if [ -n "$serial" ]; then echo "UART: $serial"; else echo "UART: not found; set ATLASC_SERIAL in $home/env once the board is plugged in"; fi
case $os in
  macos) echo "If OpenOCD cannot open the adapter, macOS's FTDI driver may hold JTAG channel A; see docs/laptop.md." ;;
  *)
    echo "For USB access without sudo, install this udev rule once, then replug the board:"
    echo "  echo 'SUBSYSTEM==\"usb\", ATTR{idVendor}==\"0403\", ATTR{idProduct}==\"6010\", MODE=\"0660\", TAG+=\"uaccess\"' | sudo tee /etc/udev/rules.d/60-atlas-ftdi.rules"
    echo "  sudo udevadm control --reload-rules && sudo udevadm trigger && sudo usermod -aG dialout \"\$USER\""
    echo "On Arch the group is uucp, not dialout. Log in again for the group to take effect."
    [ $os = linux ] || {
      echo "WSL2 reaches the board through usbipd. In an administrator PowerShell on Windows:"
      echo "  usbipd list; usbipd bind --busid BUSID   (BUSID of the FTDI 0403:6010 device)"
      echo "Then, after each replug: usbipd attach --wsl --busid BUSID"
    } ;;
esac
