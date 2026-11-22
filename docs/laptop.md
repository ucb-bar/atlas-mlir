# Running kits from a laptop

A laptop runs `atlasc run` against a board plugged into its USB port. `atlasc deploy` still runs on BWRC: the laptop's `atlasc deploy` uploads the export over ssh, deploys it with the BWRC install, and copies `OUT` back, so the kit is the one BWRC would build. The compiler tools are Linux x86-64 binaries and the exactness checks bind their digests, so deploy cannot run on the laptop itself.

Status: on Nov 18 the macOS steps ran on one Apple silicon Mac with a board at 100 MHz: setup, laptop deploys over ssh (their ELFs matched the BWRC kits'), dry runs, the process and UART checks by hand, a dry connect (halt, then read the PC), and one live positive and negative pair each for the K32 and K64 Linear plus ReLU kits, with the same verdicts and output hashes as on the bench. That is one Mac and two small kits, nothing about timing or other models. Nothing has run on a Linux laptop or WSL2; the Linux setup path has run only in a clean environment on a BWRC host with ssh to that same host, whose `/bin/sh` is bash, so setup has not run under dash (Ubuntu's `/bin/sh`). The Linux and Windows steps below are the design, not results. The Linux laptop path will first run on a teammate's Linux machine outside BWRC.

## Setup

You need a BWRC login with ssh access, Python 3.11 or newer (macOS's own `python3` is 3.9: install a newer one, for example Homebrew's `python@3.12`, and set `ATLASC_PYTHON` to it), and the `HOST:DIR` of the published run bundle from your instructor. Then run, from an atlas-mlir checkout:

```sh
sh scripts/laptop-setup.sh you@HOST:DIR
export PATH="$HOME/.atlasc/bin:$PATH"
```

The script builds nothing. It fetches the bundle into `~/.atlasc` and refuses it unless `SHA256SUMS` lists every file with its digest, `LINKS` lists every link with its target, and each link resolves, through any links it reaches, to an existing entry without leaving its own part. It then installs OpenOCD, finds the board's UART, writes `~/.atlasc/env` (`NOV_OPENOCD` and `ATLASC_SERIAL`), installs the `atlasc` launcher, and runs `atlasc run` on the bundle's reference kit with `--dry-run`. Set `ATLASC_SERIAL` before running it, or edit `~/.atlasc/env` later, if it does not find the UART.

- macOS: it downloads a pinned micromamba (1.5.10, osx-64) and checks its sha256, then installs conda-forge OpenOCD 0.12.0 for osx-64; both run under Rosetta on Apple silicon. That build is declared in the runner's table, and setup keeps an installed OpenOCD only when its digest is a declared build, so a rerun keeps it and a partial install is replaced. It takes the second `/dev/cu.usbserial-*` device of the FTDI adapter (channel B) as the UART. On the one Mac so far the adapter appeared as `/dev/cu.usbserial-2100` and `-2101`, OpenOCD reached JTAG on channel A, and the live runs read the UART on `-2101`, so that choice held once. Setup ran there before the board was plugged in, so its own detection has not run with a board attached; the UART was set by hand in `~/.atlasc/env`. OpenOCD prints `libusb_detach_kernel_driver() failed with LIBUSB_ERROR_ACCESS, trying to continue anyway`; that warning is expected, and it opened the adapter without sudo or driver changes. If OpenOCD cannot open the adapter, macOS's FTDI serial driver may be holding the JTAG channel.
- Linux: it uses the bundle's copy of the declared Linux bench build of OpenOCD and the adapter's `/dev/serial/by-id/...if01...` link. It refuses unless `pgrep` (procps) and `fuser` (psmisc) are on PATH, since live runs use them. It prints a udev rule to install once with sudo and the command that adds you to `dialout` (`uucp` on Arch); the group takes effect after you log in again.
- Windows: run the script inside WSL2, where it follows the Linux steps. It prints the `usbipd` commands that attach the board to WSL2. These steps are unverified. Native Windows has no termios, so the runner cannot read the UART there.

Live runs accept only an OpenOCD build declared in the runner's [openocd-builds.json](../python/atlas_mlir/chiprunner/openocd-builds.json). Today it declares the Linux bench build and the conda-forge macOS build. The script reports the sha256 of a build that is not declared. Dry runs still work. Live runs refuse until that build passes a dry connect on the board, is declared, and the kits are redeployed.

## Use

```sh
atlasc deploy EXPORT --jtag -o OUT
atlasc run OUT --dry-run
atlasc run OUT
```

`atlasc run` here is the same command as on BWRC, with `--serial` taken from `~/.atlasc/env`. The laptop `deploy` accepts `--jtag`, `--clock-mhz`, and `-o`. It uses a scratch directory under your BWRC home, `atlasc-remote/`, removes it when done, and, like deploy on BWRC, ends a passing deploy by printing the path of `OUT`, here the laptop's copy. If the upload, the check for `OUT`, or the copy back fails, it exits 1 and leaves that directory in place, naming it, so you can copy `OUT` yourself. If only the removal fails, it keeps the copied `OUT`, names the directory left behind, and exits nonzero.

A live run's `openocd.log` has, besides the banner, two expected pairs of lines from `verify_image`: `Error: No working memory available. Specify -work-area-phys to target.` and `Warn : not enough working area available(requested 1112)`. The kit's OpenOCD configuration sets no work area, so OpenOCD verifies the loaded image by reading it back; the bench's Linux runs log the same lines, and the checker confirms the output bytes on its own. On macOS the libusb warning above is the only other expected message.

## Publishing a bundle

On BWRC, with `NOV_OPENOCD` set to the declared Linux build:

```sh
scripts/laptop-publish.sh INSTALL/activate OUT DEST
```

`OUT` is a packaged deployment made with that install. It becomes the reference kit. `DEST` must be new. The bundle holds the install's `atlas_mlir` package (standard library only on the run path), the reference kit, the OpenOCD prefix, the activate path that laptop deploys use, the package's pin record (its wheel's dist-info, or `pyproject.toml` from a source checkout), `SHA256SUMS`, and `LINKS`. It refuses an OpenOCD that is not `PREFIX/bin/openocd` and any link that does not resolve to an existing entry inside its part. The script refuses unless the bundled package alone passes `atlasc run` with `--dry-run` on a copy of the reference kit. A read-only install publishes: the bundle is made writable by its owner, whatever the modes of the files it copies. A publish that fails or is interrupted removes `DEST.partial`; after a killed publish, a rerun refuses and says how to remove what is left. Publish again after every install or OpenOCD table change, since kits stage the runner and its table.

[test_laptop_scripts.py](../test/test_laptop_scripts.py) runs publish, setup, and the laptop deploy over a local ssh stand-in and a synthetic kit. It qualifies no laptop.
