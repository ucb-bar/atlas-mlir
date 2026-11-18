"""Populate Cargo's cache and surface Rust build failures before running tests."""
import subprocess
import tempfile

from merlin.semantic_compiler.snapshot import packaged_egg_bridge_path


def main():
    manifest = ["--manifest-path", str(packaged_egg_bridge_path() / "Cargo.toml")]
    subprocess.run(["cargo", "--version"], check=True)
    subprocess.run(["rustc", "--version"], check=True)
    subprocess.run(["cargo", "fetch", "--locked", *manifest], check=True)
    with tempfile.TemporaryDirectory(prefix="merlin-bridge-") as target:
        # Match the snapshot build and let Cargo print its own diagnostics.
        subprocess.run(["cargo", "build", "--locked", "--offline", "--release",
                        *manifest, "--target-dir", target], check=True)


if __name__ == "__main__":
    main()
