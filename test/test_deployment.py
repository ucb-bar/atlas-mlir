"""Exercise internal deployment forwarding without invoking a compiler."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from atlas_mlir import cli, driver, provenance, toolchain


class DeploymentTest(unittest.TestCase):
    def build_commands(self, *options):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captured = root / "capture"
            captured.mkdir()
            (captured / "meta.json").write_text("{}")
            policy = root / "policy.json"
            policy.write_text("{}")
            snapshot = root / "snapshot"
            snapshot.mkdir()
            target = root / "target"
            bins = target / "build/bin"
            bins.mkdir(parents=True)
            for name in ("atlas-opt", "atlas-emit"):
                (bins / name).write_bytes(b"test tool identity")
            (target / "docs").mkdir()
            (target / "docs/selected-variant-inventory.json").write_text("{}")
            commands = {}

            def run(command, **kwargs):
                name = command[3]
                commands[name] = list(map(str, command))
                output = Path(command[command.index("--out") + 1])
                if name == "native-verify":
                    output.write_text("{}")
                else:
                    output.mkdir()

            with patch.object(driver.toolchain, "data_root", return_value=target), \
                    patch.object(driver.toolchain, "bin_dir", return_value=bins), \
                    patch.object(provenance, "environment_record", return_value={}), \
                    patch.object(driver, "_run", side_effect=run), \
                    redirect_stdout(io.StringIO()):
                status = cli.main([
                    "build", "--capture", str(captured), "--output", str(root / "out"),
                    "--precision-policy", str(policy), "--snapshot", str(snapshot),
                    "--application-memory-bytes", "1024", *options,
                ])
            self.assertEqual(status, 0)
            return commands

    def test_onchip_reaches_compile_and_verify_with_default_board(self):
        commands = self.build_commands("--application-deployment", "onchip")
        self.assertEqual(set(commands), {"stage-capture", "native-compile", "native-verify"})
        self.assertNotIn("--application-deployment", commands["stage-capture"])
        for name in ("native-compile", "native-verify"):
            command = commands[name]
            self.assertEqual(command.count("--application-deployment"), 1)
            self.assertEqual(command[command.index("--application-deployment") + 1], "onchip")
            self.assertNotIn("--board", command)

    def test_declared_deployments_place_dense_views_and_bench_builds_do_not(self):
        for options, views in ((("--application-deployment", "onchip"), True),
                               (("--application-deployment", "board"), True), ((), False)):
            commands = self.build_commands(*options)
            for name in ("native-compile", "native-verify"):
                self.assertEqual("--application-buffer-views" in commands[name], views, (options, name))

    def test_a_measured_host_reaches_the_onchip_compile_and_verify_only(self):
        commands = self.build_commands("--application-deployment", "onchip", "--measured-host-bytes", "113552")
        for name in ("native-compile", "native-verify"):
            command = commands[name]
            self.assertEqual(command[command.index("--measured-host-bytes") + 1], "113552")
        self.assertNotIn("--measured-host-bytes", self.build_commands("--application-deployment", "onchip")["native-compile"])
        for deployment in ("board", None):
            with self.assertRaisesRegex(ValueError, "a measured host bounds only an on-chip deployment"):
                driver.build(captured=Path("capture"), out=Path("out"), precision_policy=Path("policy.json"),
                             application_memory_bytes=1024, application_deployment=deployment, measured_host_bytes=1)

    def test_build_environment_records_the_placement_solver(self):
        def record(name):
            return {"version": "0.68.0" if name == "xdsl" else name + " version"}

        with patch.object(provenance, "unique_plugins"), patch.object(provenance, "build_record", return_value={}), \
                patch.object(provenance, "dependency_record", side_effect=record):
            dependencies = provenance.environment_record()["dependencies"]
        self.assertEqual(dependencies["z3-solver"], {"version": "z3-solver version"})

    def test_install_pins_match_provenance(self):
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]
        requirements = [*project["dependencies"], *project["optional-dependencies"]["capture"]]
        for name, suffix in (("merlin", ".zip"), ("m2m", "")):
            urls = [r.split(" @ ", 1)[1] for r in requirements if r.split("[")[0].split(" @ ")[0] == name]
            self.assertEqual(len(urls), 1, name)
            self.assertTrue(urls[0].endswith(provenance.PINS[name] + suffix), (name, urls[0]))

    def test_install_pins_parse_wheel_requirements(self):
        merlin = "https://example.invalid/merlin/archive/" + "a" * 40 + ".zip"
        m2m = "git+https://example.invalid/model2MLIR.git@" + "b" * 40
        wheel = ["numpy", f"merlin[semantic-search]@ {merlin}", f'm2m@ {m2m} ; extra == "capture"']
        self.assertEqual(provenance.install_urls(wheel), {"merlin": merlin, "m2m": m2m})
        for broken in (wheel[:2], [*wheel, wheel[1]], ["merlin @ https://example.invalid/merlin.zip", wheel[2]]):
            with self.assertRaises(toolchain.ToolchainError):
                provenance.install_urls(broken)


if __name__ == "__main__":
    unittest.main()
