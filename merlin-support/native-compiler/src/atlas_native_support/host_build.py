"""Atlas EE290 host binding to Merlin's shared bare-metal build recipe.

The generic recipe owns invocation and source ordering. This module supplies
only the selected Atlas driver and EE290 software ABI facts. It neither reads
evaluation inputs nor executes or validates a target program.
"""

from __future__ import annotations

from pathlib import Path

from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe


def ee290_baremetal_recipe(
    *,
    compiler: Path,
    link_script: Path,
    specs: Path,
    driver_source: Path | None = None,
) -> HarnessBuildRecipe:
    """Return a build recipe for a compiled RISC-V host plus Atlas driver.

    The caller selects and pins the actual compiler, HTIF script and specs.
    This recipe is scoped to the diagnostic EE290SimConfig software environment.
    A reviewed platform binding must replace it for a deployed host or Zephyr.
    """
    compiler = Path(compiler).resolve(strict=True)
    link_script = Path(link_script).resolve(strict=True)
    specs = Path(specs).resolve(strict=True)
    driver = (Path(driver_source).resolve(strict=True) if driver_source is not None else
              (Path(__file__).parent / "runtime/atlas_host.c").resolve(strict=True))
    header = driver.with_suffix(".h")
    if not all(path.is_file() for path in (compiler, link_script, specs, driver, header)):
        raise ValueError("EE290 host build requires regular compiler, script, specs, and driver files")
    return HarnessBuildRecipe(
        compiler=compiler,
        include_roots=(driver.parent,),
        support_sources=(driver,),
        link_script=link_script,
        load_address=0x80000000,
        cflags=("-std=gnu99", "-O2", "-Wall", "-Wextra", "-fno-common",
                "-fno-builtin-printf", "-march=rv64imafd", "-mabi=lp64d",
                "-mcmodel=medany", "-static", f"-specs={specs}"),
    )
