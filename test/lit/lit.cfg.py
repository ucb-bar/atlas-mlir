import os
import lit.formats

config.name = "Atlas CFG DMA timing"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = lit_config.params.get("exec_root", config.test_source_root)
config.environment["PATH"] = lit_config.params.get("atlas_bin", "") + os.pathsep + os.environ.get("PATH", "")
