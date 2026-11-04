"""Run LLVM's lit driver with portable local process workers."""

import multiprocessing
import runpy
import sys

# Fork workers need no local IPC listener, including with Python 3.14's
# forkserver default. Platforms without fork keep their usual start method.
if "fork" in multiprocessing.get_all_start_methods():
    multiprocessing.set_start_method("fork", force=True)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
