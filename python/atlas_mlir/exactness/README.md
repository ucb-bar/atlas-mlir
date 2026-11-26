# Exactness checker

Deploy's EXACT gate (`deployment_gate.exact`) runs `check_model.py` on the packaged bundle: it builds the generated host C as a shared library, runs it at the published addresses with every device launch executed by the pinned npu_model (b97b2f5, `numerics="rtl"`), compares the output bytes with the frozen expected output, and then runs a one-byte input mutation that must change an output. Staged deploys and `atlascc run OUT --simulate` run each step with the same `functional_runtime.py`.

```
python check_model.py --bundle OUT/bundle --out OUT/exact --receipt OUT/check.json \
  --atlas-root ATLAS_SUPPORT_FILES --qualification QUALIFICATION_DIR [--flip-offset N]
```

The gate supplies every argument: the interpreter is `ATLAS_MODEL_CHECKER_PYTHON`, `--atlas-root` is `toolchain.data_root()`, and `--qualification` is `ATLAS_CHECKER_QUALIFICATION`, the directory holding `qualification.json` (native dot) and `memory-qualification.json` (sparse memory). It hashes all of them, and this directory's files, before and after the run.

## Independence

The gate runs this file by path, never as `atlas_mlir.exactness`, so `atlas_mlir/__init__.py` and the compiler are not imported. The directory has no `__init__.py`; its modules import one another by name, and `sources.py` first loads each of them, and Atlas's decoder, from the files the gate hashed: `tools` becomes a package of `--atlas-root`'s `tools/` alone, so neither the import path's order nor a same-named module or package elsewhere can supply other code, and a name already loaded from another file is refused. The staged step workers (`chiprunner/composed_sim.py` and `ddr_sim.py`) bind the same way. Outside this directory the modules load only these declared interfaces:

- npu_model at the pinned revision, which `model_checkout_identity` and `authenticate_model` check before, during, and after execution.
- Merlin's `merlin.runtime.uniform_sparse_memory`, whose source hash the memory qualification binds.
- Atlas's instruction decoder, `tools/check_program_decode.py` (with `tools/check_variant_inventory.py`) and `tools/model_instructions.py`, and `docs/selected-variant-inventory.json`. They decode words only: each decoded program is re-encoded by npu_model and must equal the image bytes.
- numpy, torch, gcc, and git.

`test/test_exactness_imports.py` checks the import statements here; `test/test_exactness_isolation.py` runs the checker's and a step worker's binding and execution in a fresh interpreter with an import audit hook, and fails on a compiler module loaded at any depth (dynamic imports included) or a declared module loaded from another file. Outside a kit, the step worker's runner also reads its memory windows from the SoC declaration (`atlas_mlir_native.soc`); the checker does not.

## Files that must stay byte-identical

`qualified_native_dot.py` and `qualified_native_dot.c`: the native dot qualification records their SHA-256, with the compiler, Python, torch, and numpy it ran on, and `load_qualified_math` refuses any other bytes. Changing either needs a new qualification. `model_runtime.py` and `functional_runtime.py` are kept as they were qualified, except that `functional_runtime.py` also executes kits that declare `provider.imem_residency` (resident images; see `docs/host-launch.md`), each start bound to its published image and entry word; the npu_model modules they import are the set `authenticate_model` hashes into every execution record.
