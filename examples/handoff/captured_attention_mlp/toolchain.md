# LLVM toolchain for the graph host build

`build_graph_mailbox.sh` compiles LLVM IR produced by torch-mlir. Clang 18
rejects that IR (`getelementptr inbounds nuw`), so build the host object with
LLVM/MLIR 23: [llvm/llvm-project](https://github.com/llvm/llvm-project) at
`a47bddccec30255619bb8c37fa59700e661d4e66` (`23.0.0git`). Merlin pins the same
commit as its `third_party/llvm-project` submodule:

```sh
git submodule update --init --depth 1 third_party/llvm-project
git -C third_party/llvm-project rev-parse HEAD   # must match the commit above
```

Then build Clang and MLIR with the X86 and RISCV targets as described in
Merlin's `docs/guides/llvm_toolchain.md` (about 13 GB for source, build and
install). An existing install of the same commit works too.

Pass the install's `bin` directory as the third argument:

```sh
bash examples/handoff/captured_attention_mlp/build_graph_mailbox.sh \
  "$MERLIN_ROOT" "$COMPILER_PYTHON" "$LLVM_INSTALL/bin" \
  "$SOURCE_LINALG" "$COMPILED_NATIVE" "$FRESH_OUT"
```

`$FRESH_OUT` must not exist yet. Check `"$LLVM_INSTALL/bin/clang-23" --version`
and that it can emit a RISC-V object. If the install is outside the Merlin
checkout, set `MERLIN_MLIR_INSTALL`, `MERLIN_CLANG` and `MERLIN_MLIR_TRANSLATE`.
The script links with `-fuse-ld=lld`, which picks up the system `ld.lld`
unless you point it elsewhere; record the linker version for board builds.

This only builds the host ELF around an already compiled Atlas program. A
board image also needs the bitstream's memory map and a cache maintenance
hook; see [host launch](../../../docs/host-launch.md).
