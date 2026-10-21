# Captured attention applications

`build_c_package.sh` compiles a completed Model2MLIR capture through Merlin's
application compiler. The output includes standalone C/H, Atlas images, constants,
and an ABI description. It does not build a host executable.

Use Merlin's `examples/workloads/attention_mlp/loader.py` for projected attention
and a two-layer MLP. Its `tiled_core_loader.py` exercises multiple tiles and tails
with query [33,65], keys [65,65], values [65,35], and MLP widths 67 and 29.
Both keep declared parameters separate from runtime inputs.

Follow the [C integration guide](../../../docs/baremetal-c-integration.md) for
capture, explicit precision settings, compilation, and Baremetal-IDE integration.
The script accepts the Merlin checkout, compiler interpreter, Atlas tools,
completed capture directory, and a fresh output directory.

Precision policies are diagnostic and must be selected explicitly. Report
original-model error separately from agreement with target-program arithmetic.
Model or standalone-core execution does not qualify an FPGA build.
