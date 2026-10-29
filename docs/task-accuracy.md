# Task accuracy

The full ResNet50, TinyLlama, and SmolVLA hosts are bit-exact against the Atlas arithmetic: every launch matches a reference that follows the MXU's FP8 E4M3 operands, power-of-two tensor scales, exact products, and BF16 rounding after every multiply-add in ascending K. That is the correctness claim for the compiler, runtime, and host code. It is not a claim that the outputs match the original models in PyTorch.

Because the arithmetic is coarser than PyTorch's, task outputs drift. On one input each:

| Model | PyTorch | Atlas |
| --- | --- | --- |
| TinyLlama, 128-token prompt | Tokens [278, 3711, 5652] | Tokens [278, 14263, 658] |
| SmolVLA policy | Reference actions | 20.4% action relative L2 error |
| ResNet50, church image | Class 497 | Class 562 |

ResNet50 matches PyTorch's top-1 on the three other images checked. An independent CPU emulation of the Atlas arithmetic reproduces all three results, so the gap comes from the number format rather than the compiler or host code. BF16 accumulation over long K dominates for TinyLlama; FP8 weight precision dominates for ResNet50; SmolVLA has both.

## Possible opt-in precision modes

The MXU always rounds to BF16 after each multiply-add, so higher precision has to come from software. Three recipes use only existing hardware interfaces and have been checked in offline emulation, not implemented:

- Finer scales: one power-of-two scale per activation row and per weight output column instead of per tensor, applied through prepacking and a host output factor. TinyLlama then produces PyTorch's tokens with no extra stores or launches. Per-row activation scales lower SmolVLA's error from 20.4% to 15.4%, but per-column weight scales raise it, so the choice has to be measured per model.
- K split: run a contraction as shorter K chunks, pop each raw BF16 partial without extra rounding, add the partials at higher precision on the host, and round to BF16 once. With 512-element chunks, SmolVLA's action error falls from 20.4% to 4.8%; with 32-element chunks, TinyLlama produces PyTorch's tokens. Cost: extra partial stores, host merge work, and more launches (2.3x for SmolVLA at K512 with the existing launcher) or a multi-partial program.
- Two-term weights: encode each weight as fp8(W) plus fp8 of its residual, each with its own power-of-two scale, run both matmuls on the same activations, and merge them in FP64 on the host. ResNet50 then classifies the church image as 497 and keeps PyTorch's top-1 on the other three images. Cost: twice the MXU matmul work and weight bytes.

Any such mode would be off by default, leave every default program and ELF byte-identical, and be checked bit-exact against its own declared arithmetic. These are single-input results; a task accuracy claim needs a dataset evaluation.
