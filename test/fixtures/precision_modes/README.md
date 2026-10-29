# Frozen default contraction cells

Each fixture retains one activation row and one weight column over the entire reduction, the scales of the original full tensors, and one independently archived output cell. BF16 source words and the promoted FP32 output use little-endian hexadecimal bytes. The data comes from the PR #26 run artifacts, not a forward pass through this package.

| Model | Archived site | K | Activation scale | Weight scale |
| --- | --- | ---: | ---: | ---: |
| TinyLlama | First prefill query projection, first panel, row 0, column 0 | 2048 | -6 | -8 |
| SmolVLA | First vision patch projection, row 0, column 0, before bias | 768 | -8 | -11 |
| ResNet50 | Church image final linear contraction, row 0, column 0, before bias | 2048 | -6 | -8 |

These cells exercise the shared default codec, scale application, full ordered reduction, and output bytes without model dependencies. They are deliberately small and do not substitute for the full default gates in the documented model commands. Those gates compare TinyLlama's three complete logit vectors and tokens, SmolVLA's 300 actions and trajectory fields, and ResNet50's 54 contractions and 1,000 logits.
