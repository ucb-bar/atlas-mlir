# Selected-RTL column-reduction compatibility decision

For the executable compatibility contract bound to `atlas-npu`
`0079c0541111197741a231c002e3843fa6f545b2`, VPU BF16 column minimum
and maximum use the behavior observed on the selected RTL: a register pair is
read as **64 physical rows of 16 lanes**, one reduction is computed per lane,
and the 16 results are broadcast into both destination halves. The project
owner selected fidelity to the executing RTL on 2026-09-29 after reviewing the
discriminating standalone-core results for both operations.

The architectural description and inspected `npu_model-atlas`
`5bb08624d6bdc05ee5ea6e6f73b9c44c02f1459d` describe a 32-by-32 logical
tile with 32 separate column results. Those remain documented as intended
behavior and as a discrepancy; they are not substituted for the selected
RTL behavior in an executable program. See the [minimum](vpu-col-min-observation.md)
and [maximum](vpu-col-max-observation.md) observations for the source paths,
test vectors, object-word checks, and actual standalone-core outputs.

This selects the compatibility **layout and reduction extent** only. The two
bounded finite-normal panels per operation do not qualify all BF16 encodings,
arbitrary register pairs, timing, or integrated SoC behavior. The dialect's
`software_admitted` count remains zero and gate D remains blocked until those
obligations are checked. Any future corrected RTL or architectural mode needs
a separately identified target configuration and tests; it must not silently
change the meaning of this selected revision.

The later [column-sum observation](vpu-col-sum-observation.md) found the same
64-by-16 physical reduction and broadcast on two bounded panels. Its widened
serial binary32 additions and final upper-bit chop are separate numerical
requirements from the min/max comparisons. The inspected model's logical
32-by-32 BF16-sum routine remains a source discrepancy. This observation
extends the executable layout choice to column sum for this RTL revision;
it does not qualify its full numerical or timing domain.
