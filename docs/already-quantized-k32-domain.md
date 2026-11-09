# Already-quantized K32 partial domain

For selected model47 MXU1 arithmetic, every admitted K32 readout is positive
zero or finite normal BF16, with nonzero magnitude in `[2^-26, 2^32]`.
Reset starts this invariant; ordered BF16 readout/reseed preserves it for any K.
Padding is positive zero. Final exponent restoration has its separate domain;
an arbitrary external BF16 seed is outside this proof.

Admitted E4M3 normals are multiples of `g = 2^-18` after multiplication,
but MXU1 small-integer packing can leave that grid: the reset fixture
`50 d0 09 88` × `50 50 09 0a`, padded with zeros, reads `3681`.
Four continuation chunks can reach `3280` (`2^-26`). An exact dot-product
substitution would therefore change the selected arithmetic.

Let `h = 2^-26`, `U = 2^32`, and the incoming seed be zero or in `[h,U]`.
MXU1 uses `anchor = max(product exponents, seed exponent) + 7` and units
`q = 2^(anchor-31)`, aligning magnitudes toward zero. With at most 32 products
and one seed, the signed total is below `33*255*2^17 < 2^31`, so cannot wrap.
Product shifts are at most 51; nonzero seed shifts lie in `[-17,26]`.
If `q >= h`, every aligned value is a multiple of `q`. Otherwise, seeds of
magnitude at least `g/2` and products lie on the `h` grid; a smaller seed
retains its leading bit when truncated, and any nonzero product sum differs
from it by more than `g/2`. Every nonzero total thus has magnitude at least h.
Both ordinary RNE and selected small-integer packing retain that lower bound.
Alignment cannot increase magnitude: `U + 32*448^2` is below the upward RNE
midpoint at U. Exceptional packing is below `256*q <= 2^16`. This closes the
upper bound and excludes negative zero, subnormals, Inf, and NaN.

`test/test_quantized_k32_domain.py` checks all 239 admitted codes, 57,121
products, 116,895 seed-magnitude/anchor pairs, integer headroom, and rounding
boundaries. These lattice checks overapproximate 32-product cancellation;
the algebra establishes closure without changing compiler limits or weights.
