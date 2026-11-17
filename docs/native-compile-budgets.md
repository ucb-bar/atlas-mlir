# Native compilation work budgets

Atlas tools bound each named algorithm invocation to 100,000,000 counted work units and each process to 1,000,000,000 charged units. Units count record, candidate, edge, fixed-point, and reservation probe visits. A charge through a limit succeeds; the next charge refuses with the budget name, consumed work, requested charge, and limit. Exhaustion aborts compilation, including speculative allocation trials. It cannot select a fallback program. Under the limits, counters preserve algorithm and iteration order.

Standalone atlas-opt accepts --atlas-max-work and --atlas-max-total-work overrides. --atlas-report-work enables JSON statistics on stderr for tests and qualification. Production compilation uses tool defaults without injecting these flags. Compiler and solver implementations are bound by tool and provider hashes. Publications retain their existing format, and older publications can rebind. Pure-tool cache identities include the actual command and binary hash.

The process cap is calibrated against measured per-process maxima of 381,189,998 units for a refused compile and 243,989,659 units for an emitting compile, giving about 2.6 and 4.1 times headroom. These measurements do not establish a universal runtime bound.

Native buffer placement uses Merlin's allocate_buffers API with timeout_ms=None and rlimit=100000000 per query, including lexicographic minimization queries. Unknown results abort with the named buffer_solver_rlimit_per_query refusal. Solver versions and compiler identities must remain fixed when comparing results.

Native subprocesses retain a 600-second operational hang detector. Expiry is compile_error with an internal error diagnosis. Counted refusals raise CountedWorkExhausted, a resource_limit NativeCompilationError subclass; capacity-refusal catches must re-raise it, and re-wrapping must preserve it. The hang detector is excluded from output identities because expiry aborts rather than changing the selected program.

Merlin's generic selector still uses wall_timeout_s and solver_timeout_ms. Its remaining wall limits abort rather than redirect selection, so host load can turn an emit into search_timeout but cannot change emitted bytes. The selector allocator has no supported deterministic resource-limit argument.

Existing spiller rewrite, scalar proof, loop replay, DMA disjunct, delay insertion, and scheduler idle limits remain separate. Their implementations and defaults are bound by tool and provider hashes. A rewrite refusal names its pass option and consumed allowance, for example budget=spill-atlas-virtual.max-spills, limit=256, used=256, requested=1.

The structural entry-control-arguments limit=256 refusal is also classified as counted work because its diagnostic includes budget=.
