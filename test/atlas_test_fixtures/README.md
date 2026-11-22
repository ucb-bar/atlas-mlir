# Atlas integration fixtures

These source builders were adapted from `legacy-merlin/merlin/tests/ir/` and
`legacy-merlin/merlin/tests/runtime/` at Merlin commit
`77a9bb13112a47430cc1edf1d4a9dc5fc40bb3eb`.

Each module retains only helpers used by Atlas integration tests and their
helper dependencies. Test functions, test classes, toolchain fixtures, and
unrelated imports were omitted. Imports now use the current Merlin owners;
cross-fixture imports stay within this package. The module names correspond to
the original test module names with the `test_` prefix removed.

Atlas owns these fixtures so collection does not depend on Merlin's private
test layout. Preserve the source graph semantics when updating the builders.
