"""Shared opt-in rule for tests requiring external model evidence."""
import os
from pathlib import Path
import unittest

EVIDENCE_VARIABLES = ("ATLAS_TEST_OP_TIMING", "ATLAS_TEST_PAIR_TIMING")


def model_evidence_available():
    """Skip only an unconfigured environment; incomplete or missing pins fail."""
    if not any(name in os.environ for name in EVIDENCE_VARIABLES):
        return False
    for name in EVIDENCE_VARIABLES:
        value = os.environ.get(name)
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"{name} must name an existing evidence file: {value!r}")
    return True


def requires_model_evidence(test):
    return unittest.skipUnless(model_evidence_available(),
                               "set ATLAS_TEST_OP_TIMING and ATLAS_TEST_PAIR_TIMING for pinned evidence")(test)
