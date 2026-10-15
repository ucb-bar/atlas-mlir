#!/usr/bin/env python3
"""Bound svsim's hardcoded unlimited build parallelism to two jobs."""

import os
import sys

arguments = sys.argv[1:]
for index, value in enumerate(arguments):
    if value == "-j" and index + 1 < len(arguments):
        arguments[index + 1] = "2"
    elif value == "-j0":
        arguments[index] = "-j2"
    elif value == "-MAKEFLAGS" and index + 1 < len(arguments):
        arguments[index + 1] += " " + os.environ["ATLAS_MAKE_OVERRIDES"]
executable = os.environ["ATLAS_REAL_VERILATOR"]
os.execv(executable, [executable, *arguments])
