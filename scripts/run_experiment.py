#!/usr/bin/env python3
"""Compatibility adapter for ``python -m experiments.esn_longhaul.run_experiment``."""

import sys

from experiments.esn_longhaul import run_experiment as _implementation

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
else:
    sys.modules[__name__] = _implementation


def __getattr__(name):
    return getattr(_implementation, name)
