"""Compatibility import for the ESN study's equalizers and checkpoint API."""

import sys

from experiments.esn_longhaul import reservoir as _implementation

sys.modules[__name__] = _implementation
