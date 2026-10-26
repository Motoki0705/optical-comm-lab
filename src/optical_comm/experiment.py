"""Compatibility import for :mod:`experiments.esn_longhaul.experiment`.

New studies should import the shared optical_comm modules and own their own
orchestration. The module alias preserves existing imports and monkeypatches.
"""

import sys

from experiments.esn_longhaul import experiment as _implementation

sys.modules[__name__] = _implementation
