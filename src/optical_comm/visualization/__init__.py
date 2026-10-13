"""Physically scaled field data for explanatory visualizations.

These APIs propagate illustrative pulses; they do not generate new held-out
communications scores or modify the main experiment's saved provenance.
"""

from optical_comm.visualization.pulses import (
    analytic_dispersion_gaussian,
    analytic_lossy_kerr,
    display_data,
    field_observables,
    gaussian_pulse,
    propagate_snapshots,
    qualify_pulse_trajectories,
    time_grid,
)

__all__ = [
    "analytic_dispersion_gaussian",
    "analytic_lossy_kerr",
    "display_data",
    "field_observables",
    "gaussian_pulse",
    "propagate_snapshots",
    "qualify_pulse_trajectories",
    "time_grid",
]
