# Physical field data for visualization

`pulses.py` provides deterministic isolated-pulse input, distance snapshots of
the existing public `propagate_fiber` SSFM, physical observables, analytic limits,
and qualified compact display export. Its stable array axes are
`(distance, time, polarization)`; fields remain in sqrt(W). These APIs are for
explanatory films and future field visualizations, with a separate manifest and
configuration. They do not change the main communication experiment or its
saved source hashes.

Extend a visualization observable beside its owning pulse/field mechanism.
Keep reusable numerical logic here, scene production in `scripts/`, named
settings in `configs/`, compact solver results in `artifacts/`, and the reader's
films/figures in `reports/`. Solver physics continues to be owned by
`channel.py`; do not implement another propagation engine in a scene script.
The documented contract and qualification scope are in
[kerr-physics.md](../../../docs/kerr-physics.md).
