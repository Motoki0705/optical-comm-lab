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

`spatial.py` owns a separate scalar paraxial CW mechanism with two transverse
coordinates. `SpatialConfig`, `propagate_spatial`, the independent Gaussian and
LP01 limits, `power_current`, and `integrate_power_streamlines` are public APIs
under `optical_comm.visualization.spatial`. Fields are complex128, coordinates
are micrometres, and intensity has analytic input peak 1. These fields are not
the sqrt(W) temporal pulse or communications data above. CPU NumPy supports
small independent tests; requesting CuPy requires CUDA and never silently
falls back or changes precision.

`spatial_artifacts.py` owns compact section/flow export, physical isosurfaces,
bounded qualification, and saved-data audit. Extend the owning numerical
mechanism in `spatial.py`; put export/coordinate contracts beside the artifact
mechanism, and keep CLI and Blender scene logic out of both numerical APIs.
The circular model and its scope are documented in
[spatial-kerr.md](../../../docs/spatial-kerr.md).
