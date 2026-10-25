# Shared controls and compatibility settings

The canonical ESN study settings are
[`pilot.json`](../experiments/esn_longhaul/configs/pilot.json) and
[`full.json`](../experiments/esn_longhaul/configs/full.json), documented in the
[study config README](../experiments/esn_longhaul/configs/README.md).
The two same-named paths here are filesystem symlinks, preserving existing CLI
arguments without maintaining duplicate live configurations. GitHub renders the
symlink target; follow the canonical links above to read the actual JSON.

`numerical.json` owns independent shared-channel qualification. The separate
Kerr controls below retain their existing purposes and paths. New study configs
belong with that study under `experiments/<study>/configs/` and must be named for
an interpretable purpose. Saved runs always retain their resolved config.

`kerr_films.json` fixes a separate deterministic isolated Gaussian pulse for
distance-resolved explanatory films: 50 mW XY-total **peak** power, T0=20 ps,
80 km of unamplified lossy fiber, and separate Kerr-only, dispersion-only and
combined cases. It uses the same Manakov solver with a 0.25 km step and a 2 ns
window at dt=0.244140625 ps. Its ordinary pulse qualification and display-data
contract are in [kerr-physics.md](../docs/kerr-physics.md). These parameters are
not an alternative training/test configuration for the 1600 km data stream.

`spatial_kerr.json` fixes a distinct circular-core CW experiment: physical core
radius 25 µm, two transverse coordinates, analytic input peak intensity 1,
and four retained normalized Kerr coefficients. It is an educational spatial
model, without a watt scale, glass-air boundary, communication BER or temporal
pulse. Numerical grid, artificial absorber, stored ROI/spacing, fixed 25 flow
seeds, intensity cutoff and three isosurface levels are versioned together.
Changing these settings requires a fresh output/scratch directory and a new
manifest; weak-reference absolute accuracy is not inferred from the stronger
cases or from visual similarity.
