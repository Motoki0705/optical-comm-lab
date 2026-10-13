# Versioned experiment settings

`pilot.json` is a short exploratory end-to-end run. `full.json` is the qualified
held-out-study configuration: 4 samples/symbol, 1 km forward and DBP steps,
65,536 symbols/frame, and 1024 excluded symbols at each edge. The measured
scope and limits are in [numerical-qualification.md](../docs/numerical-qualification.md)
and the production recommendation in `artifacts/numerical-qualification/`.
Keep that resolved configuration with its run; do not relabel a pilot or reuse
an existing run after changing settings.

| Configuration | Symbols/frame | Train / validation / test frames | Context past/future; washout | Counted test bits/receiver/power |
| --- | ---: | --- | --- | ---: |
| pilot | 8,192 | 2 / 1 / 3 | 2 / 2; 64 | 182,688 |
| qualified full | 65,536 | 2 / 2 / 5 | 4 / 4; 128 | 2,534,080 |

Counts assume 256 excluded symbols at each edge for the pilot and 1024 for
the full run, with a contiguous valid frame. All four receivers share guards,
context exclusions, and washout.
`mode=final` requires at least one million configured test bits per power.
The pilot and full root seeds differ. Whole-frame bit and ASE seeds are derived
and saved by split, power, and index.

Both files describe the accepted 1600 km ideal DP-16QAM link and XY-total launch
powers. DBP uses 4 km steps in the pilot and 1 km in the full study.
These DBP settings are declared computational references, without an optimality
claim. CuPy is the default backend; `--backend numpy` is an explicit override
that changes run provenance and needs its own run ID.

The full grid has six 128-state ESN dynamics and four ridge strengths. The pilot
has 64/128-state candidates and three ridge strengths. Reservoir seeds are
derived once per power; candidates share the same underlying random draw where
their shape permits. Ridge paths reuse reservoir states and sufficient
statistics. See `docs/learning-methods.md` for the objective and tie rule.

Add a configuration only for a named, interpretable experimental purpose.
`ExperimentConfig.from_dict` rejects unknown fields and candidate-specified
seed/ridge values; the hierarchy and ridge path own those values. `--dry-run`
prints the resolved settings, bit budget, and declared future-context delay.

`kerr_films.json` fixes a separate deterministic isolated Gaussian pulse for
distance-resolved explanatory films: 50 mW XY-total **peak** power, T0=20 ps,
80 km of unamplified lossy fiber, and separate Kerr-only, dispersion-only and
combined cases. It uses the same Manakov solver with a 0.25 km step and a 2 ns
window at dt=0.244140625 ps. Its ordinary pulse qualification and display-data
contract are in [kerr-physics.md](../docs/kerr-physics.md). These parameters are
not an alternative training/test configuration for the 1600 km data stream.
