# Kerr film physical data

Computed illustrative pulse trajectories for [Issue #15](https://github.com/Motoki0705/optical-comm-lab/issues/15). The existing complex128 Manakov SSFM propagates a 50 mW **XY-total peak** Gaussian with intensity 1/e half-width 20 ps through 80 km of lossy fiber. No amplifier or ASE is applied. This is a separate deterministic demonstration, not a new 1600 km communications test.

| Case at 80 km | RMS time (ps) | RMS spectral width (GHz) | Raw XY peak (mW) | Center envelope phase (rad) |
| --- | ---: | ---: | ---: | ---: |
| pure_spm | 14.142136 | 8.253889 | 1.255943 | 1.223114 |
| dispersion_only | 62.937174 | 5.626977 | 0.282213 | -0.672079 |
| combined | 49.412961 | 4.798636 | 0.356625 | -0.015562 |

All 81 distance snapshots and full-window display fields are saved in the three NPZ files. `derived_metrics.json`/`summary.csv` use full solver precision; `qualification.json` records analytic, energy, step/grid/window, and display-export checks. `manifest.json` hashes numerical inputs and every artifact, declares units/conventions, and gives reproduction commands. `tests.json` and `tests.txt` retain ordinary verification.

`spectrum_xy_J_per_Hz` is pulse spectral **energy** density per polarization, not W/Hz. Orthogonal XY powers/spectra are summed separately. Raw fields retain 16 dB loss over 80 km; loss-compensated film views must be labeled. Phase/chirp NaNs are low-power mask values, not missing propagation. Optical carrier periods and fiber geometry in the films may be exaggerated; the saved complex envelope is not hand-authored. See [kerr-physics.md](../../docs/kerr-physics.md) for the stable data contract and scope.
