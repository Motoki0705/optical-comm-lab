# Transformer nonlinear-compensation reproduction

Reserved workspace for the next study. There is no model, simulation, training
entry point, or claimed result here yet. Shared propagation/DSP/metrics will be
imported from `optical_comm`; the ESN study remains an independent peer.

The execution roadmap is tracked by [parent Issue #40](https://github.com/Motoki0705/optical-comm-lab/issues/40):

1. Complete and verify the experiment-workspace refactor in
   [Issue #42](https://github.com/Motoki0705/optical-comm-lab/issues/42), including
   packaging, CLI compatibility, regression checks, and historical provenance.
2. Then implement the Transformer reproduction in
   [Issue #43](https://github.com/Motoki0705/optical-comm-lab/issues/43), recording
   its source method, scientific settings, data isolation, and acceptance criteria
   before experiments. The currently authorized scope is the main 16QAM/64QAM
   reproduction; this refactor does not choose new scientific settings.
3. Deliver [physics/data #44](https://github.com/Motoki0705/optical-comm-lab/issues/44),
   [model/mask #45](https://github.com/Motoki0705/optical-comm-lab/issues/45),
   [training/evaluation #46](https://github.com/Motoki0705/optical-comm-lab/issues/46),
   and [main runs/report #47](https://github.com/Motoki0705/optical-comm-lab/issues/47)
   as separately scoped work after the refactor. Report measured results,
   uncertainty and deviations with reproducible configs,
   checkpoints, figures and source/input hashes. PMD/rate expansion, CNN-LSTM
   comparisons and additional ablations require later consultation.

Use the [extension procedure](../README.md) when implementation starts. No
Transformer experiment runs as part of the structural refactor.
