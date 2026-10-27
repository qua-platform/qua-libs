# Resonator DRACHMA — Kappa Calibration

Backs the `23f_readout_drachma_kappa_calibration` calibration node
(`calibrations/1Q_calibrations/23f_readout_drachma_kappa_calibration.py`).

## The experiment

Fine-tunes `kappa_ground_hz` and `kappa_excited_hz` of the DRACHMA readout pulse (Jerger et al.,
arXiv:2406.04891) by minimising the photon population left in the resonator right after the pulse. It is
the same measurement as the zeta calibration (`23e_readout_drachma_zeta_calibration`, see its README),
but the scanned knob is kappa. For every scan point (run for the ground and for the excited state):

1. Reset the qubit; for the excited state also play x180.
2. DRACHMA readout pulse built with that point's kappa (result discarded).
3. Immediately, a dedicated zero-amplitude probe pulse (`PROBE_OPERATION`, length
   `Parameters.probe_length`) whose integrated I/Q is saved. Its power `I^2 + Q^2`, shot-noise bias
   removed, is the residual-photon proxy that is minimised.

## Kappa scan

`kappa_num_points` points spaced by `kappa_step_hz`. Ground and excited are scanned in lockstep: point `i`
is one DRACHMA pulse with `kappa_ground[i]` and `kappa_excited[i]` set together, and both the ground and the
excited state are measured with it.

- Linear grid per state around its current kappa, which is on the grid (for an even number of points the
  extra point is above it) (`analysis.build_kappa_grid`).
- If the lower end would drop below `kappa_step_hz`, that grid is shifted so that `kappa_step_hz` is its
  minimum (kappa must stay positive); the current kappa is then no longer on the grid.

Every scan point is its own temporary operation on `qubit.resonator.operations` (a copy of the DRACHMA
pulse with the kappas changed), indexed by point, removed again in `save_results` — never persisted to
QuAM state. The current kappas are stored in the raw dataset (`current_kappa_*_hz`) for the flatness rule.

## Choosing the result

Each point measures **both** states with the same pulse (both kappas changed together), so the effect of one
kappa on the other state's residual is visible. The result is chosen per state with
`readout_drachma_common.select_points`:

1. **Flatness gate.** A chi-square test of "the residual power is constant over the scan" (inverse-variance
   weighted mean, `N - 1` degrees of freedom; `readout_drachma_common.flatness_pvalue`) is run for each
   state. `p > Parameters.flatness_alpha` (default 0.01) means the state does not respond to kappa: it keeps
   its current value and is not written to state.
2. **Both states respond:** they share one scan point, the one minimising the sum of the two noise-weighted
   excess costs. The excess cost of a point is `max(0, z - 1)^2` with
   `z = (P - P_min) / sqrt(err^2 + err_min^2)`, so points within about 1 sigma of the state's own minimum
   are free; ties go to the point nearest the current values (`readout_drachma_common.excess_power_cost`).
3. **Only one state responds:** that state takes its own lowest-cost point; the flat state keeps its
   current value.

A changed value on the edge of the scan is still written to state, with a warning (consider rescanning).

## Extra depletion check

Every shot also measures a single no-operation reference (reset, then the probe only, no DRACHMA pulse).
A z-test (chi-squared on the I/Q mean difference, 2 degrees of freedom) of each (state, point) against it is
computed (`readout_drachma_common.plot_noop_pvalue_vs_point`); `p > alpha` (`Parameters.alpha`, default 0.01)
means the probe field is indistinguishable from the reference, i.e. the resonator is depleted. It is logged
and plotted as `-log10(p)` (`-log10 p = chi2 / (2 ln 10)`, so it cannot underflow) on a fixed 0-100 axis. A
state that is never depleted at any scanned kappa gets a warning (a residual that stays far above the
no-operation level over the whole scan usually points to another mis-set parameter, e.g. chi or the
resonator frequency, rather than kappa), but the outcome is unchanged.

## State update

`kappa_ground_hz` and `kappa_excited_hz` of the DRACHMA readout pulse (power minima), only for a state whose
scan is not flat (`changed_<state>`). Failed qubits are skipped.

## Prerequisites

- Calibrated readout (02a, 02b) and x180 pulse (03a, 04b).
- A `"readout_drachma"` operation (`DrachmaReadoutPulse`) with kappa, chi and amplitude set, on
  `qubit.resonator.operations`; ideally with zeta tuned by `23e_readout_drachma_zeta_calibration`.
  Qubits without it are skipped.

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `kappa_num_points`, `kappa_step_hz`, `drachma_operation`,
  `probe_length`, `depletion_time`, `alpha`, `flatness_alpha`, ...).
- **`analysis.py`** — `build_kappa_grid`, `fit_raw_data` (per-state flatness gate and point selection),
  `log_fitted_results`.
- **`plotting.py`** — `plot_power_vs_kappa`: residual probe power vs kappa, ground and excited overlaid, with the
  no-operation level and the selected values labelled in the legend.
- Shared pieces (traces, `process_raw_dataset`, `plot_noop_pvalue_vs_point`, `plot_combined_objective_vs_point`,
  point selection, batching) live in `calibration_utils/readout_drachma_common/`.
