# Resonator DRACHMA — Zeta Calibration

Backs the `23e_readout_drachma_zeta_calibration` calibration node
(`calibrations/1Q_calibrations/23e_readout_drachma_zeta_calibration.py`).

## The experiment

Fine-tunes `zeta_ground_hz` and `zeta_excited_hz` of the DRACHMA readout pulse (Jerger et al.,
"Dispersive Qubit Readout with Intrinsic Resonator Reset", arXiv:2406.04891) by minimising the
photon population left in the resonator right after the pulse. For every scan point the sequence is:

1. Reset the qubit; for the excited state also play x180.
2. DRACHMA readout pulse built with that point's zeta (result discarded).
3. Immediately, a dedicated zero-amplitude probe pulse (`PROBE_OPERATION`, length
   `Parameters.probe_length`) whose integrated I/Q is saved. Its power `I^2 + Q^2`, with the shot-noise
   bias `(Var(I) + Var(Q)) / num_shots` removed, is the residual-photon proxy that is minimised.

## Zeta scan

Per qubit and state, `zeta_num_points` points spaced by `zeta_step_hz`. Ground and excited are
scanned in lockstep: point `i` uses `zeta_ground[i]` for the ground-state run and `zeta_excited[i]` for
the excited-state run, the other zeta staying at its current value.

- Linear grid around the current zeta, which is on the grid (for an even number of points the extra
  point is above it): `zeta - ((zeta_num_points - 1) // 2) * step` upwards in steps of `zeta_step_hz`
  (`analysis.build_zeta_grid`).
- If the lower end would go below 0, the whole grid is shifted so that 0 is its minimum.

Every scan point is its own temporary operation on `qubit.resonator.operations` (a copy of the DRACHMA
pulse with only the zeta changed), indexed by state and point, and removed again in `save_results` —
never persisted to the QuAM state.

## Choosing the result

The two zetas are chosen with `readout_drachma_common.select_parameter_pair`, limited by
`Parameters.max_zeta_difference_hz` (`None` = no limit): among the allowed (ground point, excited point)
pairs the one with the lowest noise-weighted excess residual power above each state's own minimum is
taken. If a pulse would exceed full scale at its stored amplitude (`max_waveform_peak`), the state
is not updated and a message asks for a lower pulse amplitude.

## Extra depletion check

A ground-vs-excited z-test (chi-squared on the I/Q mean difference, 2 degrees of freedom,
`p = exp(-chi2 / 2)`) is computed at every point; no no-operation reference is used. `p > alpha` means
the two states leave indistinguishable probe fields, i.e. the resonator is depleted. It is logged and
plotted only.

## State update

`zeta_ground_hz` and `zeta_excited_hz` of the DRACHMA readout pulse (power minima).

## Prerequisites

- Calibrated readout (02a, 02b) and x180 pulse (03a, 04b).
- A `"readout_drachma"` operation (`DrachmaReadoutPulse`) with kappa, chi and amplitude set, on
  `qubit.resonator.operations` (name set by `Parameters.drachma_operation`). Qubits without it are
  skipped.

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `zeta_num_points`, `zeta_step_hz`,
  `max_zeta_difference_hz`, `max_waveform_peak`, `drachma_operation`, `probe_length`, `alpha`, ...).
- **`analysis.py`** — `build_zeta_grid`, `fit_raw_data` (per-qubit power minima and pair selection),
  `log_fitted_results`.
- **`plotting.py`** — `plot_power_vs_zeta`: residual probe power vs zeta, ground and excited overlaid.
- Shared pieces live in `calibration_utils/readout_drachma_common/`: the trace fetching and
  `process_raw_dataset` (`traces.py`), `plot_ge_pvalue_vs_point` (`plotting.py`), pair selection
  (`pairs.py`), waveform-peak helpers (`waveform.py`) and the batching helpers (`batching.py`).
