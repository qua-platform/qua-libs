# Resonator DRACHMA — Kappa Calibration

Backs the `23f_readout_drachma_kappa_calibration` calibration node
(`calibrations/1Q_calibrations/23f_readout_drachma_kappa_calibration.py`).

## The experiment

Fine-tunes `kappa_ground_hz` and `kappa_excited_hz` of the DRACHMA readout pulse (Jerger et al.,
arXiv:2406.04891) by minimising the photon population left in the resonator right after the pulse. It is
the same measurement as the zeta calibration (`23e_readout_drachma_zeta_calibration`, see its README),
but the scanned knob is kappa. For every scan point:

1. Reset the qubit; for the excited state also play x180.
2. DRACHMA readout pulse built with that point's kappa (result discarded).
3. Immediately, a dedicated zero-amplitude probe pulse (`PROBE_OPERATION`, length
   `Parameters.probe_length`) whose integrated I/Q is saved. Its power `I^2 + Q^2`, shot-noise bias
   removed, is the residual-photon proxy that is minimised.

## Kappa scan

Per qubit and state, `kappa_num_points` points spaced by `kappa_step_hz`, ground and excited scanned in
lockstep (point `i` uses `kappa_ground[i]` for the ground-state run and `kappa_excited[i]` for the
excited-state run, the other kappa staying at its current value).

- Linear grid around the current kappa, which is on the grid (for an even number of points the extra
  point is above it) (`analysis.build_kappa_grid`).
- If the lower end would drop below `kappa_step_hz`, the whole grid is shifted so that `kappa_step_hz` is
  its minimum (kappa must stay positive); the current kappa is then no longer on the grid.

Every scan point is its own temporary operation on `qubit.resonator.operations` (a copy of the DRACHMA
pulse with only the kappa changed), removed again in `save_results` — never persisted to QuAM state.

## Joint choice of the two kappas

The two kappas are chosen jointly, not independently: `|kappa_ground - kappa_excited|` must stay within
`Parameters.max_kappa_difference_hz` (default 200 kHz, `None` = no limit). Among all (ground point,
excited point) pairs inside the limit (`readout_drachma_common.select_parameter_pair`), the pair with the
lowest noise-weighted excess residual power above each state's own minimum is taken. A state that barely
reacts to kappa follows the other one at no cost; if both react but their minima are too far apart, the
pair trades off by benefit. A minimum on the edge of the scan is still written to state, with a warning
(consider rescanning); the fit fails only if no pair satisfies the limit.

## Amplitude limit

The DRACHMA waveform is normalised to a fixed area, so a larger kappa gives a peakier waveform. If any
scan point would exceed full scale (`max_waveform_peak`), all scan pulses of that qubit get one common
amplitude reduction (logged) so the points stay comparable with each other. The pulse amplitude
written to the QuAM state is never changed; if the chosen pulse would peak above 1 at its stored
amplitude, the state is not updated and a message asks for a lower amplitude.

## Extra depletion check

A ground-vs-excited z-test (chi-squared on the I/Q mean difference, 2 degrees of freedom) is computed at
every point (`readout_drachma_common.plot_ge_pvalue_vs_point`); `p > alpha` means the two states leave
indistinguishable probe fields, i.e. the resonator is depleted. It is logged and plotted only.

## State update

`kappa_ground_hz` and `kappa_excited_hz` of the DRACHMA readout pulse (jointly chosen power minima).

## Prerequisites

- Calibrated readout (02a, 02b) and x180 pulse (03a, 04b).
- A `"readout_drachma"` operation (`DrachmaReadoutPulse`) with kappa, chi and amplitude set, on
  `qubit.resonator.operations`; ideally with zeta tuned by `23e_readout_drachma_zeta_calibration`.
  Qubits without it are skipped.

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `kappa_num_points`, `kappa_step_hz`,
  `max_kappa_difference_hz`, `max_waveform_peak`, `drachma_operation`, `probe_length`, `alpha`, ...).
- **`analysis.py`** — `build_kappa_grid`, `fit_raw_data`, `log_fitted_results`.
- **`plotting.py`** — `plot_power_vs_kappa`.
- Shared pieces (traces, `process_raw_dataset`, pair selection, waveform helpers, batching) live in
  `calibration_utils/readout_drachma_common/`.
