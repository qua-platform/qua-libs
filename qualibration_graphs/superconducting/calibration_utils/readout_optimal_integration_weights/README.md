# Optimal Readout Integration Weights (OPX1000 MW-FEM)

Backs the `23d_readout_optimal_integration_weights` calibration node
(`calibrations/1Q_calibrations/23d_readout_optimal_integration_weights.py`).

## The experiment

The node measures the averaged complex ADC trace for the qubit in |g> and in |e>, demodulates it in
software, and builds the time-dependent matched-filter weight `W(t) = env_e(t) - env_g(t)` that best
separates the two states on the MW-FEM `dual_demod` readout (I gets the optimal weight, Q its
90-degree-rotated orthogonal partner).

Each qubit is measured with its own `"readout"` operation exactly as calibrated: readout length and
`resonator.smearing` are per-pulse properties and are never overridden here. Raw ADC traces are fetched
per qubit at each qubit's own native length and assembled by hand (not via `XarrayDataFetcher`, which
needs one common shape across qubits) into one dataset, NaN-padded on a shared `readout_time` axis sized
to the longest qubit's capture window.

## Weight construction

- **Normalisation.** The weight is normalised against the OPX fixed-point limits (2**-15 weight
  resolution, per-sample and cumulative demod-multiply limits) and chunked to the 4 ns weight grid
  (`weights.normalization_factor`, `signal_bound`, `chunk4`, `check_weight_limits`).
- **IQ imbalance** (`Parameters.correct_iq_imbalance`, `max_iq_imbalance`). A mirror-image model
  `y = x + b*conj(x)` is estimated from the ground-state trace and folded into the weight rather than
  applied to the data, so the single complex weight written to hardware (applied to the element's own
  IF-demodulated baseband of the RAW, uncorrected ADC) reproduces the imbalance-corrected matched filter
  (`weights.fold_iq_imbalance_into_weight`). An estimate above `max_iq_imbalance` indicates a failed fit,
  not real hardware imbalance, and falls back to no correction.
- **Smoothing** (`Parameters.smooth_bandwidth_hz`, off by default). A zero-phase Hann-windowed low-pass
  (`weights.lowpass_hann`) can be applied to `env_g`/`env_e` before the weight fit, well above any real
  resonator bandwidth, to suppress averaging jitter and residual carrier leakage (see
  `plot_weight_spectrum`). It does not touch the raw traces the overflow bound or the demod-phase-offset
  diagnostic rely on.
- **Variance weighting** (`Parameters.use_variance_weighting`). Divides the weight by the pooled
  per-sample variance. The measured per-sample variance also gives an absolute SNR for the
  constant-vs-optimal weight comparison (`weights.snr_for_weight`).

## Data volume

Every shot streams a full 1 ns-resolution trace (`save_all`, nothing is averaged on the OPX); the
per-sample mean and variance are computed on the PC in `execute_qua_program`. The node is therefore
limited by the OPX stream-processing bandwidth, not by acquisition time: too many shots, qubits or
streams make the OPX drop samples, which the cloud backend turns into an EMPTY result dict for the whole
job ("Data loss detected in data for job ..."). Hence the modest `num_shots` default and
`max_qubits_per_fem`; see the `Parameters.num_shots` docstring for the sizing formula.

## Limitation: complex weights

A stock quam `ReadoutPulse` can only express a REAL per-sample weight rotated by one GLOBAL phase
(`integration_weights_angle`). It cannot hold a weight whose phase drifts in time, which the matched
filter generally does while the g/e intracavity fields separate. Writing the computed weight therefore
needs a complex integration-weight form that quam_builder does not support yet. Until it does,
`update_state` writes only the weight AMPLITUDE (real, one value per 4 ns) and discards its phase, so the
deployed weight is an approximation of the matched filter, while the predicted SNR gain in the
diagnostics assumes the full complex weight. Review the diagnostics (does the predicted SNR gain over
today's constant weights justify this, and is the demodulation self-consistent) before relying on the
update.

## Plots

By default: predicted SNR, weight spectrum, envelopes, weight and a polar IQ trajectory. With
`Parameters.debug_plots`: variance, demodulation comparison and spectrum, and normalisation.

## State update

`q.resonator.operations["readout"].integration_weights`: `|W|` as a list of `(weight, 4 ns)` pairs
(phase discarded). Re-run `07_iq_blobs` afterwards: the existing threshold and `integration_weights_angle`
no longer match the new weights.

## Prerequisites

- Calibrated mixer or Octave (nodes 01a or 01b).
- Calibrated readout parameters (nodes 02a, 02b and/or 02c).
- Calibrated qubit x180 pulse (nodes 03a, 04b).

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `use_variance_weighting`, `max_qubits_per_fem`,
  `min_snr`, `debug_plots`, `correct_iq_imbalance`, `max_iq_imbalance`, `smooth_bandwidth_hz`, ...).
- **`weights.py`** — the numerical building blocks listed above (demodulation, envelopes, IQ imbalance,
  optimal weight, normalisation, fixed-point checks, SNR).
- **`analysis.py`** — `process_raw_dataset`, `fit_raw_data` (returns an `OptimalWeightsFit` per qubit),
  `has_variance`, `log_fitted_results`.
- **`plotting.py`** — the plots above.
