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
to the longest qubit's capture window. The temporary core labels the node assigns for the run are cleared
again before the state is saved.

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
- **Smoothing** (`Parameters.smooth_bandwidth_hz`, 10 MHz by default, `None` turns it off). A zero-phase
  Hann-windowed low-pass (`weights.lowpass_hann`) is applied to `env_g`/`env_e` before the weight fit, well
  above any real resonator bandwidth, to suppress averaging jitter, residual carrier leakage and the noise
  the weight would otherwise fit (see "Predicted SNR" below). It does not touch the raw traces the overflow
  bound or the demod-phase-offset diagnostic rely on.
- **Measuring the variance** (`Parameters.measure_variance`, off by default). See "Data volume".
- **Variance weighting** (`Parameters.use_variance_weighting`, off by default). Divides the weight by the
  pooled per-sample variance. It needs a measured variance: `use_variance_weighting=True` with
  `measure_variance=False` raises a `ValueError` (in the parameters, in the node actions and when replaying
  data acquired without a variance) instead of silently falling back.

| `measure_variance` | `use_variance_weighting` | result |
|---|---|---|
| False | False | default: averaged trace, flat noise, SNR in arbitrary units |
| True | False | absolute SNR, unweighted weight shape |
| True | True | absolute SNR, weight divided by the pooled variance |
| False | True | invalid, raises |

## Data volume

By default (`measure_variance=False`) only the OPX-averaged trace is streamed: the SNR is in arbitrary
units and `min_snr` is not enforced, but the ratios between weight shapes are exact for white noise.

With `measure_variance=True` every shot streams a full 1 ns-resolution trace (`save_all`); the per-sample
mean and variance are computed on the PC in `execute_qua_program`. The node is then limited by the OPX
stream-processing bandwidth, not by acquisition time: too many shots, qubits or streams make the OPX drop
samples, which the cloud backend turns into an EMPTY result dict for the whole job ("Data loss detected in
data for job ..."). Hence the modest `num_shots` default and `max_qubits_per_fem`; see the
`Parameters.num_shots` docstring for the sizing formula.

## Saved dataset

`ds_fit` has no complex variables, because NetCDF 3 (a fallback storage engine) cannot save them: `env_g`,
`env_e`, `W`, `W_norm` and `W_chunked` are stored as `<name>_re` / `<name>_im` float pairs. Read them back
with `complex_values(ds_fit, "W")`, which also accepts a dataset saved with the old complex variables.

## Real weight and the global angle

A stock quam `ReadoutPulse` can only express a REAL per-sample weight rotated by one GLOBAL phase
(`integration_weights_angle`). It cannot hold a weight whose phase drifts in time, which the matched
filter generally does while the g/e intracavity fields separate. So `update_state` writes the best real
weight for that format (`Parameters.deployed_weight`):

- `"signed"` (default): `w(t) = Re[W(t) * exp(-i*phi)]` with `phi = 0.5 * arg(sum W * D)`,
  `D = env_e - env_g` (`weights.signed_real_weight`). This is the optimal real weight for a real weight with
  one global angle. Unlike `|W|` it keeps the sign where the g/e difference points away from the global
  axis, so those stretches add separation instead of cancelling. It contains negative values.
- `"magnitude"`: `|W|`, the previous behaviour. Use it if the hardware rejects negative weights.

`phi` lives in the software demodulation frame (the time-of-flight/smearing phase is not in `omega_t`), so
it is not the hardware angle: `07_iq_blobs` still has to fit `integration_weights_angle` and the threshold.
The full complex matched filter stays in the diagnostics as the SNR ceiling.

When the IQ-imbalance correction is applied, `W` carries a fold term `-b*exp(-2i*omega*t)*conj(W)`: a copy
of the weight at `-2*f_IF` (marked in the weight-spectrum plot, aliased into the +/-500 MHz band). It is
expected, not a residual carrier. A real weight holds it only to first order, and the 4 ns weight grid
averages part of it away for high-IF qubits.

## Predicted SNR and success

`snr_for_weight` gives `|sum conj(W) D| / sqrt(sum |W|^2 var)` for four weight shapes: constant, `|W|`,
signed real and full complex. A weight built from the averaged difference `D` also fits the noise in `D`,
so scoring it against the same `D` overstates every shape except the constant one (for pure noise the
complex filter still scores `sqrt(sum |n|^2)`). The node therefore also runs its own weight construction
on seeded pure-noise differences (`weights.noise_only_snr2`; noise level from the measured variance, else
from the high-frequency residual of the difference, `weights.estimate_difference_noise_variance`) and
subtracts the noise-only expectation from SNR^2. The raw values are kept (`snr_*`), the corrected ones are
the `snr_*_net` fields and the bars in the SNR plot.

A qubit is marked successful when all of these hold, judged on the weight actually deployed (normalised,
chunked to 4 ns and fixed-point rounded) rather than on the complex ceiling:

- the three OPX overflow margins are within bounds;
- `snr_deployed_net >= min_snr` (only when the variance was measured);
- the deployed SNR^2 is at least `min_detection_sigma` standard deviations above its noise-only expectation
  (`detection_sigma`): without this a weight fitted to noise passes with a gain over the constant weight;
- `snr_gain = snr_deployed_net / snr_constant_net` is at least 1 and not above `max_snr_gain` (a huge gain
  means the constant-weight SNR is near zero, not a real improvement).

## Plots

By default, in reading order (data, weight, check, result): demodulated envelopes; the deployed weight at
1 ns and 4 ns with the unwrapped phase of the complex weight; the weight spectrum (carrier check, with
`+/-f_IF` and the `-2*f_IF` fold term marked); the predicted SNR of the four weight shapes (bars: noise-fit
bias removed, ticks: raw); and the I-Q trajectory of the mean g and e envelopes coloured by time (light to
dark, start and end marked). With `Parameters.debug_plots`: variance, demodulation comparison and spectrum,
and normalisation.

## State update

`q.resonator.operations["readout"].integration_weights`: the deployed real weight (signed by default) as a
list of `(weight, 4 ns)` pairs, for the qubits whose fit succeeded. The global angle is not written. Re-run
`07_iq_blobs` afterwards: the existing threshold and `integration_weights_angle` no longer match the new
weights.

## Prerequisites

- Calibrated mixer or Octave (nodes 01a or 01b).
- Calibrated readout parameters (nodes 02a, 02b and/or 02c).
- Calibrated qubit x180 pulse (nodes 03a, 04b).

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `measure_variance`, `use_variance_weighting`,
  `max_qubits_per_fem`, `min_snr`, `min_detection_sigma`, `deployed_weight`, `max_snr_gain`, `debug_plots`,
  `correct_iq_imbalance`, `max_iq_imbalance`, `smooth_bandwidth_hz`, ...) and `validate_variance_options`.
- **`weights.py`** — the numerical building blocks listed above (demodulation, envelopes, IQ imbalance,
  optimal and signed real weight, normalisation, fixed-point checks, SNR, noise-only SNR).
- **`analysis.py`** — `process_raw_dataset`, `fit_raw_data` (returns an `OptimalWeightsFit` per qubit),
  `has_variance`, `log_fitted_results`.
- **`plotting.py`** — the plots above.
