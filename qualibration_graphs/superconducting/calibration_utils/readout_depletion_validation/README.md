# Resonator DRACHMA — Residual Photon Diagnostic

Backs the `23c_readout_depletion_validation` calibration node
(`calibrations/1Q_calibrations/23c_readout_depletion_validation.py`).

## The experiment

DRACHMA (Jerger et al., "Dispersive Qubit Readout with Intrinsic Resonator
Reset", arXiv:2406.04891) is a shaped readout pulse whose waveform — derived
from the resonator's kappa and its dispersive shifts (chi_ground,
chi_excited) — is designed to ring the intracavity field down to (near)
vacuum by the end of the pulse, without a separate depletion segment.

This node checks whether that cancellation actually works, by comparing the
residual photon population left in the resonator right after two "test"
operations:

1. **drachma** — the shaped DRACHMA readout pulse.
2. **no_operation** — nothing at all, as a ~vacuum reference.

A third, **readout** condition (a regular, unshaped readout pulse, as a
passive-ring-down baseline) can be added via
`Parameters.include_readout_baseline` (off by default).

Each condition is run with the qubit in both **ground** and **excited**
(DRACHMA's cancellation depends on chi_ground *and* chi_excited together, so
both states need checking). Immediately after the test pulse, a
zero-amplitude ("pure listening", no drive) probe pulse measures the
resonator's residual field, and the final plot compares `|IQ|` across
conditions per qubit/state — DRACHMA should decay fastest, using
`no_operation` as the zero-photon reference.

**The probe pulse is always a dedicated, zero-amplitude `SquareReadoutPulse`**
(`PROBE_OPERATION`, built fresh per qubit in `create_qua_program` and removed
again in `save_results` — it never reaches the saved QuAM state), regardless
of which conditions are enabled. Its length is `Parameters.probe_length`
(default 3000 ns), the same for every qubit. This keeps the probe identical
across conditions (so they stay directly comparable) and independent of
whichever pulse the `"readout"` operation on the resonator points at.

`InOutIQChannel.measure_sliced` (`qua.demod.sliced`) requires the probed
pulse's integration-weight duration to be an **exact** multiple of
`segment_length * 4ns` — no partial trailing segment. This is only enforced
by the real QM compiler at job-execution time, so the node checks up front
that `segment_length_ns` is a multiple of 4 and divides `probe_length`.

## State update

`qubit.resonator.depletion_time` is set to the ground-vs-excited depletion time of the
DRACHMA condition (clamped to at least `min_depletion_time_ns`, default 16 ns). A qubit that is
not depleted within the probe window gets a logged warning and `probe_length` as its depletion time.

## Prerequisites

- Calibrated readout (02a, 02b) and x180 pulse (03a, 04b).
- A `"readout_drachma"` operation (`DrachmaReadoutPulse`) on `qubit.resonator.operations`
  (name set by `Parameters.drachma_operation`). Qubits without it are skipped rather than failing
  the node: in multiplexed mode a single qubit playing an undefined operation would abort the shared
  real-time program and silently drop every qubit's results.

## Implementation

The probe measurement uses QuAM's **sliced dual demodulation**
(`InOutIQChannel.measure_sliced`) instead of streaming a full raw ADC trace:
it integrates the probe on-chip into a fixed number of equal segments
(`probe_length // segment_length_ns`, default 3000/300 = 10) and returns four QUA arrays
(`II, IQ, QI, QQ`) per shot — one per demod-weight/output-port pairing —
instead of one point per ns. This cuts the streamed data volume by orders of
magnitude and moves the IF demodulation on-chip.

The four arrays are streamed as-is (no on-chip combination, to avoid adding
QUA control flow mid-shot) and combined into `I = II + IQ`, `Q = QI + QQ` in
Python analysis — the same projection pairing QuAM's full dual-demod
`measure()` uses. Because the segment count and length are shared across qubits,
every qubit produces a uniform `(num_segments,)` result. The segment duration
is still carried as a per-qubit coordinate (`segment_length_ns`, all equal).

**Volts.** Each segment integrates only `segment_length_ns` of the probe, not
the pulse's full length — so `analysis.convert_sliced_demod_to_volts`
normalizes by `segment_length_ns` not by a full pulse length the way `qualibration_libs.convert_IQ_to_V`
does for a regular (non-sliced) measurement.

The QUA `stream_processing()` block also computes each shot's I/Q second
moments (`I_sq = (II+IQ)^2`, `Q_sq = (QI+QQ)^2`, each averaged over shots —
same element-wise stream-arithmetic technique
`08a_readout_frequency_optimization.py` uses for its own `I_g_sq`/`Q_g_sq`
streams). `process_raw_dataset` turns these into `IQ_abs_std`, the
amplitude's shot-noise std via first-order error propagation of
`amp = sqrt(I^2+Q^2)`, using only the diagonal `Var(I)`/`Var(Q)` terms. The
I-Q cross/covariance term (`2*I*Q*Cov(I,Q)/amp^2`) was also implemented and
tested against real run data (run #52, 5000 shots/condition): it contributed
under 1.5% (median) to the total variance for every `drachma`/`no_operation`
(qubit, state) signal, with no consistent sign across segments — consistent
with sampling noise in the `Cov(I,Q)` estimate itself rather than a real I/Q
correlation — so it was dropped as negligible rather than kept as permanent
streaming/compute overhead.

## Depletion-time tests

Two independent chi-squared (df=2) tests, both built on the same
two-sample-z-test machinery (`analysis._chi2_pvalue`, `_scan_depletion_time`):

- **`compute_stat_depletion_time`** — per segment, per (test_condition,
  qubit, state), a two-sample z-test of `mean_I`/`mean_Q` against the
  `no_operation` baseline, combined into `chi2_stat = z_I^2 + z_Q^2`. Both
  samples' own pooled-over-segments variance contribute to the standard
  error (a proper two-sample SE, not just the baseline's).
- **`compute_ge_depletion_time`** — same machinery, but compares `ground` vs
  `excited` directly within a condition (no `no_operation` reference at
  all): if they're statistically indistinguishable, no state information
  remains in the readout, independently confirming depletion.

Both assume `Cov(I,Q)=0` (see the covariance measurement above). Depletion
time = first segment where `p_value > node.parameters.alpha` holds for
`node.parameters.depletion_debounce_segments` (K) consecutive segments; `NaN`
(logged) if that's never reached within the measured window — "never reached
significance" is a meaningfully different outcome from "reached it exactly
at the last segment", so neither test falls back to the last segment, or to
the full readout length, in that case.

Results (`p_value`/`depletion_time_stat_ns`, `p_value_ge`/
`depletion_time_ge_ns`) are plotted by `plot_pvalue_grid`/
`plot_ge_pvalue_grid` and summarized to stdout by `log_depletion_summary`.

Not yet implemented: the spec's split-half validation (divide `no_operation`
into two halves, e.g. by shot parity, treat one as "fake drachma", and check
the resulting p-values are ~uniform on [0,1]) — this needs per-shot data,
but the current QUA program only streams shot-averaged moments. Documented
here as a follow-up, not a silent gap.

## Files

- **`parameters.py`** — `Parameters` (`num_shots`, `operation`,
  `drachma_operation`, `include_readout_baseline`, `probe_length`, `segment_length_ns`,
  `depletion_debounce_segments`, `alpha`, ...).
- **`analysis.py`** — `resolve_conditions` (the acquired/analysed condition
  tuples, from `parameters.include_readout_baseline` — call only from
  finalized parameters, see its docstring), `fetch_sliced_iq_traces` (pulls
  the `II/IQ/QI/QQ/I_sq/Q_sq` result handles per condition/state/qubit and
  stacks them), `process_raw_dataset` (volts conversion, stacking into
  `(condition, state, qubit, segment)`, I/Q combination, `IQ_abs`/`var_I`/
  `var_Q`, and shot-noise std `IQ_abs_std`), `compute_stat_depletion_time`
  and `compute_ge_depletion_time` (the two tests above, sharing
  `_chi2_pvalue`/`_scan_depletion_time`), and `log_depletion_summary`.
- **`plotting.py`** — `plot_drachma_residuals`: `|IQ|` vs probe segment with
  ±std error bars, one subplot per qubit, one trace per (condition, state)
  actually present in the dataset. `plot_pvalue_grid`/`plot_ge_pvalue_grid`:
  same grid layout, p-value (log scale) vs probe segment for the two
  depletion-time tests, sharing `_draw_pvalue_axis`.
- Multiplexing/core-assignment helpers (`build_batch_groups`,
  `assign_core_labels`) live in `calibration_utils/readout_drachma_common/batching.py`.
- **`__init__.py`** — public re-exports used by the node script.
