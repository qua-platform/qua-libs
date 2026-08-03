# XY–Z Delay Calibration

[`16a_xyz_delay.py`](../../../../../calibrations/1Q_calibrations/16a_xyz_delay.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Measures and corrects the relative timing offset between a qubit's XY (drive) and Z (flux) control lines, so a flux pulse played concurrently with a drive pulse actually overlaps it in time on the qubit.

## Purpose

The XY drive and Z flux-bias lines are physically distinct signal chains from the OPX to the chip: the drive line is up-converted through an IQ mixer or Octave and carries a microwave tone, while the flux line is a DC-coupled baseband path through its own cabling, filtering, and bias-tee **[Kra+2019]**. These two chains generally have different net propagation delays, often by tens of nanoseconds. Any protocol that plays an XY pulse and a Z pulse *concurrently* — using the flux pulse to shift the qubit's instantaneous frequency during part of a drive rotation — implicitly assumes the two pulses arrive at the qubit at the same time. If they don't, part of the drive rotation happens at the wrong qubit frequency, and the rotation comes out incomplete or distorted. This matters directly for this repository's own flux-line-distortion nodes (`17_pi_vs_flux_long_distortions`, `18_cryoscope`), which both co-time an XY pulse with a Z pulse and rely on that alignment to correctly attribute a measured effect to the flux pulse's timing rather than to the alignment error itself.

This node measures the timing offset directly rather than inferring it: it bakes the XY drive pulse and a same-duration flux pulse together at 1 ns resolution, scans their relative timing over a symmetric window, and looks for where the flux pulse's frequency shift begins to distort an otherwise-clean $x180$ rotation. Comparing the response for an initial $|e\rangle$ preparation (which is sensitive to the timing-dependent detuning during the $x180$) against an initial $|g\rangle$ preparation (an idle wait of the same duration, used as a baseline) isolates that timing-dependent distortion from any static offset. The fitted quantity is not a frequency or amplitude but a pure digital delay, so no cosine/Rabi/Ramsey lineshape physics applies here — the analysis is a geometric cross-correlation problem (see Mechanism), and the node's own docstring attributes the fitting approach to Chen's PhD thesis (p. 108), not to any of the broader superconducting-qubit review literature.

## Mechanism

For each qubit, the node first bakes `2 * zeros_before_after_pulse` distinct combined XY+Z waveform segments (`baked_flux_xy_segments`), one for every possible 1 ns relative shift across the scan window: each segment plays a rectangular flux pulse (amplitude `z_pulse_amplitude`, duration equal to the qubit's `x180` length) shifted by `i` nanoseconds relative to a zero-padded copy of the `x180` I/Q waveform, using `qualang_tools.bakery` with symmetric-left padding.

For each averaging shot, for each of two initial-state preparations, for each of the baked relative-shift segments:

1. Reset the qubit using `qubit.reset(reset_type, simulate)` — this node **honors** `reset_type` (`"thermal"`/`"active"`/`"active_gef"`), unlike some other nodes in this library that hardcode thermal reset regardless of the parameter.
2. Prepare $|e\rangle$ by playing `x180`, or prepare $|g\rangle$ by waiting the same duration (`qubit.xy.wait(...)`) — this gives a matched-duration, unexcited baseline for the same relative-shift segment.
3. Wait a coarse pre-delay of `zeros_before_after_pulse // 4` clock cycles. This integer division by 4 is hardcoded, converting the ns-scale padding built into every baked segment into clock cycles; it is not itself a tunable parameter.
4. Dispatch the correct baked segment for the current relative shift via a QUA `switch_`/`case_` on the `segment` loop variable, and run it (playing the co-timed flux + XY waveform pair).
5. Measure the resonator (`I`/`Q`, or discriminated `state` if `use_state_discrimination=True`).

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/xyz_delay/analysis.py`):

1. Convert I/Q to volts if not using state discrimination, and compute `difference = data(init_state="e") - data(init_state="g")` at every relative-time point.
2. Fit `difference` vs. relative time to a `triangle_peak` model — `amp * max(0, 1 - |t - t0| / half_width) + offset` — which is the expected shape of a cross-correlation between two equal-duration rectangular pulses, not a Gaussian or Lorentzian.
3. Accept the fit only if it passes four checks: the peak center sits at least one `x180` duration away from both scan edges, the fit's timing uncertainty is below 5 ns, the fitted amplitude is positive, and the peak's signal-to-noise ratio (amplitude over the standard deviation of the "wings" far from the peak) exceeds 3. If any check fails (or the fit itself raises), the node falls back to a raw argmax-of-signal estimate and marks that qubit `success=False`.

> **Baseline handling differs by acquisition mode.** `difference -= difference.mean()` is applied only when using raw I/Q (`data == "I"`); it is skipped when `use_state_discrimination=True`, since discriminated `state` is already a bounded 0/1 quantity rather than an arbitrary-offset voltage. The peak-fitting heuristics (10th-percentile floor for `amp_guess`/`offset_guess`) are shared between both modes regardless of this difference.

## Prerequisites

- A calibrated $x180$ pulse for the qubit (the node's own docstring states this explicitly) — the baked segment length and the fit's edge/SNR checks are both derived from the *currently configured* `qubit.xy.operations["x180"].length` at run time, so re-calibrating the $x180$ duration after running this node invalidates the fit.
- (Optional) IQ-blob calibration (`07_iq_blobs`) if `use_state_discrimination=True`.
- Basic bring-up (mixer/Octave calibration, time of flight, readout, rough qubit frequency) — this node is not wired into either bring-up graph (`80_.../81_...`), so it is run manually once those are in place.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 50 | – | Averages per (initial-state, relative-time) point. | Lowers noise on the `difference` trace at each point; the main lever if the fit's SNR check keeps failing. |
| `zeros_before_after_pulse` | `int` | 60 | ns | Zero-padding on each side of the baked XY+Z pulses; the total scanned relative-time range is `2 ×` this value, and it directly sets how many baked segments are generated. | Must be wide enough that the true delay lies at least one `x180` duration inside the window (see the fit's edge-distance assert); too narrow causes the "peak too close to edge" failure mode. Larger values linearly increase baking time and the number of QUA `switch_`/`case_` branches. |
| `z_pulse_amplitude` | `float` | 0.1 | V | Amplitude of the flux pulse used to detune the qubit during the scan. | Must be large enough to produce a detectable population difference between $|e\rangle$ and $|g\rangle$ preparations at pulse overlap; too small starves the fit's SNR check (Parameter Tuning Heuristics #1). |

## Outputs

**Measured:** `I`/`Q` (or discriminated `state`), and the derived `difference = data(e) - data(g)`, at every (initial-state, relative-time) point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `flux_delay` | ns | ✅ | Relative timing offset between the XY and Z pulses, from the fitted triangle-peak center (or an argmax fallback if the fit failed). |
| `flux_delay_std` | ns | – | Fit uncertainty on `flux_delay` (`NaN` on the argmax fallback path). |
| `success` | bool | – | Gates whether `flux_delay` is applied to state (see State Updates). |

**Success criterion:** all four checks in `fit_delay_trace` pass — peak at least one `x180` duration from both scan edges, fit uncertainty `< 5` ns, fitted amplitude `> 0`, and SNR `> 3` — checked per qubit inside a `try`/`except` with an argmax fallback on any failure.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.z.opx_output.delay` | fitted `flux_delay` (rounded to `int` ns) | **increment (`+=`)** | outcome successful |

`opx_output.delay` is a port-level integer-nanosecond output delay (`quam.components.ports.analog_outputs.LFAnalogOutputPort.delay`, default `0`) applied to *every* pulse played on that Z port when the QOP configuration is generated — not just to this node's own baked calibration waveform. Because the update is additive rather than a replace, this has the same "repeated runs compound" behavior documented for `qubit.z.joint_offset` in `03b_qubit_spectroscopy_vs_flux`: re-running this node on an already-well-aligned qubit keeps adding whatever small residual the fit finds, rather than converging to a fixed value (see Troubleshooting #1).

## Troubleshooting

1. **`qubit.z.opx_output.delay` keeps changing by small amounts on successive re-runs of an already-aligned qubit** → this is expected, not drift: the update is additive (`+=`), not a replace. Check the currently configured `opx_output.delay` before re-running, the same way a joint-flux-point qubit's `joint_offset` must be checked in `03b_qubit_spectroscopy_vs_flux`.
2. **Fit degrades specifically when run with `multiplexed=True` but is clean with `multiplexed=False`** → concurrently baked flux pulses on other qubits' Z lines can crosstalk onto this qubit during the scan. Re-run the same qubit alone to confirm, then treat it as a crosstalk-compensation issue rather than a delay-calibration bug.
3. **Fits look worse than expected after switching `reset_type` to `"active"` or `"active_gef"`** → unlike some other nodes in this library, this node *does* honor `reset_type` — an uncalibrated active-reset method will silently degrade the $|g\rangle$/$|e\rangle$ contrast this fit depends on. Verify the active-reset calibration first, or fall back to `"thermal"`.
4. **A previously-good `flux_delay` fit stops making sense after changing the $x180$ pulse length or amplitude** → both the baked-segment duration and the fit's edge/SNR checks are computed from `qubit.xy.operations["x180"].length` *at run time*. Re-run this node after any change to the $x180$ calibration.

## Parameter Tuning Heuristics

1. **Fit fails outright, or the `difference` trace looks flat/noisy with no visible triangle** → `z_pulse_amplitude` (default 0.1 V) is likely too small to produce a detectable detuning-driven population difference between the $|e\rangle$ and $|g\rangle$ preparations. Increase it and re-check the raw trace before touching anything else.
2. **Fit fails with the peak reported very close to a scan edge** → the true delay falls close to or outside `± zeros_before_after_pulse`. Widen `zeros_before_after_pulse`; remember this both extends the scan range *and* proportionally increases the number of baked segments and total run time.
3. **Fit succeeds but `flux_delay_std` is large, or the fitted value is inconsistent across repeated runs at the same settings** → the SNR is marginal even though it cleared the `> 3` threshold. Increase `num_shots` before concluding the line has genuinely drifted.
4. **You changed `z_pulse_amplitude` and now the previously-triangular `difference` trace looks asymmetric or has a shoulder** → a large flux pulse can shift the qubit far enough that the $x180$ drive itself becomes partially off-resonant even where the two pulses do overlap, distorting the expected triangle-of-cross-correlation shape assumed by `triangle_peak`. Reduce the amplitude back toward the point where the trace is cleanly triangular.

## Next Steps

This node is not wired into the automated bring-up (`80_.../90_...`) or retuning (`81_.../91_...`) graphs — it is run manually, once a calibrated $x180$ pulse (and, optionally, IQ-blob calibration) is in place. Its output is a direct, explicitly-named prerequisite for `17_pi_vs_flux_long_distortions` ("Calibrated XYZ delay" in that node's own docstring), and the same underlying timing-alignment concern applies equally to `18_cryoscope`, which also plays a co-timed XY+Z sequence. Both of those nodes' filter-fitting outputs also warn that writing a digital flux-line filter adds a global port delay — see their own docs' Prerequisites sections — which in turn means `16a_xyz_delay` should be **re-run after** either of them writes a filter, to re-align XY/Z timing under the new, filtered flux response.

## References

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.
