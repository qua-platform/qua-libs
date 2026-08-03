# Qubit Spectroscopy E to F

[`12_Qubit_Spectroscopy_E_to_F.py`](../../../../../calibrations/1Q_calibrations/12_Qubit_Spectroscopy_E_to_F.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Populates $|e\rangle$ with a calibrated `x180`, then sweeps a saturation drive near the anharmonicity-shifted $|e\rangle\leftrightarrow|f\rangle$ transition to locate $f_{12}$ and refine the qubit's anharmonicity.

## Purpose

A transmon is not a true two-level system but a weakly anharmonic oscillator: its $1\leftrightarrow2$ ("e"↔"f") transition frequency sits *below* the $0\leftrightarrow1$ ("g"↔"e") transition by the anharmonicity, $f_{12} = f_{01} + \alpha$ with $\alpha < 0$ **[Koc+2007]**. Because $|f\rangle$ has essentially zero thermal or drive-induced population starting from $|g\rangle$, there is no way to drive the e↔f transition directly from a qubit sitting in $|g\rangle$ — the transition simply doesn't exist as a two-level problem from that starting point. This node's first step is therefore not optional bookkeeping: it plays a calibrated `x180` to move the population to $|e\rangle$, and only then sweeps a drive near $f_{01}+\alpha$ to find the e↔f resonance, exactly mirroring `03a_qubit_spectroscopy`'s g↔e search but with an added state-preparation step in front.

The search itself is the same saturation-spectroscopy technique as `03a`: a continuous-style drive tone shifts population out of whichever state the qubit was prepared in, and — via the dispersive qubit-resonator coupling **[BGGW2021]** — that population shift changes the resonator's demodulated response at readout. The same power-broadening caveat from `03a`/`03b` applies here too: a stronger drive gives a bigger signal but also broadens and AC-Stark-shifts the observed line, so there is no substitute for sweeping `operation_amplitude_factor` deliberately if the line doesn't resolve cleanly **[AE1987]**, **[GRTW2021]**. There is an additional, EF-specific constraint on top of that: the drive pulse's spectral width must stay comfortably inside the anharmonicity, or the same tone can off-resonantly re-drive the g↔e transition it's supposed to be ignoring **[GRTW2021]** — worth keeping in mind when adjusting `operation_len_in_ns` (a shorter pulse is spectrally wider).

Finally, the node's own docstring flags a real, device-level failure mode worth repeating here: it's possible to excite the qubit via the image sideband or LO leakage rather than the intended drive sideband, especially with external mixers or the Octave. A spurious "peak" from this effect is indistinguishable at a glance from a genuine e↔f resonance, which is why the docstring recommends having mixer calibration done first.

## Mechanism

This node does not have its own `calibration_utils` package — it imports `Parameters`, `process_raw_dataset`, `fit_raw_data`, `plot_raw_data_with_fit`, and `log_fitted_results` directly from `calibration_utils.qubit_spectroscopy` (the same module `03a_qubit_spectroscopy` uses), unmodified. Every parameter, fit field, and plot in this node is therefore literally the g↔e-spectroscopy code, repointed at a different drive frequency by the QUA sequence below — nothing in the shared module itself is aware this node is looking for the e↔f line rather than the g↔e line. That has concrete consequences flagged throughout this page.

For each (drive-frequency detuning, qubit) point in the sweep, repeated `num_shots` times for averaging:

1. Wait `2 × qubit.thermalization_time` on `qubit.xy` — **not** `qubit.reset(reset_type, ...)`. This node never calls `qubit.reset()`; `reset_type` is declared (inherited from `QubitsExperimentNodeParameters`) but has **no effect** on the sequence. The wait is doubled relative to a plain thermal reset specifically to give any residual $|f\rangle$ population from the previous shot's EF drive time to decay too.
2. Reset `qubit.xy`'s intermediate frequency to its base value and play `x180` — populates $|e\rangle$.
3. Retune `qubit.xy` to `df − qubit.anharmonicity + intermediate_frequency`, i.e. center the swept detuning `df` on the qubit's *currently seeded* e↔f frequency, then play `operation` (default `"saturation"`) at that frequency, scaled by `operation_amplitude_factor` for `operation_len_in_ns` (or the pulse's own configured length).
4. `align()`, measure the resonator (`qubit.resonator.measure("readout", ...)`).

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/qubit_spectroscopy/analysis.py`, shared verbatim with `03a`):

1. Convert I/Q to volts, find the detuning where the signal deviates most from its mean, and use that point's I/Q to compute a rotation angle (`iw_angle`) that aligns the g/e-vs-f (here: e-vs-f) separation onto a single quadrature.
2. Search for the most prominent peak/dip in the rotated `I_rot` trace (`peaks_dips`, `prominence_factor=5`), returning `position` (peak location, relative to the swept center), `width` (FWHM), and `amp`.
3. From `position` and `width`, derive: the would-be absolute frequency (`res_freq = position + qubit.xy.RF_frequency`), the saturation amplitude that would give the configured `target_peak_width`, and the `x180`-equivalent amplitude implied by the same peak width.

> **The logged/plotted "frequency" is not the e↔f transition frequency — only `relative_freq` is trustworthy here.** The shared module computes `res_freq = position + qubit.xy.RF_frequency`, which is the correct absolute frequency *only* when the sweep is centered on `RF_frequency` (true for `03a`'s g↔e search). This node centers its sweep on `RF_frequency − qubit.anharmonicity` instead (step 3 above), so the true absolute e↔f frequency is `res_freq − qubit.anharmonicity`, not `res_freq` itself. This mislabeling propagates into the generated plot's "RF frequency [GHz]" x-axis too (`plot_individual_data_with_fit`, `calibration_utils/qubit_spectroscopy/plotting.py`, uses the same uncorrected `full_freq = detuning + RF_frequency`) — the axis tick values are systematically offset from the actual drive frequency by the (pre-fit) anharmonicity. Only `relative_freq` (= `position`, the in-window offset from the swept center) is used correctly, in the state update below.

> **`update_pulses_amplitude` and `target_peak_width` have no effect in this node**, despite being declared parameters that `03a_qubit_spectroscopy` actively uses to rescale `saturation`/`x180` amplitudes. Node 12's `update_state` (below) only ever writes `qubit.anharmonicity`; the `saturation_amplitude`/`x180_amplitude` fields the shared analysis module computes from `target_peak_width` are logged (via `log_fitted_results`) but never written to state here.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration`) — the docstring specifically calls this out, since image-sideband/LO-leakage excitation is easy to mistake for a genuine e↔f line.
- A calibrated `x180` pulse (`04b_power_rabi`) — this node's entire premise depends on reliably populating $|e\rangle$ first.
- Readout calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- A rough qubit frequency already found (`03a_qubit_spectroscopy` / `03b_qubit_spectroscopy_vs_flux`), since `qubit.xy.RF_frequency` seeds the sweep's frame of reference.
- A rough `qubit.anharmonicity` already present in state (from the device design / populate script, e.g. ~150–310 MHz on this system's example hardware) — the sweep is centered on `RF_frequency − anharmonicity`, so a seed that's off by more than half of `frequency_span_in_mhz` will place the true line outside the swept window entirely.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the (identical-to-`03a`) node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per (qubit, detuning) point. | Lowers I/Q noise; no effect on extracted frequency; linear cost in run time. |
| `frequency_span_in_mhz` | `float` | 100.0 | MHz | Full width of the drive-frequency sweep, centered on `RF_frequency − anharmonicity` (i.e. the seeded e↔f frequency, **not** `RF_frequency` itself). | Must be wide enough to cover the true anharmonicity's deviation from the seed value. |
| `frequency_step_in_mhz` | `float` | 0.25 | MHz | Step size of the frequency sweep. | Finer steps sharpen the peak position, at proportional run-time cost. |
| `operation` | `str` | `"saturation"` | – | QUAM pulse operation played on `qubit.xy` (after the `x180` state prep). | Selects the drive pulse used for the e↔f search itself. |
| `operation_amplitude_factor` | `float` | 1.0 | – (restricted to `[-2, 2)`) | Amplitude scale applied to `operation`. | Same value as `03a`'s default (no automatic reduction for the EF case) — see Parameter Tuning Heuristics #2 on power broadening. |
| `operation_len_in_ns` | `Optional[int]` | `None` (pulse's configured length) | ns | Overrides the duration of `operation`. | Shorter drives are spectrally wider — risk of off-resonantly re-driving g↔e (see Purpose). |
| `target_peak_width` | `float` | 3e6 | Hz | Target FWHM used to back-compute an "ideal" saturation amplitude for logging. | **Not used for any state update in this node** — see callout above. |
| `update_pulses_amplitude` | `bool` | `False` | – | Documented (for `03a`) as gating whether `saturation`/`x180` amplitudes get rewritten from the fitted peak width. | **Has no effect here** — this node's `update_state` never reads it. |

> **`reset_type` is declared but ignored.** Regardless of its value, this node always does a fixed `qubit.xy.wait(2 × thermalization_time)`; it never calls `qubit.reset()`. This mirrors the same pattern documented for `03b_qubit_spectroscopy_vs_flux`.

## Outputs

**Measured:** `I`/`Q`, `IQ_abs`, `phase`, `I_rot` at every (qubit, detuning) point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `relative_freq` | Hz | ✅ (via anharmonicity update) | Peak position relative to the swept center — the only frequency-domain fit value that is *not* mislabeled; this is what actually updates `qubit.anharmonicity`. |
| `frequency` (`res_freq`) | Hz | – | `relative_freq + qubit.xy.RF_frequency` — **do not read this as the e↔f transition frequency**; see the Mechanism callout above. |
| `fwhm` | Hz | – | Peak FWHM from `peaks_dips`. |
| `iw_angle` | rad | – | Rotation angle aligning the e/f separation onto a single quadrature (added to the qubit's existing readout `integration_weights_angle`, for logging only — not written back here). |
| `saturation_amp` | V | – | Saturation amplitude that would deliver `target_peak_width`; logged, not applied (see callout above). |
| `x180_amp` | V | – | `x180`-equivalent amplitude implied by the fitted width; logged, not applied. |

**Success criterion:** `_extract_relevant_fit_parameters` computes `freq_success = |res_freq| < frequency_span_in_mhz(Hz) + RF_frequency` and an analogous `fwhm_success`. Because `res_freq` (and `fwhm`) are bounded by construction to the swept axis, and `RF_frequency` is on the order of several GHz while the span is at most a few hundred MHz, **both of these checks are true by construction whenever a peak search returns any finite value at all** — they do not meaningfully gate on whether the peak actually looks sane. The check that actually matters is `saturation_amp_success = |saturation_amplitude| < instrument_limits(qubit.xy).max_wf_amplitude`: when `peaks_dips` fails to find any peak, `width`/`position` come back `NaN`, which propagates into a `NaN` `saturation_amplitude` — and `NaN < limit` is `False` in NumPy, so a genuinely peak-less scan is still (indirectly) reported as `"failed"`. In short: trust the outcome label for "no peak found," but don't read a `"successful"` outcome as proof the frequency-span check did anything.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.anharmonicity` | `anharmonicity − relative_freq` | replace (decrement) | outcome successful |

`qubit.f_12` (a separate QUAM field on `BaseTransmon`) is **not** touched by this node, and — as of the current codebase — isn't read anywhere in `calibrations/` or `calibration_utils/` either. Every downstream consumer (`13_power_rabi_ef`, `14_gef_readout_frequency_optimization`, `15_iq_blobs_gef`, and the two-qubit CZ leakage nodes) derives the e↔f drive frequency directly as `intermediate_frequency − anharmonicity`, so `anharmonicity` is the one quantity this whole EF chain actually depends on being accurate.

## Troubleshooting

1. **Outcome reported "failed" despite a clean-looking peak in the raw trace** → the actual failure gate is the saturation-amplitude check, not a direct visibility check (see Outputs). A very narrow fitted `width` inflates the `target_peak_width`-implied `saturation_amplitude` past `instrument_limits.max_wf_amplitude`. Either accept that the requested `target_peak_width` is unrealistically small for this peak, or ignore the outcome label and trust the visible peak (this node doesn't act on `saturation_amp`/`x180_amp` anyway — see the Parameters callout).
2. **Logged "Qubit frequency" or the plot's "RF frequency" axis look offset from where you'd expect $f_{12}$ to be** → this is expected, not a bug: those values are computed by code shared with `03a` that doesn't know this node shifted its drive by `−anharmonicity`. They're off from the true e↔f frequency by exactly the (pre-fit) anharmonicity. Only trust `relative_freq` / the resulting `qubit.anharmonicity` update.
3. **Peak found, but the resulting `anharmonicity` update looks physically implausible** (e.g. wildly different from the device's design value, or with the wrong sign of correction) → suspect image-sideband or LO-leakage excitation rather than a genuine e↔f resonance, exactly as the node's own docstring warns. Re-run `01a_mixer_calibration` and repeat.
4. **Line looks weak/absent but not obviously mis-centered** → check that `x180` is actually populating $|e\rangle$ well (re-run `04b_power_rabi` if it hasn't been calibrated, or its calibration has drifted) before assuming the EF drive itself is the problem — with little population in $|e\rangle$, there's nothing for the EF drive to act on.
5. **One qubit's fit degrades only under `multiplexed=True`** → as with `03a`/`03b`, concurrent multiplexed drive/readout can perturb the effective qubit response. Re-run with `multiplexed=False` to isolate whether this is a real crosstalk effect.

## Parameter Tuning Heuristics

1. **No peak visible anywhere in the swept span** → the seed `qubit.anharmonicity` is likely off by more than `frequency_span_in_mhz / 2` from the true value, so the sweep (centered on `RF_frequency − anharmonicity`) misses the real line. Widen `frequency_span_in_mhz` first; if that doesn't help, sanity-check the seed anharmonicity against the device's design value.
2. **Peak is broad, split, or the fitted anharmonicity drifts run-to-run** → likely power broadening or an AC-Stark-like shift from too strong a drive; unlike `03b`, this node inherits `03a`'s default `operation_amplitude_factor = 1.0` rather than a reduced value, so it's worth deliberately lowering it here (à la `03b`'s 0.1 default) if the line doesn't resolve cleanly **[AE1987]**.
3. **Line detected, but a second, spurious feature appears roughly one anharmonicity-width away in the same scan** → if `operation_len_in_ns` was shortened to speed things up, its spectral width may now be wide enough to off-resonantly re-drive the g↔e transition it's supposed to leave alone. Lengthen the pulse (or reduce `operation_amplitude_factor`) to keep the drive's bandwidth well inside the anharmonicity **[GRTW2021]**.

## Next Steps

`13_power_rabi_ef` — needs `qubit.anharmonicity` from this node to place the `EF_x180` pulse's drive frequency correctly.

## References

**[Koc+2007]** J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Charge-insensitive qubit design derived from the Cooper pair box," *Phys. Rev. A*, vol. 76, no. 4, p. 042319, 2007.

**[BGGW2021]** A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[AE1987]** L. Allen and J. H. Eberly, *Optical Resonance and Two-Level Atoms*. New York: Dover Publications, 1987.
