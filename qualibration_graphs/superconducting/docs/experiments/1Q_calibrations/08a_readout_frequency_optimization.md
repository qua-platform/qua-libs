# Readout Frequency Optimization

[`08a_readout_frequency_optimization.py`](../../../../../calibrations/1Q_calibrations/08a_readout_frequency_optimization.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Sweeps the readout tone's frequency around its currently configured point, comparing the resonator's $|g\rangle$ and $|e\rangle$ response at each point, to locate the frequency that best resolves the two qubit states and to estimate the dispersive shift $\chi$.

## Purpose

In the dispersive regime, the resonator's dressed frequency shifts by $\pm\chi$ depending on whether the qubit is in $|g\rangle$ or $|e\rangle$ **[BGGW2021]**, so the two states produce two distinguishable resonance lineshapes, split by $2\chi$, each with linewidth set by the resonator's total decay rate $\kappa$. How well a fixed-frequency readout tone can tell the two states apart is not simply "the bigger $\chi$ the better": **[GRTW2021]** frames the practically optimal regime as the point where "$\chi = \kappa$" — the dispersive shift and the resonator linewidth should be *comparable*, since a resonator that's too broad ($\kappa \gg \chi$) blurs the two lineshapes together, while a resonator that's too narrow ($\kappa \ll \chi$) means only a narrow slice of drive frequencies actually distinguishes the states well, and typically implies a resonator that empties too slowly between shots.

This node doesn't change $\chi$ or $\kappa$ themselves — those are set by the resonator design and coupling. What it does is find, for the device *as built*, the specific readout frequency at which the $|g\rangle$- and $|e\rangle$-conditioned resonator responses differ most in the IQ plane — i.e., where the state-dependent lineshapes are best resolved given whatever $\chi/\kappa$ ratio the hardware provides — and it separately reads off $\chi$ itself from the frequency separation between the two response minima, so that ratio can be checked against the $\chi \approx \kappa$ heuristic. As with any dispersive-readout tuning, this frequency choice interacts with readout duration and amplitude rather than being independently maximizable: too short a measurement doesn't accumulate enough SNR regardless of frequency, and per **[Gam+2007]**, achievable readout fidelity given a fixed relaxation time trades off directly against how fast the measurement completes — frequency, amplitude (`08b_readout_power_optimization`), and duration all have to be reasoned about jointly.

## Mechanism

For each (drive-frequency detuning) point in the 1D sweep, centered on the qubit's currently configured readout frequency:

1. `update_frequency` on the resonator element in real time (no re-compilation per point) to the swept intermediate frequency.
2. Reset the qubit (`reset_type`), measure the resonator with the hardcoded `"readout"` operation (not the common `operation` parameter — there is none for this node) to record $I_g, Q_g$.
3. Reset again, play `x180` on `qubit.xy`, `align()`, then measure again to record $I_e, Q_e$.
4. Average over `num_shots` repetitions per detuning point (QUA `stream_processing().average()`), not per-shot statistics — this node characterizes the *mean* response at each frequency, unlike `07_iq_blobs`'s per-shot histograms.

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/readout_frequency_optimization/analysis.py`):

1. Convert I/Q to volts; compute $D = \sqrt{(I_g-I_e)^2+(Q_g-Q_e)^2}$ (the distance between the averaged $|g\rangle$- and $|e\rangle$-response points in the IQ plane at each detuning) and $|IQ|_{g}$, $|IQ|_{e}$ (response magnitude in each state).
2. Smooth $D$ with a 5-point rolling mean and take the `argmax` over detuning to get `optimal_detuning`/`optimal_frequency` — the frequency point where the two states' responses are most separated.
3. Estimate `chi` as half the frequency separation between the two states' response minima: $\chi = \big(\underset{\text{detuning}}{\mathrm{argmin}}\,|IQ|_e - \underset{\text{detuning}}{\mathrm{argmin}}\,|IQ|_g\big)/2$, matching the convention that the resonator dip/peak shifts by $\pm\chi$ around the bare resonator frequency depending on qubit state.

**Flagged from source:**

- **The frequency-sanity check in the success criterion is a no-op.** `_extract_relevant_fit_parameters` computes `freq_success = np.abs(np.isnan(ds_fit["optimal_detuning"])) < 400e6`. `np.isnan(...)` returns a boolean (0 or 1); `np.abs` of that is still 0 or 1, which is *always* `< 400e6`. This line was almost certainly intended to check `np.abs(ds_fit["optimal_detuning"]) < 400e6` (i.e., reject an optimum found implausibly far from the current frequency), but as written it always evaluates `True` regardless of the actual detuning found. In practice the success criterion reduces entirely to "are `chi`/`optimal_detuning`/`optimal_frequency` non-NaN" — see Troubleshooting #1.
- **The 5-point rolling-mean smoothing window is a fixed point-count, not a fixed frequency width.** At the default `frequency_step_in_mhz=0.1`, that's a 0.5 MHz smoothing window; widening `frequency_step_in_mhz` silently widens that window proportionally in frequency, which can smear a sharp peak at coarse steps.
- The readout operation is hardcoded to `"readout"` in `create_qua_program` — there is no equivalent of `07_iq_blobs`'s `operation` parameter here.

## Prerequisites

- Readout parameters calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- Qubit `x180` pulse calibrated (`03a_qubit_spectroscopy`, `04b_power_rabi`).
- Per the bring-up graph topology: run after `08b_readout_power_optimization` — the readout amplitude is optimized first, then this node's frequency sweep is done at that amplitude.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per detuning point. | Smooths the mean $I,Q$ per point; note this is much lower than `07_iq_blobs`'s default of 2000, since this node needs a clean average per frequency, not per-shot histogram statistics. |
| `frequency_span_in_mhz` | `float` | 10.0 | MHz | Full width of the frequency sweep, centered on the qubit's currently configured readout frequency. | Must be wide enough to contain the actual $D$-maximizing frequency; too narrow forces the fit toward the span edge (see Parameter Tuning Heuristics #4). Directly bounds how far `optimal_frequency` can land from the starting point. |
| `frequency_step_in_mhz` | `float` | 0.1 | MHz | Step size of the frequency sweep. | Finer steps resolve the $D(\text{detuning})$ peak more precisely, but also shrink the *effective* smoothing window relative to the fixed 5-point rolling mean (see Mechanism) — very coarse steps can make that window too wide relative to real spectral features. |

## Outputs

**Measured:** `I_g`/`Q_g`/`I_e`/`Q_e` (volts, averaged over `num_shots`), `D`, `IQ_abs_g`, `IQ_abs_e` at every detuning point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `optimal_frequency` | Hz | ✅ | Absolute readout frequency at the detuning maximizing smoothed $D$. |
| `optimal_detuning` | Hz | – | `optimal_frequency` minus the qubit's `RF_frequency` at the time of the run; the quantity nominally (but not actually, see Mechanism) checked against 400 MHz. |
| `chi` | Hz | ✅ | Estimated dispersive shift, from half the $|g\rangle$/$|e\rangle$ response-minimum separation. |

**Success criterion:** none of `chi`, `optimal_detuning`, `optimal_frequency` are NaN. The additional "is `optimal_detuning` within 400 MHz" check in the source code is dead code (see Mechanism) and imposes no actual constraint — inspect the logged `optimal_detuning`/`chi` values yourself rather than trusting the `"successful"` outcome to catch an implausible result.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.resonator.f_01` | fitted `optimal_frequency` | replace | outcome successful |
| `qubit.resonator.RF_frequency` | fitted `optimal_frequency` | replace | outcome successful |
| `qubit.chi` | fitted `chi` | replace | outcome successful |

Setting `qubit.resonator.RF_frequency` also updates the resonator's `intermediate_frequency`, which is derived from `RF_frequency - LO_frequency` via a QUAM reference by default.

## Troubleshooting

1. **Outcome is `"successful"` but the applied `optimal_frequency`/`optimal_detuning` looks implausible (e.g. detuned by more than a resonator linewidth from the previous `02a`/`02c` frequency)** → do not rely on the `"successful"` flag to catch this: the source's 400 MHz sanity check is a no-op due to a `np.abs(np.isnan(...))` bug (see Mechanism), so an aberrant `optimal_detuning` is still marked successful and written to `qubit.resonator.RF_frequency`. Always sanity-check the logged detuning/chi by eye, especially after widening `frequency_span_in_mhz`.
2. **$D$ barely rises anywhere across the whole sweep — no real separation achievable at any frequency in range** → check that the `x180` pulse is actually driving full population transfer (independently verify via `04b_power_rabi`) before assuming the frequency sweep itself is at fault; also consider that with an unfavorable device $\chi/\kappa$ ratio, no single frequency choice compensates for $\kappa$ that's much larger than $\chi$ **[GRTW2021]** — that requires resonator/coupling redesign, not a wider sweep.
3. **Node run right after other flux/power drift, and the sweep is centered somewhere clearly off from the resonator's true response** → the sweep is centered on the qubit's *currently configured* `RF_frequency`; if `02a`/`02c` resonator spectroscopy is stale, the whole span is off-target. Re-run resonator spectroscopy first.
4. **`13_power_rabi_ef` or the GEF-readout nodes (`14_gef_readout_frequency_optimization`, `15_iq_blobs_gef`) behave oddly downstream** → `13_power_rabi_ef`'s own docstring explicitly lists "having calibrated the readout resonator dispersive shift (chi)" via this node as a prerequisite; confirm `qubit.chi` reflects a genuine, recent, successful run of this node for the qubits in question, not a stale or default value.

## Parameter Tuning Heuristics

1. **$D(\text{detuning})$ is flat, with no clear peak anywhere in the swept span** → `frequency_span_in_mhz` (default 10 MHz) is likely too narrow to contain the full separation between the state-dependent lineshapes. Per the $\chi=\kappa$ framing, if $2\chi$ is comparable to or larger than the span, you're only sampling part of one state's resonance. Widen the span and re-run.
2. **`chi` comes out anomalously small (or noisy run-to-run) despite a visibly clear splitting in the `figures["iq_abs"]` plot** → the `idxmin`-based estimate is sensitive to noise in `IQ_abs_g`/`IQ_abs_e` at low `num_shots`; a single noisy sample near the true minimum can shift the detected minimum's position. Increase `num_shots` for cleaner traces before trusting a surprising `chi` value.
3. **`optimal_frequency` visibly doesn't sit at the peak of `D` in `figures["distances"]`, especially at coarse `frequency_step_in_mhz`** → the fixed 5-point rolling-mean window (see Mechanism) becomes too wide relative to the true peak width when `frequency_step_in_mhz` is large. Reduce the step size rather than trying to compensate with `frequency_span_in_mhz`.
4. **`optimal_frequency` lands at or very near the edge of the swept span** → the true $D$-maximizing frequency may lie outside the window. Re-run with a wider `frequency_span_in_mhz` rather than accepting an edge value — an edge result should be treated as "inconclusive," not "optimal."

## Next Steps

In both bring-up graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`), this node runs after `08b_readout_power_optimization` and immediately before `07_iq_blobs`, which performs the per-shot discrimination-threshold characterization at the frequency (and amplitude) this node and `08b` establish. The dispersive shift `qubit.chi` this node writes also feeds the extended e-f-level pipeline (`13_power_rabi_ef`, and onward to `14_gef_readout_frequency_optimization`/`15_iq_blobs_gef`), which is not wired into the default bring-up graphs but consumes `chi` as a starting point when used.

## References

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[BGGW2021]** A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.

**[Gam+2007]** J. Gambetta, W. A. Braff, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, "Protocols for optimal readout of qubits using a continuous quantum nondemolition measurement," *Phys. Rev. A*, vol. 76, p. 012325, 2007.
