# Readout Power Optimization

[`08b_readout_power_optimization.py`](../../../../../calibrations/1Q_calibrations/08b_readout_power_optimization.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Sweeps the readout pulse amplitude, comparing $|g\rangle$/$|e\rangle$ single-shot response at each power, to find the amplitude that maximizes assignment fidelity — while explicitly gating out amplitudes where the single-shot statistics start breaking down.

## Purpose

Raising the readout drive amplitude increases the average photon number $\bar{n}$ in the resonator, which — up to a point — linearly improves the coherent separation between the $|g\rangle$- and $|e\rangle$-conditioned IQ responses, and hence the achievable single-shot SNR. But this is not a free knob to crank: **[GRTW2021]** names this the "**T1 versus $\bar n$ problem**": *"in practice, a powerful [readout] drive triggers a significant level of qubit relaxation... its exact physical mechanism is still an ongoing topic of research."* Past some device-specific power, the readout tone itself starts inducing extra qubit relaxation or state transitions **during the measurement**, and there is no simple closed-form threshold for where this onset occurs on a given device — it must be found empirically, and it can *look like* rising SNR right up until it doesn't. This node's job is exactly that empirical search, and it treats it as more than a single-number optimization: alongside `meas_fidelity` (how well a two-cluster classifier assigns shots), it separately tracks a "non-outlier" fraction — the share of shots that are *not* poorly explained by a simple two-Gaussian model — and only considers amplitudes where that fraction clears `outliers_threshold` before picking the fidelity-maximizing one. A qubit that starts leaking states mid-measurement produces exactly this signature: shots that don't sit cleanly in either Gaussian blob. This mirrors the broader point in **[Gam+2007]** that achievable readout fidelity trades off against measurement duration and relaxation rate — cranking power to compensate for a fixed, non-negotiable measurement time is not the same thing as actually improving the readout.

## Mechanism

For each amplitude prefactor `a`, linearly spaced between `start_amp` and `end_amp` over `num_amps` points, and for each of `num_shots` repetitions per point:

1. Reset the qubit, measure the resonator with the hardcoded `"readout"` operation using `amplitude_scale=a` (a QUA runtime scale factor applied on top of whatever `operations["readout"].amplitude` is already configured) to record $I_g, Q_g$.
2. Reset again, play `x180`, `align()`, then measure again with the same `amplitude_scale=a` to record $I_e, Q_e$.

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/readout_power_optimization/analysis.py`):

1. Convert I/Q to volts; tag each point with an absolute `readout_amplitude = amp_prefactor * operations["readout"].amplitude` (using whatever baseline amplitude is currently configured — the sweep is always relative to that, not absolute). Re-arrange $I_g/I_e$ and $Q_g/Q_e$ into a single `I`/`Q` pair indexed by a `state` dimension (0 = g, 1 = e). (There is a re-entrancy guard: if the raw `Ig`/`Qg`/`Ie`/`Qe` columns are already gone, `process_raw_dataset` returns the dataset unchanged — relevant when re-analyzing via `load_data_id`.)
2. **Per amplitude**, fit a 2-component Gaussian Mixture Model (spherical covariance, initialized from the actual per-state sample means/variance) jointly on the pooled $g$+$e$ shots at that amplitude. From it, compute:
   - `meas_fidelity`: average of (fraction of $g$-prepared shots the GMM assigns to cluster 0) and (fraction of $e$-prepared shots assigned to cluster 1) — the GMM analog of the confusion-matrix diagonal.
   - `outliers`: the fraction of shots whose GMM log-likelihood is **not** far below the maximum (`log-likelihood > log(0.01) + max_ll`) — i.e., despite the name, this is an **inlier** fraction (the plotting code's own legend calls it "non-outliers"). Low values mean many shots don't fit the clean two-Gaussian picture — exactly the signature of mid-measurement state transitions.
3. `valid_amps`: the subset of swept amplitudes where this inlier fraction is $\geq$ `outliers_threshold` (default 0.98). Amplitudes are **excluded from consideration entirely** if too many of their shots look statistically anomalous — this is the concrete mechanism tying the "T1 vs $\bar n$" concern directly into the fit.
4. `optimal_amp`: among only the surviving `valid_amps`, the one with the highest `meas_fidelity`.
5. At that one winning amplitude, the node re-runs `07_iq_blobs`'s own fit (`fit_iq_blobs`, imported directly from `calibration_utils.iq_blobs.analysis`) on just the shots taken at that amplitude — i.e., **this node performs essentially the same per-shot characterization as `07_iq_blobs`, but restricted to the amplitude it just selected**, to get `iw_angle`/`ge_threshold`/`rus_threshold`/`confusion_matrix` at that specific operating point. The final `success` flag comes from that inner IQ-blobs fit's own NaN check (see `07_iq_blobs`'s Outputs), not from the amplitude-selection step itself.

**Flagged from source:**

- **`plot_raw` is a dead parameter.** It's declared in `NodeSpecificParameters` (`calibration_utils/readout_power_optimization/parameters.py`) but is never referenced anywhere else in the node file, the analysis module, or the plotting module. Setting it has no effect.
- **No graceful handling of an empty `valid_amps`.** If `outliers_threshold` is strict enough (or the qubit's transitions bad enough) that *no* swept amplitude clears it for a given qubit, the subsequent `argmax` over an empty selection is not explicitly guarded in the source — expect an error or an undefined result rather than a clean `"failed"` outcome for that qubit.
- **The sweep is always relative to whatever baseline amplitude is currently configured** (`amp_prefactor * operations["readout"].amplitude`), not an absolute voltage sweep — if that baseline is already far from sane (e.g., left over from a bad earlier run), the entire swept range shifts with it.
- **`end_amp=1.99` sits right at the QUA `amplitude_scale` hardware ceiling** (roughly $[-2, 2)$); raising `end_amp` toward or past 2.0 risks a compile-time/runtime error rather than a clean bounds check in this node's own parameter validation.
- The readout operation is hardcoded to `"readout"` in `create_qua_program`, same as `08a_readout_frequency_optimization` — there is no `operation` parameter here (contrast `07_iq_blobs`).

## Prerequisites

- Readout parameters calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- Qubit `x180` pulse calibrated (`03a_qubit_spectroscopy`, `04b_power_rabi`).
- Per the bring-up graph topology: this is the **first** of the readout-optimization trio to run — it runs immediately after `04b_power_rabi` and before `08a_readout_frequency_optimization`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 2000 | – | Shots per amplitude point, split evenly between $g$/$e$ preparation. | More shots give a cleaner GMM fit (`meas_fidelity`, `outliers`) at each amplitude; linear cost in run time, multiplied by `num_amps`. |
| `start_amp` | `float` | 0.5 | – (amplitude prefactor) | Lower end of the swept `amplitude_scale`, relative to the currently configured `"readout"` amplitude. | Sets the low-power end of the search; too high a `start_amp` may skip over a legitimately lower optimal amplitude. |
| `end_amp` | `float` | 1.99 | – (amplitude prefactor) | Upper end of the sweep. | Near the QUA hardware ceiling for `amplitude_scale` (see note above) — raising it further risks a runtime/compile error rather than probing higher power safely. |
| `num_amps` | `int` | 10 | – | Number of amplitude points across `[start_amp, end_amp]`. | More points resolve the fidelity-vs-power curve (and the outlier onset) more finely, at proportional cost in run time. |
| `outliers_threshold` | `float` | 0.98 | – (fraction, 0–1) | Minimum required "non-outlier" (GMM inlier) fraction for an amplitude to be considered a candidate at all. | **Directly implements the T1-vs-$\bar n$ safeguard**: amplitudes where too many shots look like mid-measurement transitions are excluded from the fidelity-maximization step entirely, before "highest fidelity" is even evaluated among them. Too strict a value can empty out `valid_amps` for a genuinely noisy qubit (see Troubleshooting #2 and Parameter Tuning Heuristics #1). |
| `plot_raw` | `bool` | `False` | – | Documented as "plot raw data." | **No-op** — not referenced anywhere in the node, analysis, or plotting source. |

## Outputs

**Measured:** `I`/`Q` (volts, indexed by `state` $\in\{0,1\}$ and `amp_prefactor`), `readout_amplitude` (V, absolute).

| Per-amplitude fit quantity | Unit | Description |
|---|---|---|
| `meas_fidelity` | – (0–1) | GMM two-cluster assignment accuracy at that amplitude. |
| `outliers` | – (0–1) | GMM inlier fraction at that amplitude — despite the name, *higher is better/cleaner* (see Mechanism). |

| Final fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `optimal_amplitude` | V | ✅ | Absolute readout amplitude at the fidelity-maximizing point among `valid_amps`. |
| `iw_angle` | rad | ✅ (as a decrement) | From the inner `07_iq_blobs` fit at `optimal_amplitude` — see that node's Mechanism for the sign convention. |
| `ge_threshold` | V | ✅ (unit-converted) | From the inner fit at `optimal_amplitude`. |
| `rus_threshold` | V | ✅ (unit-converted) | From the inner fit at `optimal_amplitude`. |
| `readout_fidelity` | % | – | From the inner fit; logged, not written directly (the confusion matrix is). |
| `confusion_matrix` | – (2×2 list) | ✅ | From the inner fit at `optimal_amplitude`. |

**Success criterion:** inherited from the inner `07_iq_blobs`-style fit at the chosen `optimal_amplitude` — none of `iw_angle`/`ge_threshold`/`rus_threshold`/`readout_fidelity` are NaN. This is a **second-stage** check: an amplitude must first survive the `outliers_threshold` gate and win the `meas_fidelity` comparison before this NaN check even runs on it.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.resonator.operations["readout"].integration_weights_angle` | current angle $-$ fitted `iw_angle` | **decrement** | outcome successful |
| `qubit.resonator.operations["readout"].threshold` | `ge_threshold * operation.length / 2**12` | replace | outcome successful |
| `qubit.resonator.operations["readout"].rus_exit_threshold` | `rus_threshold * operation.length / 2**12` | replace | outcome successful |
| `qubit.resonator.operations["readout"].amplitude` | fitted `optimal_amplitude` | replace | outcome successful |
| `qubit.resonator.confusion_matrix` | fitted `confusion_matrix` | replace | outcome successful |

Unlike `07_iq_blobs`, this node always targets the `"readout"` operation specifically (it's hardcoded, not parameterized), so the confusion-matrix write here is **unconditional** on outcome — there's no `operation == "readout"` gate to worry about.

## Troubleshooting

1. **[T1-vs-$\bar n$ — read this one first] `optimal_amplitude` keeps landing at or very near `end_amp`, and `meas_fidelity` is still rising at that edge** → it's tempting to just push `end_amp` higher to chase more SNR. Before doing that, check the `outliers` (inlier-fraction) curve in `figures["amplitude"]` alongside the fidelity curve: per **[GRTW2021]**'s T1-vs-$\bar n$ problem, raising readout power eventually triggers measurement-induced relaxation with no simple closed-form threshold for where it kicks in on this device. If `outliers` is already trending down as amplitude rises within your current range, that decline *is* the early warning sign — don't just extend the range and re-run; treat a declining inlier fraction as the real ceiling, not the fidelity curve alone.
2. **`valid_amps` comes back empty — the node errors, or produces a NaN/undefined result instead of a clean `"failed"` outcome for a qubit** → `outliers_threshold` (default 0.98) is excluding every swept amplitude for that qubit, meaning even the *lowest* tested amplitude already shows a below-threshold inlier fraction — this isn't a graceful failure mode in the current source.
3. **High `meas_fidelity` at the chosen amplitude, but the inner IQ-blobs fit's `readout_fidelity` (or the confusion matrix) looks worse than expected** → the two numbers come from different classifiers on the same data — a 2D Gaussian-mixture assignment (used to pick the amplitude) versus a 1D-threshold classifier (used for the final state update). A GMM can tolerate elongated/rotated blobs that a single-axis threshold resolves less cleanly. Cross-check visually with `figures["iq_blobs"]`/`figures["confusion_matrix"]`, not just the logged `meas_fidelity` number.
4. **`07_iq_blobs`, run immediately after this node in the bring-up graph, reports different `threshold`/`iw_angle` values than this node just wrote** → expected, not a conflict. `07_iq_blobs` doesn't touch `operations["readout"].amplitude`; it simply recomputes `iw_angle`/`threshold`/`rus_exit_threshold` fresh at whatever amplitude this node left configured. Small run-to-run differences reflect ordinary shot noise, not disagreement between the two nodes.
5. **Set `plot_raw=True` expecting additional diagnostics, but nothing changes** → this parameter is a no-op in the current source (see Mechanism/Parameters) — it isn't read anywhere after being declared. Inspect `figures["amplitude"]`, `figures["iq_blobs"]`, and `figures["confusion_matrix"]` directly instead.
6. **Result is inconsistent between `multiplexed=True` and `multiplexed=False` runs for the same qubit** → concurrent readout activity on other qubits' resonators during multiplexed execution can crosstalk into this qubit's response at high power specifically (readout crosstalk tends to scale with drive amplitude). Re-run the qubit alone to confirm; if the inconsistency disappears, treat it as a multiplexed-readout-power crosstalk issue, not a bug in the fit.
7. **A previously-good qubit suddenly needs a much lower `optimal_amplitude` than before, with `outliers` degrading at powers that used to be fine** → this can indicate a genuine change in the qubit's relaxation behavior at readout power (device aging, TLS environment shift, etc.), consistent with the still-not-fully-understood physical origin of the T1-vs-$\bar n$ effect noted in **[GRTW2021]**. Treat a shifted onset power as a real device-characterization data point worth tracking over time, not just noise to re-fit away.
8. **`optimal_amplitude` is capped by the T1-vs-$\bar n$ effect well before you'd like, and no `outliers_threshold`/range tweak helps** → this node only optimizes within the conventional low-power dispersive-readout regime; it has no path to the fundamentally different high-power "bright-state"/Jaynes–Cummings readout regime **[Reed2013]** describes, which sidesteps the $T_1$-vs-power tradeoff entirely by driving the cavity into a qubit-state-dependent nonlinear "bright" state rather than relying on a small dispersive shift — at the cost of no longer being QND, so it's only appropriate as a final measurement, not mid-circuit. If you are hard-capped in this regime and can't get acceptable fidelity, that's a signal this readout scheme itself (not this node's parameters) has reached its ceiling. Separately: if your fidelity plateaus around 60–70% well before any T1-vs-$\bar n$ effect is visible in the `outliers` curve, **[Reed2013]** notes this is frequently amplifier-chain-noise-limited rather than a property of the qubit/readout point at all — check whether a quantum-limited (parametric) amplifier is in the chain before concluding the dispersive readout itself is fundamentally limited here.

## Parameter Tuning Heuristics

1. **`valid_amps` comes back empty for a qubit** → lower `outliers_threshold` modestly and/or lower `start_amp` before assuming something else is broken.
2. **Fidelity curve is still monotonically rising (or falling) across the entire `[start_amp, end_amp]` span, no interior peak** → the true optimum likely lies outside the swept range. Widen `start_amp`/`end_amp`/`num_amps` rather than accepting an edge value as final — same edge-of-span caution as other sweep-based nodes in this library.

## Next Steps

In both bring-up graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`), this node runs right after `04b_power_rabi` and feeds directly into `08a_readout_frequency_optimization`, which in turn feeds `07_iq_blobs`. All three exist to jointly tune readout (amplitude, then frequency, then per-shot threshold/rotation) before anything downstream relies on `use_state_discrimination=True`.

## References

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[Gam+2007]** J. Gambetta, W. A. Braff, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, "Protocols for optimal readout of qubits using a continuous quantum nondemolition measurement," *Phys. Rev. A*, vol. 76, p. 012325, 2007.

**[Reed2013]** M. D. Reed, *Entanglement and Quantum Error Correction with Superconducting Qubits*, Ph.D. dissertation, Yale University, New Haven, CT, 2013.
