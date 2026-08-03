# Ramsey (Virtual-Z)

[`06a_ramsey.py`](../../../../../calibrations/1Q_calibrations/06a_ramsey.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Plays an $x90$ – idle – $x90$ Ramsey sequence with an artificial detuning encoded as a virtual-$Z$ phase, to precisely fix the qubit's $0\to1$ frequency and extract $T_2^*$.

## Purpose

A Ramsey experiment interferes the qubit's free-evolution phase against the drive's phase reference: two $\pi/2$ pulses separated by an idle time $t$ map any detuning $\delta$ between the qubit's true transition frequency and the drive frequency onto an oscillation of the final population at frequency $\delta$, decaying at the qubit's *inhomogeneous* dephasing rate $1/T_2^*$ **[Ramsey1950]**, **[Kra+2019]**. Fitted against the decaying-oscillation form **[GRTW2021]** (their Eq. 54):

$$S(t) = A\, e^{-(t/T_2^*)^n}\left[\cos(2\pi f t + \phi) + C\right] + B$$

this node reports both $\delta$ (used to correct `f_01`) and $T_2^*$ in one measurement.

This implementation never actually detunes the local oscillator. Instead, right after the first $x90$ pulse it applies a virtual-$Z$ rotation (`frame_rotation_2pi`) to the drive frame equal to the phase a real detuning of `frequency_detuning_in_mhz` would have accumulated over the current idle time, before playing the second $x90$ on-resonance. This is the "artificial detuning" trick described in **[GRTW2021]**: *"pulses are often slightly detuned intentionally"* — either physically or, as here, by phase-shifting the second pulse — *"because it is difficult to distinguish an oscillation due to a low frequency (small detuning) from an exponential decay."* Without it, a well-calibrated qubit (small true detuning) would produce an almost-flat, slowly-varying trace that is nearly indistinguishable from pure exponential decay, making $T_2^*$ unreliable to extract and the sign of any residual detuning invisible. The virtual-$Z$ approach gets the same disambiguating benefit while keeping every physical pulse on resonance **[GRTW2021]**.

## Mechanism

The node sweeps idle time **and** the sign of the artificial detuning together, interleaving both signs at each idle time (rather than sweeping them as two separate passes) so that slow common-mode drift between the two traces partially cancels. For each `(idle_time, detuning_sign)` point, repeated `num_shots` times:

1. `reset_frame(qubit.xy.name)`, then reset the qubit (`qubit.reset(reset_type, ...)`). The frame reset is required because `frame_rotation_2pi` (step 3) permanently offsets the element's phase reference — without resetting it every shot, the artificial-detuning phase would accumulate across shots instead of being recomputed fresh each time.
2. Play `x90` on `qubit.xy`.
3. Compute the virtual-detuning phase for the *current* idle time, $\phi = \pm({\tt frequency\_detuning\_in\_mhz}) \times t_{\rm idle}$ (sign selected by `detuning_sign \in \{-1, +1\}$), and apply it as an instantaneous frame rotation (`frame_rotation_2pi`) — i.e. the phase a real detuning would have accumulated *continuously* over the wait is instead injected as a single virtual-$Z$ gate immediately after the first pulse.
4. `qubit.xy.wait(idle_time)`, then play the second `x90`.
5. Measure the resonator (or discriminated `state`).

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/ramsey/analysis.py`):

1. Convert I/Q to volts (skipped for discriminated `state`).
2. Independently fit each detuning-sign trace along `idle_time` to a decaying cosine, `a * exp(-t*decay) * cos(2*pi*f*t + phi) + offset` (`fit_oscillation_decay_exp`). **Note:** this fit uses a plain single exponential envelope ($n=1$ in the Eq. 54 form above) — the repository does not fit the stretching exponent $n$ automatically. See Troubleshooting #1 for what that means in practice.
3. Disambiguate the true residual detuning's sign and magnitude by comparing the two fitted oscillation frequencies (`calculate_fit_results`): if both extracted frequencies stay below $2\times$ the artificial detuning (`within_detuning`), the true detuning is small compared to the artificial one and a simple sign-weighted average (`frequency * detuning_signs`, averaged over signs) is used directly. Otherwise — the real detuning is comparable to or larger than the artificial one — it falls back to comparing which sign's trace oscillates faster (`positive_shift`) to infer the sign, and uses the unsigned mean frequency magnitude.
4. $T_2^* = 1/{\tt decay}$ per sign, averaged across the two signs into the reported `decay` (seconds) and `decay_error`.
5. **Success only checks that `freq_offset` and `decay` are non-NaN** — unlike `05_T1`, there is no magnitude/plausibility bound (e.g. nothing rejects a fitted $T_2^*$ far larger than `max_wait_time_in_ns`). See Troubleshooting #2.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration` or `01b`).
- Readout calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- Qubit $\pi$/$\pi$/2 pulses calibrated (`03a_qubit_spectroscopy`, `04b_power_rabi`) — the sequence plays `x90` twice per shot.
- A reasonably accurate `qubit.f_01`/`qubit.xy.RF_frequency` already set — this node corrects a *residual* detuning, it doesn't search blindly (see Parameter Tuning Heuristics #2 for what happens if the real error is too large).
- (Optional) Readout parameters optimized (`08a`, `08b`, `08c`).
- `qubit.z.flux_point` set as desired, if the qubit is flux-tunable.
- `07_iq_blobs` calibrated, only if running with `use_state_discrimination=True`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the `IdleTimeNodeParameters` group (shared with `05_T1`/`06b_echo`) and this node's own parameters.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per `(idle_time, detuning_sign)` point. | Reduces scatter on the oscillation without changing the fitted frequency/decay; linear cost in run time. |
| `frequency_detuning_in_mhz` | `float` | 1.0 | MHz | Magnitude of the artificial (virtual-$Z$) detuning applied via `frame_rotation_2pi`. | Sets both the visible fringe frequency (needed to distinguish oscillation from pure decay, see Purpose) and the window within which the true residual detuning can be unambiguously sign-resolved (`within_detuning` check, Mechanism #3). Too small reproduces the original "flat trace" problem; too large risks aliasing the fringe against the idle-time sampling — see Parameter Tuning Heuristics #1/#2. |
| `min_wait_time_in_ns` | `int` | 16 | ns | Shortest idle time swept. | Anchors the start of the Ramsey trace near $t=0$. |
| `max_wait_time_in_ns` | `int` | 30000 | ns | Longest idle time swept. | Sets the sampling bandwidth/resolution of the frequency fit — see the coarse-then-fine procedure in Parameter Tuning Heuristics #2. |
| `wait_time_num_points` | `int` | 500 | – | Number of idle-time points between the min and max. | More points reduce fit variance on both frequency and decay, at proportional cost in run time. |
| `log_or_linear_sweep` | `Literal["log", "linear"]` | `"log"` | – | Spacing of the idle-time sweep. | Log spacing concentrates points at short $t$, where the decay envelope carries the most information per point — the same reasoning as in `05_T1`. Be aware this also means fringe sampling density (points per oscillation period) is highest early and sparsest near `max_wait_time_in_ns`, which matters when picking `frequency_detuning_in_mhz` (Parameter Tuning Heuristics #1). |

## Outputs

**Measured:** `I`/`Q` (or discriminated `state`), at every `(idle_time, detuning_sign)` point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `freq_offset` | Hz | ✅ (as a correction to `qubit.f_01`/`qubit.xy.RF_frequency`) | Disambiguated residual detuning between the qubit's true $f_{01}$ and its currently configured value. |
| `decay` | s | ✅ (as `qubit.T2ramsey`) | Fitted $T_2^*$, averaged over the two detuning signs. Despite the field being named `decay`, it is a **time**, not a rate — it's already the reciprocal of the fitted decay rate. |
| `decay_error` | s | – | Propagated fit uncertainty on `decay`. |

**Success criterion:** `freq_offset` and `decay` are both non-NaN. Checked per-qubit in `_extract_relevant_fit_parameters` — see Mechanism #5 for the caveat that this is not a plausibility check.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.f_01` | `qubit.f_01 - freq_offset` | increment (`-=`) | outcome successful |
| `qubit.xy.RF_frequency` | `qubit.xy.RF_frequency - freq_offset` | increment (`-=`) | outcome successful |
| `qubit.T2ramsey` | fitted `decay` | replace | outcome successful |

`f_01` and `RF_frequency` are both decremented by the *same* `freq_offset`, keeping them in sync with each other after the correction.

## Troubleshooting

1. **Want to know whether `06b_echo` will actually help before running it** → fit the decay envelope's stretching exponent $n$ by hand (this node's own analysis does not do it automatically, see Mechanism #2): *"If $T_2^*$ is limited by $T_1$ or another source of incoherent noise, $n \approx 1$. If there is a coherent noise process, such as a slow drift in the qubit frequency, $n$ will be larger than 1. In this case, it is possible to recover part of the information by using an echo pulse"* **[GRTW2021]**. In practice, look for systematic curvature in the fit residuals away from a clean single exponential (rapid drop then a long, slowly-decaying tail is a giveaway) — that pattern predicts `06b_echo` will report a $T_2^{\rm echo}$ meaningfully longer than this node's $T_2^*$; a residual that already looks like clean single-exponential decay predicts echo won't buy you much.
2. **`qubit.T2ramsey` gets updated to an obviously unphysical value (much larger than `max_wait_time_in_ns`, or negative) but the node still reports "successful"** → this is a real gap in the current fit's success check, which only rejects NaN results (Mechanism #5), not implausible ones. Always sanity-check the plotted fit against `max_wait_time_in_ns` before trusting `T2ramsey`.
3. **No log line with `T2*`/detuning values appears, even though the node ran and updated state** → unlike `06b_echo` (which fully suppresses its log output), this node's `log_fitted_results` does print a summary — check `node.log` output/console, not the plot, if you don't see it; a missing log here more likely indicates the analysis step didn't run at all (e.g. `simulate=True` skips `analyse_data` entirely).
4. **Result is unstable / different every re-run at nominally the same detuning** → if the fringe amplitude is already small (e.g. because of $T_2^*$-limited visibility at long idle times or imperfect $x90$ calibration), that noise can dominate the fit. Re-verify `04b_power_rabi` if the amplitude itself looks too low even at short idle times.

## Parameter Tuning Heuristics

1. **Oscillation is barely visible / trace looks almost like pure decay** → `frequency_detuning_in_mhz` (default 1.0 MHz) is too small relative to the true residual detuning **or** too small in absolute terms to distinguish an oscillation from a decay at all — this is exactly the failure mode the artificial-detuning trick exists to avoid **[GRTW2021]**. As a concrete target rather than a guess: **[Reed2013]** recommends choosing the detuning so you get **4–5 full fringe oscillations within the first two $1/e$ decay times** of the expected $T_2^*$ — too few oscillations in that window gives an ambiguous fit; choose `frequency_detuning_in_mhz` from a rough prior estimate of $T_2^*$ using this rule rather than the bare 1.0 MHz default when $T_2^*$ is very short or very long. Increase `frequency_detuning_in_mhz` first before suspecting a measurement problem.
2. **Fitted frequency looks stable but is clearly wrong once cross-checked against a wider scan (aliasing)** → per **[GRTW2021]**, *"the first Ramsey experiment should cover a short time span to get a coarse frequency estimate with a large bandwidth, while subsequent iterations can increase the time span to achieve more accurate frequency estimates at a lower bandwidth."* Concretely: run once with a **small** `max_wait_time_in_ns` (and/or a larger `frequency_detuning_in_mhz`) to get a coarse, alias-free estimate of the residual detuning, apply that correction, then re-run with a **larger** `max_wait_time_in_ns` (finer frequency resolution) once you're already close to zero detuning. Don't jump straight to a large `max_wait_time_in_ns` on a qubit with a large unknown frequency error.
3. **The two detuning-sign traces (`$\Delta$ = +` / `$\Delta$ = -` in the plot) have visibly different apparent frequencies or decay rates** → since `within_detuning` requires *both* signs' fitted frequencies to stay below $2\times$ the artificial detuning, a large mismatch between the two usually means the true residual detuning is comparable to or larger than `frequency_detuning_in_mhz` itself, pushing the analysis into its fallback sign-disambiguation branch (Mechanism #3). Increase `frequency_detuning_in_mhz` so both signs land clearly inside the `within_detuning` regime, then re-run.
4. **Fit succeeds but the corrected `f_01` sends the qubit further from resonance, not closer, on the next iteration** → suspect the sign-disambiguation fallback (`positive_shift`) picked the wrong sign. This happens when the true residual detuning is close in magnitude to the artificial one, where fit noise can flip which sign's trace "looks faster." Increase `frequency_detuning_in_mhz` to put clear daylight between the two traces' frequencies, or iterate with a smaller step and re-check convergence rather than trusting a single run blindly.
5. **`qubit.T2ramsey` gets updated to an obviously unphysical value (much larger than `max_wait_time_in_ns`, or negative)** → if the fitted decay time is much longer than the swept window, the decay envelope was never actually resolved and the fit is extrapolating, not measuring — widen `max_wait_time_in_ns` and re-run.
6. **Result is unstable / different every re-run at nominally the same detuning** → low `num_shots` (default 100, lower than `05_T1`'s 1000) leaves more shot noise on each point. Increase `num_shots`.

## Next Steps

`05_T1` — the repository's bring-up calibration graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py` and `90_calibration_graph_bringup_fixed_frequency_transmon.py`) wire this node's output directly into the $T_1$ measurement that follows (`ramsey` → `T1` → `T2echo`). Diagnostically, if the fitted decay shows the coherent-noise signature described in Troubleshooting #1, the natural physics follow-up is `06b_echo` — even though it runs after `05_T1` in the graph, not immediately after this node.

## References

**[Ramsey1950]** N. F. Ramsey, "A molecular beam resonance method with separated oscillating fields," *Phys. Rev.*, vol. 78, no. 6, pp. 695–699, 1950.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.

**[Reed2013]** M. D. Reed, *Entanglement and Quantum Error Correction with Superconducting Qubits*, Ph.D. dissertation, Yale University, New Haven, CT, 2013.
