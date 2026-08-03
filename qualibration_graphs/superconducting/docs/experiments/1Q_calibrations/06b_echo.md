# Hahn Echo ($T_2^{\rm echo}$)

[`06b_echo.py`](../../../../../calibrations/1Q_calibrations/06b_echo.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Plays an $x90$ – idle – $x180$ – idle – $(-x90)$ Hahn-echo sequence for a range of idle times and fits the decay to extract $T_2^{\rm echo}$, the coherence time with slow dephasing noise refocused out.

## Purpose

A plain Ramsey experiment measures $T_2^*$, which is shortened by *any* source of dephasing, including noise that is effectively static on the timescale of a single shot (e.g. quasi-static flux noise, slow frequency drift). The Hahn echo technique **[Hahn1950]**, originally developed for spin systems, cancels exactly this kind of noise: inserting a refocusing $\pi$ pulse ($x180$) halfway through the free-evolution window flips the sign of the phase accumulated so far, so that any dephasing contribution that stayed roughly constant across the two equal-length free-evolution halves accumulates in one half and *un*-accumulates in the other, canceling at the final $\pi/2$ pulse. Fast noise — fluctuating on a timescale comparable to or shorter than the total echo time — does **not** refocus, because the qubit's phase trajectory looks different in the two halves; the echo therefore acts as a filter that removes the low-frequency part of the dephasing noise spectrum while remaining sensitive to its high-frequency part **[Kra+2019]**. The result, $T_2^{\rm echo}$, is bounded below by $T_2^*$ and above by $2T_1$ (the "pure dephasing" component vanishes for noise the echo fully refocuses, leaving only relaxation-limited decay), and the fit follows the same single-exponential-envelope form as $T_1$ **[GRTW2021]** (their Eq. 55):

$$S(t) = A\, e^{-(t/T_2^{\rm echo})^n} + B$$

Comparing $T_2^{\rm echo}$ (this node) against $T_2^*$ (`06a_ramsey`) is itself a diagnostic: a large gap between the two is direct evidence that slow, coherent frequency drift — not $T_1$ — was the dominant limit on `06a_ramsey`'s result (see that node's Troubleshooting #1 for the stretching-exponent argument that predicts this in advance).

## Mechanism

For each idle time $t$ in the sweep, repeated `num_shots` times:

1. `reset_frame(qubit.xy.name)`, then reset the qubit (`qubit.reset(reset_type, ...)`).
2. Play `x90` on `qubit.xy`, wait $t$, play `x180` (the refocusing pulse), wait $t$ again (the second free-evolution interval, equal in length to the first), then play `-x90` — a pulse with the sign of `x90` inverted, registered independently in QUAM rather than implemented as a runtime frame trick — to convert the refocused phase back into a population difference.
3. Measure the resonator (or discriminated `state`).

The two idle waits are each of length $t$ (in clock cycles), but the dataset's `idle_time` axis is registered as `2 * 4 * idle_times` — i.e. it reports the **total** free-evolution time (both halves combined), matching the usual convention for quoting $T_2^{\rm echo}$ against total dark time, not the per-half wait actually issued in QUA.

**Anomalous loop structure for multiplexed runs (flag):** unlike `05_T1` and `06a_ramsey`, which loop over the qubits in a multiplexed batch exactly once per sweep step, `06b_echo.py`'s `create_qua_program` wraps its entire `shot` × `idle_time` × (reset/manipulate/readout-all-qubits) block inside an *additional* outer `for i, qubit in multiplexed_qubits.items():` (source, `create_qua_program`, around lines 92–121) whose loop variable is never actually used — the reset/manipulate/readout steps inside all re-iterate over the full `multiplexed_qubits` dict again regardless. With `multiplexed=False` (the default), each batch contains exactly one qubit, so this outer loop runs once and is harmless. With `multiplexed=True` and more than one qubit in a batch, the entire averaging-and-sweep sequence for *all* qubits in that batch gets compiled and executed once per qubit in the batch — i.e. an $N$-qubit multiplexed batch runs $N\times$ longer than necessary, though the extra repetitions are redundant identical measurements averaged together, so the fitted numbers themselves are not biased by it, only the run time.

Analysis (`process_raw_dataset` / `fit_raw_data` in `calibration_utils/T2echo/analysis.py`):

1. Convert I/Q to volts (skipped for discriminated `state`).
2. Fit `a * exp(t * decay) + offset` along `idle_time` (`fit_decay_exp`) — the same plain single-exponential model used by `05_T1`; there is no stretching-exponent ($n$) fit here either.
3. $T_2^{\rm echo} = -1/{\tt decay}$, with propagated error `T2_echo_error`.
4. **Success only checks that `T2_echo` and `T2_echo_error` are non-NaN** — there is no magnitude/plausibility bound analogous to `05_T1`'s `tau > 16 ns` check.
5. **`log_fitted_results` is a no-op.** The function body in `calibration_utils/T2echo/analysis.py` is entirely commented out (ending in a bare `pass`), so — unlike `05_T1` and `06a_ramsey` — running this node prints **no** textual summary of the fitted $T_2^{\rm echo}$ to the log. The only ways to see the fitted value are the plotted figure (which does annotate $T_2^{\rm echo} \pm$ error) or reading `node.results["fit_results"]`/`ds_fit["T2_echo"]` directly.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration` or `01b`).
- Qubit frequency precisely calibrated (`06a_ramsey`) — per the node's own docstring, this is listed as the direct prerequisite (a poorly centered $f_{01}$ leaves residual free precession that the echo does not remove, biasing the apparent decay).
- (Optional) Readout parameters optimized (`08a`, `08b`, `08c`).
- `qubit.z.flux_point` set as desired, if the qubit is flux-tunable.
- `07_iq_blobs` calibrated, only if running with `use_state_discrimination=True`.

> **Not listed in the node's own docstring, but required by the sequence:** the $\pi/2$ and $\pi$ pulses (`x90`, `x180`, `-x90`) must already be calibrated (`04b_power_rabi`). The docstring's prerequisite list omits this — likely because it's inherited unchanged from `06a_ramsey`'s list minus the readout-calibration lines — but the sequence cannot run meaningfully without it.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the `IdleTimeNodeParameters` group (shared with `05_T1`/`06a_ramsey`) and this node's own parameter.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per idle-time point. | Reduces scatter on the exponential without changing the fitted $T_2^{\rm echo}$; linear cost in run time. |
| `min_wait_time_in_ns` | `int` | 16 | ns | Shortest **per-half** idle time swept (total dark time reported is double this). | Anchors the start of the decay curve near $t=0$. |
| `max_wait_time_in_ns` | `int` | 30000 | ns | Longest **per-half** idle time swept (total dark time reported is double this — up to 60 µs). | Must extend to several multiples of the qubit's expected $T_2^{\rm echo}$ for the decay to be fully resolved — see Parameter Tuning Heuristics #1. |
| `wait_time_num_points` | `int` | 500 | – | Number of idle-time points between the min and max. | More points reduce fit variance at proportional cost in run time (doubled again per multiplexed-batch qubit — see Mechanism). |
| `log_or_linear_sweep` | `Literal["log", "linear"]` | `"log"` | – | Spacing of the idle-time sweep. | Log spacing concentrates points at short $t$, where a single exponential's information content per point is highest — same reasoning as `05_T1`. |

## Outputs

**Measured:** `I`/`Q` (or discriminated `state`), at every idle-time point (total dark time = $2\times$ the per-half swept value).

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `T2_echo` | s | ✅ (as `qubit.T2echo`) | Fitted Hahn-echo coherence time, $= -1/{\tt decay}$. |
| `T2_echo_error` | s | – | Propagated fit uncertainty on `T2_echo`. |

**Success criterion:** `T2_echo` and `T2_echo_error` are both non-NaN. Checked per-qubit in `_extract_relevant_fit_parameters` — see Mechanism #4 above for the caveat that this is not a plausibility check.

## State Updates

**Resolved:** despite the node's own description block reading *"Next steps before going to the next node: Update the qubit T2 echo: `qubit.T2echo`"* — phrasing that reads as a manual to-do — the `update_state` run-action **does** write it automatically. `q.T2echo = node.results["fit_results"][q.name]["T2_echo"]` executes unconditionally for every qubit whose outcome isn't `"failed"`, inside `with node.record_state_updates():`, in the same automatic pattern used by every other node in this family. No manual step is actually required; the docstring wording is simply misleading, inherited phrasing rather than an accurate description of the code.

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.T2echo` | fitted `T2_echo` | replace | outcome not `"failed"` |

## Troubleshooting

See also: [General troubleshooting](_general_troubleshooting.md#general-troubleshooting) for issues common to most nodes in this library. Below are issues specific to this node.

1. **`T2_echo` comes back much longer than `06a_ramsey`'s `T2ramsey`, and the Ramsey fit's residuals showed a slow, non-exponential tail** → this is the expected signature of coherent, low-frequency noise (e.g. slow flux/frequency drift) being refocused by the echo pulse, exactly as predicted in `06a_ramsey`'s Troubleshooting #1 **[GRTW2021]**. Not a bug — it's the whole point of running this node after a Ramsey that shows that pattern.
2. **`T2_echo` comes back roughly equal to `06a_ramsey`'s `T2ramsey`** → suggests the dephasing in the Ramsey measurement was already dominated by incoherent/high-frequency noise (or by $T_1$ itself) rather than slow drift — the echo pulse can't refocus noise that isn't slow compared to the echo time. Check `05_T1`: if $T_2^{\rm echo} \approx 2\,T_1$, relaxation (not dephasing) is the binding constraint, and further gains would need to come from reducing $T_1$-limiting loss, not from more clever dephasing-refocusing sequences.
3. **No log output for the fitted $T_2^{\rm echo}$ value** → expected — `log_fitted_results` for this node is a no-op (Mechanism #5). Read the annotated figure, or `node.results["fit_results"][qubit_name]["T2_echo"]` directly; don't assume the node failed to fit just because nothing was printed.
4. **A multiplexed run with several qubits in one batch takes much longer than the equivalent `05_T1`/`06a_ramsey` run** → this is the loop-structure issue flagged in Mechanism: with `multiplexed=True`, this node's QUA program redundantly repeats its full averaging sequence once per qubit in the batch. The fitted numbers are unaffected, but if run time matters, consider `multiplexed=False` for this specific node, or budget for the $N\times$ slowdown with an $N$-qubit batch.
5. **`T2_echo` drifts from run to run even though `06a_ramsey`'s prerequisite calibration was freshly re-run each time** → if `qubit.f_01` itself is drifting between the Ramsey run and this node's run (e.g. due to flux noise on a tunable transmon away from the sweet spot), residual free precession during the echo's two dark periods won't fully refocus even though the sequence is designed to cancel *static* detuning — re-run `03b_qubit_spectroscopy_vs_flux` to confirm the qubit is parked at (or near) its flux sweet spot before treating echo-time drift as a fundamental coherence problem.

## Parameter Tuning Heuristics

See also: [General parameter tuning heuristics](_general_troubleshooting.md#general-parameter-tuning-heuristics) for guidance common to most nodes in this library. Below is guidance specific to this node.

1. **Fit fails, or `T2_echo` comes back with a huge relative error** → `max_wait_time_in_ns` is likely mismatched to the true $T_2^{\rm echo}$ (remember the reported dark time is $2\times$ this parameter). If the curve hasn't visibly reached the noise floor, widen `max_wait_time_in_ns`; if it decays almost immediately, narrow it to concentrate points on the informative part of the curve.

## Next Steps

`10b_drag_calibration_180_minus_180` — the repository's bring-up calibration graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py` and `90_calibration_graph_bringup_fixed_frequency_transmon.py`) wire this node directly into the DRAG calibration that follows (`T1` → `T2echo` → `DRAG_calibration`).

## References

**[Hahn1950]** E. L. Hahn, "Spin echoes," *Phys. Rev.*, vol. 80, no. 4, pp. 580–594, 1950.

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.
