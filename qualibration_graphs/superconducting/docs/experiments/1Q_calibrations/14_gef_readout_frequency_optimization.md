# GEF Readout Frequency Optimization

[`14_gef_readout_frequency_optimization.py`](../../../../../calibrations/1Q_calibrations/14_gef_readout_frequency_optimization.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Sweeps the readout tone's frequency while preparing $|g\rangle$, $|e\rangle$, and $|f\rangle$ in turn, and picks the detuning that maximizes the *worst* of the three pairwise IQ-centroid separations.

## Purpose

`08a_readout_frequency_optimization` finds the readout frequency that best separates $|g\rangle$ and $|e\rangle$ in the IQ plane, exploiting the state-dependent dispersive shift of the resonator's dressed frequency **[BGGW2021]**. This node extends the same idea to three levels: $|g\rangle$, $|e\rangle$, and $|f\rangle$ each pull the resonator by a different amount, so in principle they occupy three distinguishable clusters in the IQ plane — but a frequency that separates $g$ from $e$ well is not guaranteed to separate $e$ from $f$ (or $g$ from $f$) equally well, since all three pairwise distances move together as the readout frequency is swept. The node's fit therefore doesn't maximize any single distance; it maximizes the **minimum** of the three pairwise distances $\{d_{ge}, d_{ef}, d_{gf}\}$ at each detuning — a maximin objective that prevents optimizing one boundary at the expense of another.

With three states instead of two, the readout-fidelity tradeoffs described for dispersive measurement **[Gam+2007]** apply with extra force: there are now more adjacent-cluster decay paths population can fall into during the finite measurement window (e.g. $f\to e$ relaxation mid-measurement lands a shot in the wrong cluster in a way that a simple g/e experiment never has to account for). Optimizing the readout frequency for joint g/e/f separation is a necessary but not sufficient step toward good three-level discrimination fidelity — it doesn't touch readout duration or power, which interact with this choice the same way they do in `08a`/`08b` **[GRTW2021]**.

## Mechanism

If a qubit's `resonator.operations` doesn't already have a `"readout_GEF"` entry, this node **creates one on the fly**: `dataclasses.replace(readout_op, length=round(readout.length × 1.5 / 4) × 4, threshold=None, rus_exit_threshold=None)` — a clone of the standard `"readout"` pulse at 1.5× its length (rounded to a multiple of 4 ns), with the binary g/e `threshold`/`rus_exit_threshold` fields explicitly cleared (they're meaningless for a three-state pulse). This is a one-time snapshot of whatever `"readout"`'s length is *at creation time*; `15_iq_blobs_gef` contains the identical snapshot logic independently, so whichever of the two nodes runs first for a given qubit is the one that actually creates `readout_GEF`.

For each (qubit, frequency detuning `df`) point, repeated `num_shots` times (averaged, not per-shot — like `08a`, this node characterizes the *mean* response at each detuning, not single-shot statistics like `15`):

1. Update the resonator's intermediate frequency to `intermediate_frequency + GEF_frequency_shift + df` (defaulting `GEF_frequency_shift` to `0` in Python if it's still `None` — a one-time in-memory default, not persisted to state).
2. **g-state block:** wait `2 × thermalization_time`, measure `operation` (default `"readout_GEF"`) → $I_g, Q_g$.
3. **e-state block:** wait `2 × thermalization_time`, play `x180`, `align()`, measure → $I_e, Q_e$.
4. **f-state block:** `qubit.reset(reset_type, simulate)` (with an extra `2 × thermalization_time` wait if `reset_type == "thermal"`), play `x180`, retune to `intermediate_frequency − anharmonicity`, play `EF_x180`, retune back, `align()`, measure → $I_f, Q_f$.

> **Verified bug: the frequency sweep only updates one qubit's resonator when `multiplexed=True`.** Step 1's `qubit.resonator.update_frequency(...)` call sits *outside* any per-qubit loop — it reuses whichever `qubit` object was last bound by the earlier `for qubit in multiplexed_qubits.values(): node.machine.initialize_qpu(...)` loop, a plain Python variable that doesn't get re-scoped inside the `with for_(*from_array(df, frequencies)):` block. With the default `multiplexed=False`, each batch has exactly one qubit, so this is harmless. With `multiplexed=True` and more than one qubit per batch, **only the last-initialized qubit in each batch actually has its resonator frequency swept**; every other qubit in that batch measures at its own already-configured (unswept) resonator frequency for the *entire* sweep, and its measured g/e/f responses will show no meaningful detuning dependence at all. Compare with `15_iq_blobs_gef`'s analogous frequency-set step, which correctly loops `for i, qubit in multiplexed_qubits.items():` — this node does not. Run with `multiplexed=False` (the default) to avoid this entirely.

Analysis (`calibration_utils/readout_gef_frequency_optimization/analysis.py`):

1. `process_raw_dataset`: `convert_IQ_to_V` on all six I/Q streams. This helper always normalizes using `qubit.resonator.operations["readout"].length` regardless of which `operation` was actually measured (`readout_GEF` by default, 1.5× longer) — see the note below on why this doesn't affect this node's own result, unlike `15`'s.
2. Per qubit (`fit_routine`): $D_{ge}=\sqrt{(I_g-I_e)^2+(Q_g-Q_e)^2}$, and likewise $D_{ef}$, $D_{gf}$; `Distance = min(D_ge, D_ef, D_gf)` at each detuning (the maximin curve).
3. `optimal_detuning = Distance.rolling(frequency=3).mean().idxmax("frequency")` — a **3-point rolling-mean-smoothed discrete argmax over the swept grid**, not a parametric fit (the function is named `fit_routine`, but there is no curve fit here). This is exactly why `frequency_step_in_mhz` defaults far finer here than in `08a` (see Parameters): with no interpolation between grid points, the achievable precision on `optimal_detuning` is set directly by the grid spacing.
4. `success` is **unconditionally set to `True`** for every qubit that appears in the returned dataset (`ds.assign({"success": True})`). The node's own docstring describes failure conditions ("A fit can fail if the sweep span is too small or SNR too low...") that are **not actually implemented** in the current source — there is no data-quality check anywhere in `fit_routine`. The only way a qubit is excluded from `fit_results` is if it's altogether missing from the dataset (logged as a warning in `_extract_relevant_fit_parameters`), which doesn't happen in normal operation.

## Prerequisites

- Resonator frequency and power roughly calibrated (`02a_resonator_spectroscopy`, `08a_readout_frequency_optimization`, `08b_readout_power_optimization`).
- `x180` calibrated (`04b_power_rabi`).
- `EF_x180` and `qubit.anharmonicity` calibrated (`13_power_rabi_ef`, `12_Qubit_Spectroscopy_E_to_F`) — the f-state block depends on both being accurate enough to actually reach $|f\rangle$.
- An existing (possibly zero/`None`) `qubit.resonator.GEF_frequency_shift`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per (qubit, detuning) point, across all three g/e/f blocks. | Linear cost in run time; three measurement blocks per detuning point rather than `08a`'s two. |
| `frequency_span_in_mhz` | `float` | 2 | MHz | Full width of the detuning sweep around the current `intermediate_frequency + GEF_frequency_shift`. | 5× narrower than `08a`'s default 10 MHz span — this node assumes the readout frequency is already close to right and is fine-tuning, not searching broadly. |
| `frequency_step_in_mhz` | `float` | 0.01 | MHz | Step size of the detuning sweep. | 10× finer than `08a`'s default 0.1 MHz — necessary because `optimal_detuning` is a raw grid argmax (see Mechanism), not an interpolated fit; coarser steps directly limit achievable precision. |
| `operation` | `Literal["readout", "readout_QND", "readout_GEF"]` | `"readout_GEF"` | – | Resonator pulse used for all three measurement blocks. | `readout_GEF` is auto-created (1.5× `readout`'s length) if missing; `readout_QND` is not created by any node in this chain — it must already exist in QUAM state if selected. |

> **`success` is always `True` in the current implementation.** There is no automatic rejection of a poorly-separated or noisy fit — see the Mechanism callout. Always inspect the `fitted_distances` plot before trusting a run.
>
> **`multiplexed=True` silently breaks the frequency sweep for all but one qubit per batch.** See the verified bug in Mechanism. Prefer `multiplexed=False` (the default) for this node until that's fixed.

## Outputs

**Measured:** `Ig`/`Qg`/`Ie`/`Qe`/`If`/`Qf` (volts, averaged over `num_shots`) at every (qubit, detuning) point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `optimal_detuning` | Hz | ✅ (additive) | Detuning at the smoothed maximin `Distance` peak. |
| `success` | – | – (always `True`) | See Mechanism — not a real gate in the current source. |

**Success criterion:** none, in effect — `success` is hardcoded `True` for every qubit present in the fit results, so the update below fires unconditionally for every targeted qubit that made it through the run.

## State Updates

Applied to every targeted qubit (in practice, since `success` is never `False`):

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.resonator.GEF_frequency_shift` | `GEF_frequency_shift + optimal_detuning` | **increment (`+=`)** | outcome not `"failed"` (in practice: always) |

Because this is an increment rather than a replace, and because there is currently no data-quality gate to stop a bad fit from updating state, re-running this node repeatedly does **not** self-correct the way `03b`'s equivalent joint-offset caveat does either — see Troubleshooting #1.

## Troubleshooting

1. **`GEF_frequency_shift` keeps drifting further with each re-run, even though the separation already looked good** → expected given the current source, not a sign of anything wrong with your setup: the update is `+=`, and `success` is unconditionally `True`, so *every* run applies another increment regardless of fit quality. Always check the `fitted_distances` plot's peak location relative to `df = 0` before re-running; if it's not converging toward zero detuning, the thing to fix is the underlying readout/EF calibration, not this node's output.
2. **Flat or noisy `Distance` curve for one qubit specifically when run with `multiplexed=True`, but clean when run alone** → this is the verified frequency-sweep bug (see Mechanism): only the last-initialized qubit in a multiplexed batch actually has its resonator frequency swept per detuning point. Re-run with `multiplexed=False`.
3. **Node crashes with a `TypeError` involving `NoneType` during `update_state`, in a multiplexed run** → a consequence of the same bug: the Python-side `if GEF_frequency_shift is None: GEF_frequency_shift = 0` guard inside `create_qua_program` also only reliably executes for the one qubit whose `update_frequency` call actually ran. Other qubits in the batch can reach `update_state`'s `+= optimal_detuning` with `GEF_frequency_shift` still `None`. Run `multiplexed=False`, or manually initialize `GEF_frequency_shift = 0` on all qubits first.
4. **`Def`/`Dgf` (the f-state distances) look small across the whole span, even at the peak** → the qubit likely isn't reaching $|f\rangle$ reliably. Re-check `EF_x180`'s amplitude (`13_power_rabi_ef`) and `anharmonicity` (`12_Qubit_Spectroscopy_E_to_F`) before trusting this node's result — it can only optimize around whatever population distribution those pulses actually prepare.
5. **Node fails at QUA-program-build time with a `KeyError` on `qubit.xy.operations["EF_x180"]`** → this node's f-state block plays `EF_x180` unconditionally; if it doesn't exist yet in QUAM state, run `13_power_rabi_ef` at least once first (even with defaults, since it auto-creates `EF_x180`).
6. **Plotted `Distance` values (in mV) look inconsistent with `15_iq_blobs_gef`'s IQ blobs for the same qubit** → `convert_IQ_to_V` always normalizes by `qubit.resonator.operations["readout"].length`, even when this node actually measured with a different-length `operation` (`readout_GEF`, 1.5× longer by default). This node's own `optimal_detuning` is unaffected — a uniform mis-scale of all six I/Q streams doesn't move where the *minimum* of the three distances peaks — but don't directly compare absolute mV-scale numbers across nodes or `operation` choices.

## Parameter Tuning Heuristics

1. **Optimal detuning jumps around between re-runs at nominally the same conditions** → `num_shots=100` averaged over three blocks per detuning point may be insufficient at `frequency_step_in_mhz=0.01`'s fine resolution; increase `num_shots`, or accept a coarser (but noisier-tolerant) step.
2. **Choosing `operation="readout"` instead of the default `readout_GEF`** → no error, but the auto-created, longer `readout_GEF` pulse (built once per qubit, shared with `15`) simply goes unused for this run; you lose the extra integration time it was sized for. Prefer the default unless deliberately comparing operations.

## Next Steps

`15_iq_blobs_gef` — uses the resulting `GEF_frequency_shift` as the fixed operating point for full three-state single-shot IQ-blob acquisition. Downstream, `13_power_rabi_ef`'s `use_state_discrimination=True` path and the two-qubit CZ calibrations (`31_chevron_11_20`, `33a_cz_leakage_amplification`, `33b_cz_leakage_amplification_palea`) call `qubit.readout_state_gef()`, which needs this node's `GEF_frequency_shift` together with `15`'s `gef_centers`.

## References

**[BGGW2021]** A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.

**[Gam+2007]** J. Gambetta, W. A. Braff, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, "Protocols for optimal readout of qubits using a continuous quantum nondemolition measurement," *Phys. Rev. A*, vol. 76, p. 012325, 2007.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.
