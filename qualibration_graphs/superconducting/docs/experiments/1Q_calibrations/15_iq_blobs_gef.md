# IQ Blobs GEF

[`15_iq_blobs_gef.py`](../../../../../calibrations/1Q_calibrations/15_iq_blobs_gef.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Acquires single-shot $|g\rangle$/$|e\rangle$/$|f\rangle$ IQ blobs at the calibrated three-level readout frequency and fits each state's centroid, for use in runtime three-state discrimination.

## Purpose

`07_iq_blobs` characterizes single-shot g/e discrimination: repeated shots at each state trace out two clusters in the IQ plane, separated by the dispersive shift $\chi$ **[BGGW2021]**, and any real classifier has to treat this as a statistical problem rather than a deterministic one, since the two clusters have finite width and can overlap **[Gam+2007]**. This node extends that same single-shot technique to three states, using the readout frequency `14_gef_readout_frequency_optimization` already tuned to maximize the worst-case pairwise separation among $g$, $e$, and $f$. With three clusters instead of two there are more ways for a given shot to land in the wrong one — in particular, relaxation during the measurement window can move population from $|f\rangle$ toward $|e\rangle$ or $|g\rangle$ mid-shot, and from $|e\rangle$ toward $|g\rangle$, in ways a two-state experiment never has to account for **[Gam+2007]**, **[GRTW2021]**. Fitting each cluster's centroid here is what makes real-time, three-outcome discrimination (`qubit.readout_state_gef`) possible at all.

## Mechanism

**This node enforces `reset_type == "thermal"` exactly** — any other value raises `ValueError("Only 'thermal' reset is supported")` in `create_qua_program`, before any hardware access. Unlike `13`/`14`, `"active_gef"` is *not* accepted here.

If a qubit's `resonator.operations` doesn't already have a `"readout_GEF"` entry, this node creates one — identical logic to `14_gef_readout_frequency_optimization`'s auto-creation step (`dataclasses.replace(readout, length=1.5×readout.length rounded to 4 ns, threshold=None, rus_exit_threshold=None)`), duplicated independently in both nodes rather than shared. Whichever of `14`/`15` runs first for a given qubit is the one that actually creates it; the second just finds it already there and moves on.

Before the shot loop, **per-qubit** (correctly scoped, unlike `14`'s analogous step — see that node's Troubleshooting): set the resonator's intermediate frequency to `intermediate_frequency + GEF_frequency_shift` (falling back to `0` if `GEF_frequency_shift` is still `None`). For each of `num_shots` shots (default 2000, comparable to `07_iq_blobs`'s default, but with three measurement blocks per shot instead of two):

1. **g-state block:** wait `2 × thermalization_time`, measure `operation` (default `"readout_GEF"`) → $I_g, Q_g$.
2. **e-state block:** wait `2 × thermalization_time`, play `x180`, `align()`, measure → $I_e, Q_e$.
3. **f-state block:** wait `2 × thermalization_time` (a plain wait here, not `qubit.reset()` — this block relies on the preceding waits/measurement for reset, since `reset_type` is fixed to `"thermal"` anyway), play `x180`, retune to `intermediate_frequency − anharmonicity`, play `EF_x180`, retune back, `align()`, measure → $I_f, Q_f$.

Unlike `14`, results are **not averaged** — each shot's $I_g,Q_g,I_e,Q_e,I_f,Q_f$ is streamed individually (`buffer(n_runs).save(...)`, no `.average()`), giving the per-shot histogram this node's fit needs.

Analysis (`calibration_utils/iq_blobs_ef/analysis.py`):

1. `process_raw_dataset`: unwraps any tuple-valued elements in the raw dataset, then `convert_IQ_to_V`.
2. `fit_gaussian_centers`: for each of the six I/Q streams *independently*, bins into a 100-bin histogram and fits a **single 1D Gaussian** (`scipy.optimize.curve_fit`) to find its peak (`find_biggest_gaussian`). This is six independent 1D fits, not a joint 2D Gaussian and not a mixture model — it implicitly assumes each marginal histogram is dominated by one clean peak.
3. `success` is **unconditionally `True`** for every qubit that survives step 2 — the same pattern as `14_gef_readout_frequency_optimization`. There is no data-quality check.
4. `center_matrix` bundles the three $(I,Q)$ centroids into a 3×2 array per qubit, used for both plotting and the state update below.

> **No graceful failure path for a bad Gaussian fit.** `find_biggest_gaussian`'s `curve_fit` call has no `try`/`except`. If any one of the six marginal histograms doesn't look like a clean single Gaussian — e.g. heavily overlapping $e$/$f$ blobs along a given quadrature, too few shots, or a poorly-chosen `GEF_frequency_shift` — `analyse_data` raises an uncaught `scipy` `RuntimeError` and the whole node run fails, rather than marking that one qubit `"failed"` and continuing with the rest.

> **Verified units mismatch affecting `gef_centers`.** `convert_IQ_to_V` (called from `process_raw_dataset`) always normalizes by `qubit.resonator.operations["readout"].length`, *regardless* of which `operation` was actually measured. This node measures with `node.parameters.operation` (default `"readout_GEF"`, auto-created at 1.5× `readout`'s length). `update_state` then converts the fitted volts-domain centers back to "raw ADC units" — the comment in source is explicit about this intent — using `operation.length / 2**12` (i.e. `readout_GEF`'s length). Because the forward conversion used `readout.length` while the inverse uses `operation.length`, the round trip does **not** cancel: whenever `operation != "readout"` (the default), the persisted `gef_centers` end up scaled by `operation.length / readout.length` (1.5× by default) relative to the raw units `qubit.readout_state_gef()` actually compares against at runtime. This is a real, source-verified inconsistency between the forward (`convert_IQ_to_V`) and inverse (`update_state`) conversions, not a hypothetical — see Troubleshooting.

> **The runtime discrimination metric is Manhattan (L1) distance, not the "squared Euclidean distance" its own docstring claims.** `readout_state_gef` (`quam_builder`'s `BaseTransmon`) classifies each shot as `argmin` over `|I − center_I| + |Q − center_Q|` for the three stored `gef_centers` — an L1 metric with diamond-shaped decision boundaries, not the circular boundaries a Euclidean metric would give. This matters if a state's noise blob is elongated along one quadrature more than the other.

The `plot_confusion_matrices` figure this node produces uses yet a **third** metric — Euclidean nearest-center classification computed live on the raw dataset, purely for the plot — and is **not persisted** anywhere: `qubit.resonator.gef_confusion_matrix` is a real QUAM field (on `ReadoutResonatorBase`) but this node never writes it. The printed/plotted confusion matrix is therefore only an approximate, Euclidean-metric proxy for the L1-metric fidelity `readout_state_gef` will actually deliver at runtime.

## Prerequisites

- Readout parameters calibrated (`02a_resonator_spectroscopy`, and/or `02b`/`02c`).
- `x180` calibrated (`04b_power_rabi`).
- `EF_x180` calibrated (`13_power_rabi_ef`) and `anharmonicity` known (`12_Qubit_Spectroscopy_E_to_F`).
- Ideally, `GEF_frequency_shift` already optimized (`14_gef_readout_frequency_optimization`) — without it, this node still runs (falling back to a `0` shift) but the readout frequency won't be the jointly-optimized one, weakening separation between clusters.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 2000 | – | Number of single shots per qubit, each comprising a g/e/f block. | Same default as `07_iq_blobs`, but each shot here is ~1.5× longer (three blocks vs. two); more shots give cleaner Gaussian histograms for `find_biggest_gaussian`. |
| `operation` | `Literal["readout", "readout_QND", "readout_GEF"]` | `"readout_GEF"` | – | Resonator pulse used for all three measurement blocks. | `readout_GEF` is auto-created (1.5× `readout`'s length) if missing — see the units-mismatch callout above for why the choice here matters beyond just SNR. |

> **`reset_type` must be exactly `"thermal"`.** Any other value raises `ValueError` immediately — this node does not accept `"active"` or `"active_gef"` at all (contrast with `13`/`14`, which do accept `"active_gef"`).
>
> **`success` is always `True`.** As with `14`, there is no automatic rejection of a poor fit short of an outright crash (see Mechanism) — always inspect `plot_iq_blobs`/`plot_confusion_matrices` before trusting a run.

## Outputs

**Measured:** `Ig`/`Qg`/`Ie`/`Qe`/`If`/`Qf` (volts) for every one of `num_shots` shots.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `I_g_center`, `Q_g_center` | V | via `gef_centers` (rescaled) | Fitted $|g\rangle$ blob centroid. |
| `I_e_center`, `Q_e_center` | V | via `gef_centers` (rescaled) | Fitted $|e\rangle$ blob centroid. |
| `I_f_center`, `Q_f_center` | V | via `gef_centers` (rescaled) | Fitted $|f\rangle$ blob centroid. |
| `success` | – | – (always `True`) | See Mechanism — not a real gate in the current source. |

**Success criterion:** none, in effect — `success` is hardcoded `True` for every qubit that survives the Gaussian fit (a genuinely bad fit crashes the node outright rather than being flagged as failed — see Mechanism).

## State Updates

Applied to every targeted qubit (in practice, since a qubit that reaches this step already has `success = True`):

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.resonator.gef_centers` | `(center_matrix_volts × operation.length / 2**12).tolist()` | replace | outcome not `"failed"` (in practice: always) |

`qubit.resonator.gef_confusion_matrix` is **not** written by this node, despite `plot_confusion_matrices` computing an equivalent matrix for display — see the Mechanism callout.

## Troubleshooting

1. **Node crashes with a `scipy` `RuntimeError` (e.g. "Optimal parameters not found") during `analyse_data`** → one of the six marginal I/Q histograms doesn't look like a clean single Gaussian to `find_biggest_gaussian` — commonly, heavily overlapping $e$/$f$ blobs along one quadrature, too few shots, or a `GEF_frequency_shift` that isn't actually optimized yet. Confirm `14_gef_readout_frequency_optimization` has run, or inspect a re-run with `load_data_id` to see which stream is problematic before it crashes again.
2. **`readout_state_gef`-based discrimination performs much worse in practice (e.g. in `13`'s `use_state_discrimination=True` path, or a CZ leakage node) than `plot_confusion_matrices` suggested** → that plot classifies with Euclidean nearest-center distance, but the actual runtime code (`readout_state_gef`) classifies with Manhattan (L1) distance — different decision boundaries. Don't treat this node's printed confusion matrix as a guarantee of real-time fidelity.
3. **Discrimination is systematically biased toward one state** (e.g. shots that should read $|f\rangle$ consistently read as $|e\rangle$ or $|g\rangle$) even though the IQ blobs plot looks well-separated → suspect the verified units mismatch in `gef_centers`: if `operation != "readout"` (the default `readout_GEF` is 1.5× longer), the persisted centers are scaled by `operation.length / readout.length` relative to the raw units `readout_state_gef` expects. Check `qubit.resonator.operations["readout_GEF"].length / qubit.resonator.operations["readout"].length` — if it's not `1.0`, treat `gef_centers` as suspect and consider manually rescaling, or aligning the two conversions in source.
4. **Node raises `ValueError("Only 'thermal' reset is supported")` immediately** → `reset_type` was set to something other than `"thermal"`. This node cannot use `"active"` or `"active_gef"` at all; set `reset_type="thermal"`.
5. **The $|f\rangle$ blob sits suspiciously close to the $|e\rangle$ blob (weak separation, not a fitting crash)** → this is a state-preparation problem, not a readout problem: check `EF_x180`'s amplitude (`13_power_rabi_ef`) and `anharmonicity` (`12_Qubit_Spectroscopy_E_to_F`) — this node only measures whatever population distribution those pulses actually prepare.
6. **A qubit's fit "succeeds" but the plotted blobs are visibly overlapping garbage** → `success` is hardcoded `True` for any qubit that doesn't outright crash the Gaussian fit; there is currently no automatic rejection of a bad-but-technically-fittable result. Always visually inspect `plot_iq_blobs`/`plot_confusion_matrices` before trusting `gef_centers`.
7. **Run feels much slower than `07_iq_blobs` at the same `num_shots`** → expected: each shot here comprises three measurement blocks (g/e/f), each with its own `2 × thermalization_time` wait, vs. `07`'s two blocks — roughly 1.5× the per-shot time.
8. **`readout_GEF` doesn't seem to reflect a recently-changed `readout` pulse length** → `readout_GEF` is a one-time snapshot created (by whichever of `14`/`15` ran first for that qubit) at 1.5× whatever `readout.length` was *then* — it does not automatically track later changes to `readout`. Manually delete/replace the qubit's `readout_GEF` entry in QUAM state to force a fresh clone at the current `readout` length.

## Parameter Tuning Heuristics

1. **A `scipy` `RuntimeError` crash during `analyse_data` (see Troubleshooting #1) traces to too few shots feeding a noisy marginal histogram** → increase `num_shots` so `find_biggest_gaussian` has a cleaner Gaussian to fit.

## Next Steps

`gef_centers` (from this node) together with `GEF_frequency_shift` (from `14`) are what make `qubit.readout_state_gef()` usable at runtime. Verified downstream consumers: `13_power_rabi_ef`'s `use_state_discrimination=True` path, `reset_qubit_active_gef` (used wherever `reset_type="active_gef"` is selected in nodes that support it), and the two-qubit CZ calibrations `31_chevron_11_20`, `33a_cz_leakage_amplification`, and `33b_cz_leakage_amplification_palea` (which call `readout_state_gef()` directly; `33b` also plays `EF_x180` directly).

## References

**[BGGW2021]** A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.

**[Gam+2007]** J. Gambetta, W. A. Braff, A. Wallraff, S. M. Girvin, and R. J. Schoelkopf, "Protocols for optimal readout of qubits using a continuous quantum nondemolition measurement," *Phys. Rev. A*, vol. 76, p. 012325, 2007.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.
