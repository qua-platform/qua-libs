# Single-Qubit Randomized Benchmarking — Interleaved

[`11b_single_qubit_randomized_benchmarking_interleaved.py`](../../../../../calibrations/1Q_calibrations/11b_single_qubit_randomized_benchmarking_interleaved.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Repeats the randomized-benchmarking sequence of `11a` with a single, user-chosen Clifford gate inserted between every random gate, to characterize that specific gate's contribution to the decay separately from the library-averaged Clifford fidelity `11a` reports.

## Purpose

`11a_single_qubit_randomized_benchmarking` reports one number — the average error over all 24 single-qubit Clifford gates — which is exactly what makes it insensitive to SPAM error, but also means it cannot tell you *which* gate is responsible if that number looks worse than expected. **Interleaved randomized benchmarking [Mag+2012]** answers that: interleave one specific gate-under-test between every random Clifford in the sequence, fit the resulting decay just as in standard RB **[MGE2011]**, and compare the interleaved decay rate $p_{\rm gate}$ against a *reference* decay rate $p_{\rm ref}$ measured on the same qubit under the same conditions (i.e. `11a`, run separately). The ratio $r_{\rm gate} = (1-p_{\rm gate}/p_{\rm ref})/2$ isolates the tested gate's own error, canceling out common systematics (SPAM, other gate errors) that a single interleaved sequence alone cannot distinguish from the gate's true contribution.

**This implementation computes only the interleaved half of that comparison.** It reuses `11a`'s fit code (`process_raw_dataset`, `fit_raw_data`, imported unchanged from `calibration_utils.single_qubit_randomized_benchmarking`) to fit a single exponential directly to the interleaved-sequence decay curve, and reports `1 - error_per_gate` from that fit alone — it does **not** compute the $p_{\rm gate}/p_{\rm ref}$ ratio against a separate `11a` reference run. The number this node writes to state is therefore "the fidelity of a fixed pattern where this gate is inserted after every random Clifford," not the textbook Magesan-2012 isolated single-gate error. To get that, run `11a` (reference) and this node (interleaved) under matched conditions on the same qubit and combine their fitted decay rates yourself — see Mechanism and Troubleshooting.

## Mechanism

Structurally, this node is `11a` with two changes: an interleaved gate is injected into every generated sequence, and depths are always linearly spaced (there is no log-scale option here).

1. `node.machine.initialize_qpu` sets the qubit's flux point, per batch.
2. **Sequence generation** (`generate_sequence`): builds an array of length `2 * max_circuit_depth`. For each pair of raw slots `(i, i+1)`: slot `i` gets a uniformly random Clifford (as in `11a`); slot `i+1` is *forced* to `interleaved_gate_index` — the fixed Clifford index corresponding to `interleaved_gate_operation` (via `get_interleaved_gate_index`). The running group state (and therefore the recovery-gate lookup) is updated through *both* random and interleaved steps, so the eventual recovery gate correctly accounts for the interleaved gates already played.
3. **Depths**: unlike `11a`'s `get_depths()` (log- or linear-scale, with its own validation method), `11b` always uses linear spacing, computed directly and unconditionally in `create_qua_program`: `depths = np.arange(1, max_circuit_depth + 0.1, delta_clifford)`, with `assert (max_circuit_depth / delta_clifford).is_integer()` enforced every run — there is no `log_scale` escape valve.
4. **QUA depth loop**: rather than looping only over the desired depths (as `11a` does via `from_array`), the raw sequence index `depth` is stepped one at a time from 1 to `2 * max_circuit_depth`, and the measurement block is only executed `with if_(depth == depth_target)` — `depth_target` starts at 2 and advances by `2 * delta_clifford` each time it's hit, converting "logical depth" (number of random-Clifford-plus-interleaved-gate pairs) into raw sequence-index units. This means the compiled program iterates `2 * max_circuit_depth` times per random sequence regardless of how many depths are actually measured — a real cost difference from `11a`'s more direct approach for large `max_circuit_depth`.
5. At each measured depth, `num_shots` times: reset (`qubit.reset(reset_type, simulate)`, `reset_type` honored as in `11a`), play the truncated interleaved sequence (`play_sequence` — an identical 24-way pulse-decomposition switch/case to `11a`'s, including the same wait-as-identity `case_(0)` and optional `strict_timing_()`), then measure (`state` or raw `I`/`Q`).

Analysis is byte-for-byte the same as `11a`'s (imported, not reimplemented): fit a single `decay_exp` to the interleaved-sequence decay curve, then `error_per_clifford = (1-\alpha)/2` and `error_per_gate = error_per_clifford / 1.875` — the same 1.875-physical-pulses-per-Clifford conversion factor used for a *standalone* random Clifford, applied here unmodified even though every logical depth step in this sequence also contains one extra, deterministic, always-1-pulse interleaved gate that the divisor doesn't separately account for.

`interleaved_gate_operation` is restricted (`Literal["I","x180","y180","x90","-x90","y90","-y90"]`, default `"x180"`) to the 7 Cliffords realizable as a single physical pulse or a true identity — the multi-pulse composite Cliffords (e.g. the 3-pulse `Z90`/`-Z90` cases in `play_sequence`) cannot be targeted this way.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration` or the MW-FEM equivalent).
- Precisely calibrated qubit parameters: pulse amplitude (`04b_power_rabi`) and frequency/phase (`06a_ramsey`).
- Optional: readout optimization (`08a_readout_frequency_optimization`, `08b_readout_power_optimization`).
- Optional: DRAG calibration (`10a`/`10b`/`10c`) — especially relevant if `interleaved_gate_operation` is a DRAG-sensitive pulse.
- `qubit.z.flux_point` set to the intended value if flux-tunable.
- IQ-blob calibration (`07_iq_blobs`) is only required if you explicitly turn on `use_state_discrimination` — it defaults to `False` here (see the docstring callout below), unlike `11a`.
- A prior or concurrent `11a` run on the same qubit under matched conditions (`reset_type`, `multiplexed`, flux point, `use_state_discrimination`) is not enforced by the code but is required to interpret this node's result as an isolated gate error at all — see Purpose.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below (`calibration_utils/single_qubit_randomized_benchmarking_interleaved/parameters.py`). Note that `11b` has **no `log_scale` parameter** — depths are always linear (see Mechanism #3).

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `interleaved_gate_operation` | `Literal["I","x180","y180","x90","-x90","y90","-y90"]` | `"x180"` | – | The single-qubit gate to interleave between every random Clifford. | Selects which of the 7 single-pulse-realizable Cliffords is inserted; determines the dict key written under `qubit.gate_fidelity` (see State Updates). Composite multi-pulse Cliffords cannot be selected. |
| `use_state_discrimination` | `bool` | `False` | – | Return discriminated `state` instead of raw `I`/`Q`. | See docstring-mismatch callout below — the field's own docstring claims the default is `True`; it is not. |
| `use_strict_timing` | `bool` | `False` | – | Wrap `play_sequence` in `strict_timing_()`. | Same effect as in `11a`: forces gap-free playback, at the risk of a compile error if the schedule can't fit. |
| `num_random_sequences` | `int` | 100 | – | Number of independently-drawn random sequences (each with the interleaved gate inserted). | Statistical-averaging axis for the interleaved decay fit; lower default than `11a`'s 300, reflecting the smaller default `max_circuit_depth`/run-time budget. |
| `num_shots` | `int` | 20 | – | Averages per (sequence, depth) point, averaged away on the FPGA (not a retained dimension). | Same role as in `11a`; higher default (20 vs. `11a`'s 10). |
| `max_circuit_depth` | `int` | 1000 | – | Longest *logical* sequence (random-Clifford + interleaved-gate pairs) before the recovery gate. | Sizes the raw sequence array as `2 * max_circuit_depth + 1` internally. Lower default than `11a`'s 2048. |
| `delta_clifford` | `int` | 20 | – | Spacing between logical depths — **always used**, unlike `11a` where it's ignored in the default log-scale mode. | Sets `num_depths = max_circuit_depth // delta_clifford`; must divide `max_circuit_depth` evenly (see callout below). |
| `seed` | `Optional[int]` | `None` | – | Seed for QUA's `Random()` generator. | Same behavior as `11a`: `None` draws a fresh Python-side seed at every `create_qua_program` call. Fix explicitly if you need the interleaved run's randomness to be reproducible run-over-run (it does *not* need to match `11a`'s seed for the ratio comparison to be statistically valid — only the conditions need to match). |

> **Source docstring is stale: `use_state_discrimination` default.** The field's own docstring in `calibration_utils/single_qubit_randomized_benchmarking_interleaved/parameters.py` reads `"""Perform qubit state discrimination. Default is True."""`, but the actual Pydantic field default is `use_state_discrimination: bool = False`. An agent reading only the docstring would wrongly assume state discrimination is on by default for this node — it is not (contrast `11a`, where the default genuinely is `True`).

> **`max_circuit_depth / delta_clifford` must be an integer, unconditionally.** The assertion is inline in `create_qua_program` (not behind a `log_scale` toggle as in `11a`, since this node has none) and is checked on every run. The defaults (`1000 / 20 = 50`) satisfy it, but a custom `max_circuit_depth` or `delta_clifford` that doesn't divide evenly raises `AssertionError` before any hardware program is built.

## Outputs

**Measured:** `I`/`Q` (or discriminated `state` if `use_state_discrimination`), shot-averaged on the FPGA, indexed by `(qubit, nb_of_sequences, depths)`.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `fit_data` (`a`, `offset`, `decay`, + 9 covariance terms) | – | – | Raw `curve_fit` output for the interleaved-sequence decay, per qubit. |
| `error_per_clifford` | – | – | $(1-\alpha)/2$ with $\alpha=e^{\text{decay}}$, fit to the interleaved decay curve alone (**not** a Magesan-2012 ratio against `11a`'s reference decay — see Mechanism/Purpose). |
| `error_per_gate` | – | ✅ (as `1 - error_per_gate`) | `error_per_clifford / 1.875`, same conversion factor `11a` uses for a standalone Clifford. |
| `success` | bool | – | Same criterion as `11a`. |

**Success criterion:** identical to `11a` — `error_per_gate` non-NaN and strictly $0 < {\tt error\_per\_gate} < 1$.

## State Updates

Applied only when the fit succeeds — failed qubits are skipped entirely:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.gate_fidelity[interleaved_gate_operation]` | `1 - error_per_gate` | replace | outcome successful |

> **Source docstring is stale: state-update key name.** The node module's own docstring says `"State update: - The averaged single qubit gate fidelity: qubit.gate_fidelity["interleaved_gate"]."` — a fixed, literal `"interleaved_gate"` key. The actual code (`update_state`) writes `q.gate_fidelity[node.parameters.interleaved_gate_operation] = ...`, i.e. a **dynamically-named key equal to whichever gate was tested** (`"x180"`, `"y90"`, etc.), not a fixed `"interleaved_gate"` string. This is actually convenient in practice — running this node once per gate of interest accumulates separate entries in the same `gate_fidelity` dict rather than overwriting a single shared key — but don't expect the literal key name the docstring shows.

## Troubleshooting

See also: [General troubleshooting](_general_troubleshooting.md#general-troubleshooting) for issues common to most nodes in this library. Below are issues specific to this node.

1. **`qubit.gate_fidelity[gate]` from this node never differs meaningfully from `qubit.gate_fidelity["averaged"]` from `11a`, even for a gate you know is worse (e.g. one with a stale DRAG calibration)** → expected, given how the number is computed (see Purpose): this node fits the interleaved decay curve alone, it does not divide out `11a`'s reference decay rate. To isolate the gate's own error per **[Mag+2012]**, take $\alpha_{\rm gate}=e^{\rm decay}$ from this node and $\alpha_{\rm ref}=e^{\rm decay}$ from a matched `11a` run and compute $r_{\rm gate} = (1-\alpha_{\rm gate}/\alpha_{\rm ref})/2$ yourself — the node does not do this for you and there is no field for it.
2. **`ValueError`/`AssertionError` at `create_qua_program`, before any hardware access** → two independent causes to check: (a) `interleaved_gate_operation` set outside the 7 allowed values raises inside `get_interleaved_gate_index`; (b) `max_circuit_depth / delta_clifford` not an integer raises the unconditional assertion (see Parameters callout) — there is no `log_scale` fallback to bypass it here.
3. **Compilation or simulation is noticeably slower than `11a` for a comparable depth range** → this node's QUA loop steps the raw sequence index one at a time up to `2 * max_circuit_depth`, checking `if_(depth == depth_target)` at every step, rather than looping only over the measured depths via `from_array` as `11a` does (see Mechanism #4). Large `max_circuit_depth` costs proportionally more compiled steps here.
4. **`use_state_discrimination` behaves as `False` even though you expected `True` by default** → see the docstring-mismatch callout: the Pydantic default is genuinely `False` here (unlike `11a`'s `True`). Set it explicitly if you want the cleaner, bounded decay curve discrimination provides, and confirm `07_iq_blobs` is calibrated first.
5. **You expect a `qubit.gate_fidelity["interleaved_gate"]` key after running this node and don't find it** → see the state-update docstring callout: the actual key is the tested gate's name (`"x180"`, `"y90"`, ...), taken from `interleaved_gate_operation`, not the literal string `"interleaved_gate"` the module docstring describes.
6. **Decay curve deviates from a single exponential** (flattened asymptote, kink) **especially with `use_state_discrimination=True`** → as in `11a`, likely leakage misclassified by the 2-state discriminator, potentially made worse here if `interleaved_gate_operation` is itself a leakage-prone pulse **[MGRW2009]**. Compare against `use_state_discrimination=False` and check DRAG calibration for the specific interleaved gate.
7. **Fidelity for the interleaved gate looks suspiciously close to `11a`'s averaged fidelity regardless of which gate you pick** → also expected, and a corollary of #1: since this node's raw `error_per_gate` isn't a ratio against a reference, and every interleaved sequence still contains just as many *random* Cliffords as an `11a` sequence of the same logical depth, the interleaved-only fit is dominated by the same average random-Clifford error that `11a` measures — the interleaved gate's own (typically much smaller) contribution is a comparatively small perturbation on top of that, which is exactly why the Magesan-2012 ratio (not a bare fit) is needed to see it clearly.

## Parameter Tuning Heuristics

See also: [General parameter tuning heuristics](_general_troubleshooting.md#general-parameter-tuning-heuristics) for guidance common to most nodes in this library. Below is guidance specific to this node.

1. **Manually combining an `11a` reference run with this node's interleaved run produces a negative or implausibly-small isolated gate error** → per **[Mag+2012]**, this is a known statistical-estimator artifact of the ratio method when the tested gate is genuinely very high-fidelity — not evidence of "negative error." Increase `num_random_sequences` on both runs (or repeat and average) rather than distrusting the underlying gate calibration.

## Next Steps

Not wired into any of this repository's automated calibration graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py`, `81_calibration_graph_retuning_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`, `91_calibration_graph_retuning_fixed_frequency_transmon.py`) — all of them terminate at `11a_single_qubit_randomized_benchmarking`. This node is a manual diagnostic step: run it after `11a`, under matched conditions on the same qubit, once per gate of interest, whenever `11a`'s averaged Clifford fidelity is worse than expected and you need to localize the problem to a specific single-qubit gate (see Mechanism/Purpose for how to combine the two results into a genuine isolated-gate error estimate).

## References

**[Mag+2012]** E. Magesan, J. M. Gambetta, B. R. Johnson, C. A. Ryan, J. M. Chow, S. T. Merkel, M. P. da Silva, G. A. Keefe, M. B. Rothwell, T. A. Ohki, M. B. Ketchen, and M. Steffen, "Efficient measurement of quantum gate error by interleaved randomized benchmarking," *Phys. Rev. Lett.*, vol. 109, p. 080505, 2012.

**[MGE2011]** E. Magesan, J. M. Gambetta, and J. Emerson, "Scalable and robust randomized benchmarking of quantum processes," *Phys. Rev. Lett.*, vol. 106, p. 180504, 2011.

**[MGRW2009]** F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm, "Simple pulses for elimination of leakage in weakly nonlinear qubits," *Phys. Rev. Lett.*, vol. 103, no. 11, p. 110501, 2009.
