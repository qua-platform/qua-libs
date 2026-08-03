# EF Power Rabi

[`13_power_rabi_ef.py`](../../../../../calibrations/1Q_calibrations/13_power_rabi_ef.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Populates $|e\rangle$ with a calibrated `x180`, then sweeps the amplitude of an anharmonicity-shifted drive to calibrate the `EF_x180` $\pi$ pulse between $|e\rangle$ and $|f\rangle$.

## Purpose

This is the e↔f analog of `04b_power_rabi`: at a fixed pulse duration, driving strength is swept and the resulting population traces out $\sin^2$-like Rabi fringes as the drive amplitude passes through successive multiples of a $\pi$ rotation **[AE1987]**. Because $f_{12}$ sits below $f_{01}$ by the anharmonicity $\alpha$ **[Koc+2007]**, the drive has to first put population into $|e\rangle$ (there is nothing to Rabi-drive between $|e\rangle$ and $|f\rangle$ starting from $|g\rangle$) and then be retuned to the e↔f frequency before the amplitude sweep proper begins — the same two-step logic as `12_Qubit_Spectroscopy_E_to_F`, just fitting amplitude instead of frequency.

The node also auto-creates the `EF_x180` operation itself the first time it's needed, by cloning `qubit.xy.operations["x180"]` and zeroing its DRAG coefficient (`alpha`) if the pulse type has one (see Mechanism). That's a deliberate physical choice, not an oversight: DRAG's leakage-suppression term is derived to cancel population leaking *out of* the computational $\{g,e\}$ subspace and *into* $|f\rangle$ during a g↔e pulse — reusing that same correction on a pulse that is intentionally driving straight into $|f\rangle$ would fight the pulse's own purpose, so the safer default is to start from an uncorrected clone and let this node's amplitude fit take care of calibrating it on its own terms.

## Mechanism

`calibration_utils/power_rabi/parameters.py` defines `BasePowerRabiParameters` (amplitude-sweep fields) shared between `04b_power_rabi` and this node; `04b` adds `operation`/`max_number_pulses_per_sweep`/`update_x90` on top via `NodeSpecificParameters`, while this node uses the bare `EfNodeSpecificParameters` — no operation choice, no error-amplification knobs. The QUA sequence, per (qubit, amplitude-prefactor) point, repeated `num_shots` times:

1. **Validate `reset_type` immediately, before any QUA program is built.** If `reset_type == "active"`, the node raises `ValueError("'active' is not supported, use 'thermal' or 'active_gef' instead")` — see Parameters.
2. **Auto-create `EF_x180` if it doesn't already exist**: `x180 = qubit.xy.operations["x180"]`, then `EF_x180 = dataclasses.replace(x180, alpha=0.0)` if the pulse class has an `alpha` (DRAG) field, else a plain `dataclasses.replace(x180)`. This is a **one-time snapshot**: it clones whatever `x180`'s length/shape/amplitude are *at the moment this node first runs for that qubit*. If `x180` is later re-calibrated with a different duration or envelope, `EF_x180` does **not** automatically follow — see Troubleshooting.
3. Per qubit, once per batch: set the resonator's intermediate frequency to `intermediate_frequency + (GEF_frequency_shift if use_state_discrimination else chi)`. With the default `use_state_discrimination=False`, this uses `qubit.chi` (the plain g/e dispersive shift from `08a_readout_frequency_optimization`); with state discrimination on, it uses `qubit.resonator.GEF_frequency_shift` (from `14_gef_readout_frequency_optimization`) instead.
4. Per shot: `qubit.reset(reset_type, simulate)`; if `reset_type == "thermal"`, an **additional** full `thermalization_time` wait is added on top of `reset_qubit_thermal()`'s own wait — the comment in source calls this "twice the regular thermal time for proper $|f\rangle$ state reset." (`"active_gef"` doesn't get this extra wait: its own measurement-feedback loop already verifies the qubit is back in $|g\rangle$ before returning.)
5. `align()`; reset `qubit.xy`'s IF to base, play `x180` (→ $|e\rangle$); retune to `intermediate_frequency − anharmonicity`; play `EF_x180` at amplitude scale `a` (the swept `amp_prefactor`).
6. `align()`; read out via `qubit.readout_state_gef(state)` if `use_state_discrimination`, else `qubit.resonator.measure("readout", ...)`.

> **Dead code: no pulse-repetition sweep actually happens.** The source computes `N_pi_vec = get_number_of_pulses(node.parameters)` and declares QUA variables `npi`/`count`, clearly carried over from `04b_power_rabi`'s error-amplification structure. But `EfNodeSpecificParameters` has neither `max_number_pulses_per_sweep` nor `operation`, so `get_number_of_pulses`'s `HasErrorAmplification` structural-typing check fails and it unconditionally returns `np.array([1])`. More importantly, the QUA program itself never loops over `N_pi_vec`/`npi`/`count` at all — there is no `nb_of_pulses` sweep axis, and `EF_x180` is played exactly once per shot regardless. `N_pi_vec`, `npi`, and `count` are computed/declared and then simply unused.

Analysis (`calibration_utils/power_rabi/analysis.py`, shared with `04b` but branching on `node.name == "13_power_rabi_ef"`):

1. `process_raw_dataset`: converts I/Q to volts (unless state-discriminated), computes `full_amp = amp_prefactor × qubit.xy.operations["EF_x180"].amplitude` (vs. `04b`'s `operations[node.parameters.operation].amplitude` — there's no `operation` parameter here, so it's hardcoded to `EF_x180`), and — specifically for this node — also adds `IQ_abs`/`phase` via `add_amplitude_and_phase`.
2. `fit_raw_data`: since `max_number_pulses_per_sweep` doesn't exist on `EfParameters`, `getattr(..., "max_number_pulses_per_sweep", 1)` always reads `1`, taking the single-pulse branch unconditionally. It fits a cosine (`fit_oscillation`) to **`IQ_abs`** (not the rotated `I` that `04b`/`03a` use — there's no `iw_angle` rotation computed for the EF transition here, so the raw demodulated amplitude is used instead) or to `state` if discriminated.
3. `opt_amp_prefactor = (π − φ_wrapped) / (2π f)` from the fitted phase/frequency, then `opt_amp = opt_amp_prefactor × current EF_x180 amplitude` — same formula as `04b`.
4. Success: `opt_amp`/`opt_amp_prefactor` not `NaN`, **and** `opt_amp < instrument_limits(qubit.xy).max_x180_wf_amplitude` (using the *first* qubit's channel-type limit for every qubit in the batch — the same pattern documented for `03a`/`04b`). Unlike `12`/`14`/`15`, this is a real, meaningful gate.

## Prerequisites

- A calibrated `x180` pulse (`04b_power_rabi`) — both the state-preparation pulse and the template `EF_x180` is cloned from.
- The qubit's anharmonicity (`12_Qubit_Spectroscopy_E_to_F`) — needed to place the EF drive frequency.
- The readout resonator's dispersive shift `chi` (`08a_readout_frequency_optimization`) — used to set the readout tone when `use_state_discrimination=False` (the default).
- If enabling `use_state_discrimination=True`: `qubit.resonator.GEF_frequency_shift` (`14_gef_readout_frequency_optimization`) and `qubit.resonator.gef_centers` (`15_iq_blobs_gef`) must already be set — see Troubleshooting.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus `BasePowerRabiParameters` below. There is no node-specific `operation` or error-amplification parameter (unlike `04b`).

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 50 | – | Averages per amplitude-prefactor point. | Linear cost in run time; note the per-shot cost is already higher than `04b`'s due to the extra thermalization wait (see Mechanism). |
| `min_amp_factor` | `float` | 0.001 | – | Lower bound of the amplitude-prefactor sweep. | Near-zero drive. |
| `max_amp_factor` | `float` | 1.99 | – | Upper bound of the amplitude-prefactor sweep (QUA `amplitude_scale` hard limit is `[-2, 2)`). | If the fit lands at this edge, widen the range. |
| `amp_factor_step` | `float` | 0.005 | – | Step size of the amplitude sweep. | ~400 points across the default span. |

> **`reset_type = "active"` raises `ValueError` immediately.** Only `"thermal"` and `"active_gef"` are accepted; the check happens in `create_qua_program`, before any hardware access, so the failure is immediate and unambiguous. This is a hard validation, not a soft warning.
>
> **The pulse played is always `EF_x180` — there is no `operation` choice.** `04b_power_rabi`'s `operation: Literal["x180","x90",...]` field simply doesn't exist on `EfNodeSpecificParameters`; this node's amplitude sweep and state update always target `EF_x180`.
>
> **`use_state_discrimination=True` requires `GEF_frequency_shift` to already be a number, not `None`.** The resonator IF offset is computed as `intermediate_frequency + qubit.resonator.GEF_frequency_shift` in plain Python during `create_qua_program` (not inside the QUA program) — if `14_gef_readout_frequency_optimization` hasn't run yet and `GEF_frequency_shift` is still its default `None`, this raises a `TypeError` (`unsupported operand type(s) for +: 'int' and 'NoneType'`) at graph-build time, before simulation or execution. Run `14` first, or leave `use_state_discrimination=False` (which uses `qubit.chi` instead and has no such dependency).

## Outputs

**Measured:** `I`/`Q`/`IQ_abs`/`phase` (or discriminated `state`), `full_amp`, at every amplitude-prefactor point.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `opt_amp_prefactor` | – | – | Amplitude scale factor (relative to `EF_x180`'s *current* amplitude) estimated to deliver a $\pi$ rotation. |
| `opt_amp` | V | ✅ | Absolute amplitude: `opt_amp_prefactor × current EF_x180 amplitude`. |
| `operation` | – | – (metadata) | Always `"EF_x180"` for this node. |
| `success` | – | – (gates the update) | See criterion below. |

**Success criterion:** `opt_amp`/`opt_amp_prefactor` not `NaN`, **and** `opt_amp < instrument_limits(qubit.xy).max_x180_wf_amplitude` (first qubit's channel-type limit applied across the whole batch).

## State Updates

Applied only when the fit succeeds:

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.xy.operations["EF_x180"].amplitude` | fitted `opt_amp` | replace | outcome successful |

There is no EF-equivalent `x90` write — this node calibrates `EF_x180` only.

## Troubleshooting

1. **Node raises `ValueError` immediately on start, before any hardware or simulation activity** → `reset_type="active"` was passed. Use `"thermal"` (default) or `"active_gef"` instead — this node explicitly rejects `"active"`.
2. **Node raises a Python `TypeError` while building the QUA program** (not during execution) → `use_state_discrimination=True` was set but `qubit.resonator.GEF_frequency_shift` is still `None`. Run `14_gef_readout_frequency_optimization` at least once first, or set `use_state_discrimination=False` to fall back to `qubit.chi`.
3. **No Rabi oscillation visible anywhere in the amplitude sweep** → first confirm `x180` is genuinely populating $|e\rangle$ (re-check `04b_power_rabi`); if that's fine, check that the anharmonicity used to compute the EF drive frequency (`12_Qubit_Spectroscopy_E_to_F`) is accurate — a large error there means the drive isn't landing on the e↔f transition at all, and no amplitude will produce a clean oscillation.
4. **Weak, noisy oscillation that looks like population is "stuck" partway rather than cycling cleanly** → likely a residual EF drive-frequency error (refine `12`'s anharmonicity fit) or, if `use_state_discrimination=True`, unreliable `gef_centers` (`15_iq_blobs_gef` not yet accurate) corrupting the discriminated `state` readout.
5. **Just re-calibrated `x180`'s duration/shape (e.g. via `04a_rabi_chevron`), but `EF_x180` doesn't seem to reflect it** → expected: `EF_x180` is a one-time clone of `x180` created the *first* time this node ran for that qubit (`dataclasses.replace`). It does not track subsequent changes to `x180`. Manually clear/replace `qubit.xy.operations["EF_x180"]` in QUAM state (or delete the entry so this node re-creates it fresh) to force it to pick up `x180`'s current length/shape.
6. **`reset_type="active_gef"` selected to speed up resets, but reset fidelity or overall calibration seems to get *worse*, not better** → `reset_qubit_active_gef` itself depends on an already-reasonable `EF_x180`, `gef_centers`, and `GEF_frequency_shift`. Bootstrapping with `active_gef` before those are calibrated (i.e. before running this node successfully at least once with `reset_type="thermal"`) can make the reset routine's own feedback loop unreliable. Calibrate the chain once with `"thermal"`, then switch.
7. **Sequence takes noticeably longer per shot than `04b_power_rabi`** → expected: for `reset_type="thermal"`, the per-shot wait is `2 × thermalization_time` (the reset's own wait plus the extra EF-specific wait added in step 4 of the Mechanism), roughly double `04b`'s per-shot overhead.

## Parameter Tuning Heuristics

1. **Fitted `opt_amp` is clearly nonphysical (far larger than a sane $\pi$-pulse amplitude, or negative in an unexpected way)** → the cosine fit likely locked onto the wrong phase/period; inspect the raw `IQ_abs` vs. amplitude-prefactor plot for more than one visible oscillation period within `[min_amp_factor, max_amp_factor]`, and narrow the range to isolate a single period if the effective Rabi frequency is high.

## Next Steps

`14_gef_readout_frequency_optimization` — uses the now-calibrated `EF_x180` (together with the anharmonicity from `12`) to populate $|f\rangle$ for its three-state readout-frequency scan. Further downstream, the two-qubit CZ calibrations `31_chevron_11_20`, `33a_cz_leakage_amplification`, and `33b_cz_leakage_amplification_palea` call `qubit.readout_state_gef()` directly (and `33b` also plays `EF_x180` directly) — both depend on this node's calibrated amplitude.

## References

**[Koc+2007]** J. Koch, T. M. Yu, J. Gambetta, A. A. Houck, D. I. Schuster, J. Majer, A. Blais, M. H. Devoret, S. M. Girvin, and R. J. Schoelkopf, "Charge-insensitive qubit design derived from the Cooper pair box," *Phys. Rev. A*, vol. 76, no. 4, p. 042319, 2007.

**[AE1987]** L. Allen and J. H. Eberly, *Optical Resonance and Two-Level Atoms*. New York: Dover Publications, 1987.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.

**[BGGW2021]** A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," *Rev. Mod. Phys.*, vol. 93, no. 2, p. 025005, 2021.
