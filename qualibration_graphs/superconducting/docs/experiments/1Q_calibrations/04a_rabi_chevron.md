# Rabi Chevron

[`04a_rabi_chevron.py`](../../../../../calibrations/1Q_calibrations/04a_rabi_chevron.py) · **Targets:** qubits · **Category:** 1Q_calibrations

2D sweep of drive duration vs. drive-frequency detuning around the qubit's rough `x180` pulse, producing the classic Rabi "chevron" pattern — read manually to pick a pulse duration, with no automatic state update.

## Purpose

Driving a qubit off resonance still produces Rabi oscillations, just faster and with reduced contrast: the generalized Rabi frequency is $\Omega_R=\sqrt{\Omega_0^2+\Delta^2}$, where $\Omega_0$ is the on-resonance Rabi rate (set by the drive amplitude) and $\Delta$ is the detuning **[Kra+2019]**, **[AE1987]**. Sweeping both the drive duration and the detuning at a fixed amplitude therefore traces out a fringe pattern whose fringes narrow away from $\Delta=0$ and are widest exactly on resonance — the "chevron." The row of the map at zero detuning shows the plain on-resonance Rabi oscillation, whose first population inversion gives the pulse duration that implements a $\pi$ rotation *at the amplitude currently configured for `x180`*.

This node exists to make that duration visible on a 2D plot for a human to read off, before `04b_power_rabi` does the precise, automated amplitude calibration at whatever duration is chosen. Because only duration and detuning are swept here — amplitude is fixed to whatever is already in `qubit.xy.operations["x180"].amplitude` — a duration read off very early in the sweep (close to the 16 ns minimum) should be treated with some caution: a very short pulse has spectral content spread over roughly $1/\text{duration}$, and for a typical transmon anharmonicity of $\alpha/2\pi\approx 200\,\text{MHz}$, a pulse whose bandwidth starts to approach that scale risks off-resonantly driving the $|1\rangle\leftrightarrow|2\rangle$ transition — i.e. leakage-driven distortion of the fringe pattern, not just ordinary dephasing/decoherence broadening **[GRTW2021]**.

There is no example plot shipped for this node in the documentation image set.

## Mechanism

> **This node never writes to QUAM state automatically — it is a read-the-plot, then act manually node.** The description block in the source is explicit about this: *"State update: Manually set the x180 pulse duration `qubit.xy.operation["x180"].length`."* This is the same pattern the pilot doc for `03b` flags for stale docstrings, but here the "manual update" claim is verified accurate: `update_state`'s loop body is empty even for a qubit whose outcome is `"successful"` (see the callout on outcomes below) — there is no write-back code path at all, successful or not.

1. Before the sweep starts, every targeted qubit's `x180` pulse length is force-set to 16 ns via `tracked_updates(..., auto_revert=False)` — the minimum QUA pulse length. This is *not* reverted until the very end of `update_state`, i.e. it stays forced for the entire measurement.
2. For each (detuning, duration) grid point, repeated `num_shots` times for averaging:
   1. `qubit.xy.update_frequency(intermediate_frequency)` — reset the drive frequency back to the qubit's plain (undetuned) intermediate frequency *before* resetting, so that active/`active_gef` reset's readout-based discrimination is evaluated on-resonance rather than at whatever detuning the previous grid point left behind.
   2. `qubit.reset(reset_type, simulate)` — this node **does** honor the common `reset_type` parameter (unlike `03a`/`03b`).
   3. `qubit.xy.update_frequency(df + intermediate_frequency)` — re-apply the swept detuning.
   4. `qubit.xy.play("x180", duration=t)` — the operation name is hardcoded to `"x180"` (not configurable); `t` dynamically overrides the pulse duration in QUA, stretching/truncating the (now 16-ns-length) pulse to the swept value. The amplitude used is whatever is currently configured on `x180` — this node never scales or sweeps it.
   5. Measure (`readout_state` if `use_state_discrimination`, else raw `I`/`Q`).
3. At the very start of `update_state`, `tracked_qubit.revert_changes()` restores each qubit's original `x180` length — the tracked override is undone regardless of fit outcome. This revert is the *only* state mutation this node ever performs.

Analysis (`calibration_utils/rabi_chevron/analysis.py`):

- `process_raw_dataset` only converts I/Q to volts and computes the absolute frequency axis — no peak-finding, no oscillation fit.
- `fit_raw_data` / `_extract_relevant_fit_parameters` populate a `FitParameters` dataclass with a single field, `success`, which is **hardcoded to `False` for every qubit, unconditionally** — the data is never inspected. `log_fitted_results` is likewise a no-op (`pass`).
- Consequently `node.outcomes[qubit_name]` is always `"failed"`, for every qubit, on every run — this is expected behavior for this node, not a bug to chase.

> **Source docstring is stale.** `calibration_utils/rabi_chevron/parameters.py`'s `NodeSpecificParameters` class docstring describes an entirely different parameter set — it documents `num_shots`, `frequency_span_in_mhz`, `frequency_step_in_mhz`, `operation`, `operation_amplitude_factor`, `operation_len_in_ns`, and `target_peak_width`, and is explicitly labeled *"Parameters for configuring a qubit spectroscopy experiment"*. This is a direct copy-paste of `qubit_spectroscopy/parameters.py`'s docstring. The class's real fields are `num_shots`, `min_wait_time_in_ns`, `max_wait_time_in_ns`, `time_step_in_ns`, `frequency_step_in_mhz`, and `frequency_span_in_mhz` — there is no `operation`, `operation_amplitude_factor`, `operation_len_in_ns`, or `target_peak_width` on this node at all; the played pulse is unconditionally `"x180"` at whatever amplitude is already configured.

## Prerequisites

- Mixer/Octave calibration (`01a_mixer_calibration` or `01b_time_of_flight_mw_fem`).
- Qubit frequency calibrated (`03a_qubit_spectroscopy` and/or `03b_qubit_spectroscopy_vs_flux`) — this node sweeps detuning *around* the qubit's currently configured intermediate frequency, it does not search blindly.
- Flux operating point specified if relevant (`qubit.z.flux_point`).
- A rough `x180` amplitude already present in state (from a datasheet estimate, a previous device's value, or a prior rough `04b_power_rabi` pass) — this node never touches amplitude, so the chevron it draws is only meaningful relative to whatever amplitude is already configured.
- Graph topology: in both bring-up graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`), this node runs directly after `03a_qubit_spectroscopy` (flux-tunable bring-up additionally runs `03b_qubit_spectroscopy_vs_flux` in between).

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per (detuning, duration) grid point. | More shots lower I/Q noise per point; linear cost in run time, multiplied across the whole 2D grid. |
| `min_wait_time_in_ns` | `int` | 16 | ns | Lower bound of the swept `x180` duration. | Also the QUA minimum pulse length; matches the length the base pulse is force-set to for the sweep. |
| `max_wait_time_in_ns` | `int` | 250 | ns | Upper (exclusive) bound of the swept duration. | Widen if the on-resonance Rabi period at the current `x180` amplitude is longer than this — the chevron simply won't show a full oscillation otherwise. |
| `time_step_in_ns` | `int` | 4 | ns | Duration-sweep step. | The QUA loop converts this axis via `pulse_durations // 4` (integer division into clock cycles) — keep this a multiple of 4 ns, or the effective step gets silently rounded. |
| `frequency_span_in_mhz` | `float` | 100 | MHz | Full width of the detuning sweep, centered on the qubit's current intermediate frequency. | Must be wide enough that the chevron's vertex (zero-detuning column) actually falls inside the window — see Parameter Tuning Heuristics. |
| `frequency_step_in_mhz` | `float` | 4 | MHz | Detuning-sweep step. | Finer steps resolve the chevron's shape better at proportional cost in run time. |

> **No `operation` or amplitude parameter exists on this node**, despite what the (stale) source docstring documents — see the callout in Mechanism. The pulse played is always `"x180"`, at whatever amplitude is currently in QUAM state.

## Outputs

**Measured:** `I`/`Q` (volts, or discriminated `state` if `use_state_discrimination`), at every (detuning, duration) grid point. No amplitude/phase or rotated-axis quantities are computed.

**Fitted quantities: none.** The only field in `FitParameters` is `success`, and it is unconditionally `False` — no frequency, duration, or width is ever extracted programmatically.

**Success criterion:** never satisfied, by construction — `node.outcomes[qubit_name]` is `"failed"` for every qubit on every run. Do not gate downstream automation (graph orchestration, retries) on this node's outcome; it carries no information about whether the measurement itself succeeded or failed.

The plot (`plot_raw_data_with_fit`, titled "Rabi chevron") is the actual deliverable: a 2D colormap of `I` (or `state`) vs. pulse duration and detuning/absolute frequency, with no fitted overlay.

## State Updates

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.xy.operations["x180"].length` | *(reverted to its pre-run value)* | replace (temporary, then reverted) | Always, at the start of `update_state` — regardless of fit outcome. This is the node's only automated state interaction, and it is a no-net-effect revert, not a calibration write. |
| `qubit.xy.operations["x180"].length` | *(a value the operator reads off the chevron plot)* | **manual, human-applied** | After visually identifying the zero-detuning row's first population-inversion duration in the plot. |

There is no `update_pulses_amplitude`-style flag, no automatic frequency write, and no automatic duration write anywhere in this node's source.

## Troubleshooting

1. **Every qubit reports `"failed"`, every time** → expected, not a bug: `FitParameters.success` is hardcoded to `False` in `calibration_utils/rabi_chevron/analysis.py` regardless of the data collected. Don't chase this as a calibration failure or wire a graph edge that expects `"successful"` from this node.
2. **Chevron is visible but its vertex (widest fringe, i.e. resonance) isn't at zero detuning** → the qubit's currently configured intermediate frequency (from `03a`/`03b`) is off from the true resonance by that offset. Re-run `03a_qubit_spectroscopy` (or `03b_qubit_spectroscopy_vs_flux` for flux-tunable qubits) to refresh `f_01` before trusting a duration read off this chevron.
3. **The duration picked off the zero-detuning row doesn't give a clean $\pi$ pulse when later run through `04b_power_rabi`** → expected if the seed `x180` amplitude used for this chevron was far from correct: this node fixes amplitude and only locates a *duration*, so the visible "period" is specific to that stale amplitude, not a calibrated target. Always run `04b_power_rabi` with the picked duration afterward to solve for the correct amplitude — don't treat the chevron's raw duration as a finished calibration by itself.
4. **Picked duration is very short (near the 16 ns sweep floor) and the fringe near there looks distorted/asymmetric rather than a clean chevron** → possible leakage into $|2\rangle$: a short pulse's spectral width approaches the typical ~200 MHz anharmonicity, and off-resonant driving of the $1\leftrightarrow2$ transition can distort the fringe pattern near very short durations **[GRTW2021]**. Prefer reading off a duration from a later, cleaner fringe rather than the very first one if this is visible.
5. **Buffering/shape errors, or the swept duration axis looks coarser than expected** → check that `time_step_in_ns` (and `min_wait_time_in_ns`/`max_wait_time_in_ns`) are multiples of 4 ns; the QUA program divides the duration array by 4 for its clock-cycle loop variable, and non-multiples get silently truncated by integer division rather than raising an error.
6. **Active reset (`reset_type="active"`/`"active_gef"`) seems to behave inconsistently across detuning points** → verify (if you've modified this node) that the drive frequency is reset to the bare `intermediate_frequency` *before* the `qubit.reset(...)` call and only re-detuned afterward, as in the shipped source — active reset's readout-based discrimination assumes on-resonance conditions, and losing that reordering will silently degrade reset fidelity at nonzero detuning without raising an error.
7. **Trying to set `operation`, `operation_amplitude_factor`, or `target_peak_width` on this node** → these don't exist here despite appearing in the (stale) class docstring; they're artifacts of a copy-paste from `qubit_spectroscopy/parameters.py` and will be rejected or simply have no effect.

## Parameter Tuning Heuristics

1. **No visible chevron pattern — flat, featureless map** → most likely the `x180` amplitude currently in state is too small to produce a visible Rabi oscillation within `min_wait_time_in_ns`–`max_wait_time_in_ns` (16–250 ns default). Set a reasonable rough amplitude on `qubit.xy.operations["x180"]` first (datasheet estimate, prior device value, or a quick manual test), then re-run.
2. **Chevron vertex looks like it's cut off at the edge of the detuning window** → `frequency_span_in_mhz` (default 100 MHz) is too narrow to contain the fringe's full width at the current drive amplitude; widen it and re-run rather than reading off a partial fringe.

## Next Steps

`04b_power_rabi` — in both bring-up graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py`, `90_calibration_graph_bringup_fixed_frequency_transmon.py`), this node feeds directly into `04b_power_rabi`, which calibrates the amplitude at the duration manually chosen from this node's chevron plot. The retuning graphs (`81`, `91`) skip this node entirely — they assume pulse duration is already set and start straight from `04b_power_rabi`'s error-amplification stages.

## References

**[Kra+2019]** P. Krantz, M. Kjaergaard, F. Yan, T. P. Orlando, S. Gustavsson, and W. D. Oliver, "A quantum engineer's guide to superconducting qubits," *Appl. Phys. Rev.*, vol. 6, no. 2, p. 021318, 2019.

**[AE1987]** L. Allen and J. H. Eberly, *Optical Resonance and Two-Level Atoms*. New York: Dover Publications, 1987.

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.
