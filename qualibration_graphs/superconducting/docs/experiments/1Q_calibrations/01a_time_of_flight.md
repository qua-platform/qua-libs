# Time of Flight (OPX+ / LF-FEM)

[`01a_time_of_flight.py`](../../../../../calibrations/1Q_calibrations/01a_time_of_flight.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Sends a readout pulse, captures the raw ADC trace, and fits the pulse arrival time and per-controller analog-input DC offsets for OPX+/LF-FEM readout hardware.

![Raw ADC trace with fitted time-of-flight marker](images/time_of_flight.png){ .calibration-result }

## Purpose

Between the moment the OPX issues a readout pulse and the moment the reflected/transmitted signal actually arrives back at its digitizer, there is a fixed propagation and processing delay — the **time of flight (TOF)**. The OPX's acquisition window must be shifted by exactly this delay (`qubit.resonator.time_of_flight` in the QUAM config), or every subsequent demodulated I/Q measurement integrates over the wrong window and is corrupted. This is the calibration-node analog of the propagation-delay term $\tau_v$ that appears in a resonator's fitted $S_{21}$ response (Eq. 51 of **[GRTW2021]**, p. 21) — cable/signal-path delay has to be accounted for before resonator parameters can be trusted; here it is measured directly from the raw ADC trace rather than extracted as a fit parameter of a frequency sweep. Separately, small DC offsets on the OPX's analog inputs (from minor impedance mismatches in the signal path) bias every I/Q measurement by a constant unless corrected — this node measures and corrects those too. Beyond that, the literature used elsewhere in this repository only thinly covers time-of-flight itself; **[GRTW2021]**'s Sec. IV.B (p. 19) gives generic ADC hardware baseline specs (sampling rate $\geq 0.5$–$1\,\text{GSample/s}$, 16-bit vertical resolution) but does not treat TOF calibration as its own topic, so most of the grounding below comes directly from this node's own fit logic rather than from citable literature.

## Mechanism

`create_qua_program` (`calibrations/1Q_calibrations/01a_time_of_flight.py`):

1. For each qubit's `resonator`, wrap it in `tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True)` and temporarily set `resonator.time_of_flight`, `resonator.operations["readout"].length`, and `resonator.operations["readout"].amplitude` to this run's node parameters. Because `auto_revert=False`, these overrides are applied immediately to the live QUAM object used to build the QUA program/config — they are **not** rolled back automatically; that only happens explicitly in `update_state` (see the callout below). Because `dont_assign_to_none=True`, a `None`-valued parameter (the default for `time_of_flight_in_ns`) is simply *not* assigned, leaving whatever TOF is already configured in QUAM state untouched for this run.
2. For each batch of (possibly multiplexed) qubits, loop `num_shots` times; for each qubit, `reset_if_phase(qubit.resonator.name)` (needed so averaging the raw cosine ADC signal across shots doesn't wash out the trace), then `qubit.resonator.measure("readout", stream=adc_st[i])` to fire the readout pulse and capture the **raw** ADC trace (not demodulated I/Q), then wait `node.machine.depletion_time` for the resonator to ring down before the next shot.
3. Stream both the shot-averaged trace (`adcI{n}`/`adcQ{n}`) and the single-last-shot trace (`adc_single_runI{n}`/`adc_single_runQ{n}`) per qubit — the averaged trace is used for the offset fit, the single-shot trace for visualizing an individual (unaveraged) pulse.

Analysis (`calibration_utils/time_of_flight/analysis.py`):

1. `process_raw_dataset`: convert raw ADC counts to volts (`-adc / 2**12`) and compute `IQ_abs = sqrt(I² + Q²)`.
2. `fit_raw_data`: smooth `IQ_abs` with a Savitzky–Golay filter (window 11, order 3) to get `filtered_adc`; compute a `threshold` as the midpoint between the mean level in the tail (samples `[100:]`) and the mean level in the head minus its last 100 samples (`[:-100]`) — an estimate of the midpoint between "before pulse" and "after pulse" trace levels; find the first sample index where `filtered_adc` crosses that threshold, and round it to the nearest multiple of 4 ns (`delay`, i.e. `tof_to_add` — a **correction offset**, not an absolute TOF).
3. Read each qubit's controller from `resonator.opx_input_I.controller_id`, after checking (`_check_resonator_inputs`) that a resonator's I and Q inputs share the same controller — raises `RuntimeError` if not.
4. Compute each qubit's raw baseline offset (`offsets_I`/`offsets_Q`, the trace's mean level in volts over the full readout window), then **average that offset across all qubits sharing the same controller** (`mean_offset_I`/`mean_offset_Q` per `con`) and assign the shared per-controller mean back to every qubit on that controller as `offsets_I_mean`/`offsets_Q_mean` — this reflects that the analog input DC offset is a per-controller hardware setting, not a per-qubit one.
5. Success requires that not all of `delay`, `offsets_I_mean`, `offsets_Q_mean` are NaN, **and** `|offsets_I_mean| < 0.5` and `|offsets_Q_mean| < 0.5` (the ADC's full-scale input range is $\pm 0.5\,\text{V}$ — an offset at or beyond that means the signal is already at/past the rail).

## Prerequisites

- QUAM initialized (`quam_config/populate_quam_state_*.py`), per the node's own docstring.
- OPX+/LF-FEM hardware (this node's sibling, `01b_time_of_flight_mw_fem`, is the MW-FEM equivalent — see that node's doc for the hardware-path differences).
- Mixer/Octave calibration typically already run (`01a_mixer_calibration`), so the readout pulse's amplitude and frequency reach the resonator cleanly, though this node's own fit doesn't depend on frequency-domain correctness — it only needs a visible pulse edge in time.
- Each resonator's `opx_input_I` and `opx_input_Q` must be wired to the **same controller** — this is checked and enforced (`RuntimeError` on mismatch), not merely assumed.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the node-specific parameters below (`calibration_utils/time_of_flight/parameters.py`).

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Averages per qubit for the averaged trace. | More shots reduce noise on the shot-averaged trace used for offset fitting; the single-run trace (used only for visualization) is unaffected. |
| `time_of_flight_in_ns` | `Optional[int]` | `None` | ns | Temporary TOF override applied before this run. | `None` (the default) means **no override** — thanks to `dont_assign_to_none=True`, the currently configured `qubit.resonator.time_of_flight` is used as-is for this measurement, and the final state update *increments* it by the fitted correction. If explicitly set, this value is used for the run **and** becomes the new baseline that the fitted correction is added to at update time (see State Updates). |
| `readout_amplitude_in_v` | `Optional[float]` | 0.03 | V | Temporary override of `resonator.operations["readout"].amplitude` for this run. | Higher amplitude improves SNR on the pulse edge (sharper threshold crossing) but risks saturating the ADC (`±0.5 V` full scale) or distorting the trace; too low can hide the pulse in noise, making `delay` and the offset fit unreliable. |
| `readout_length_in_ns` | `Optional[int]` | 1000 | ns | Temporary override of `resonator.operations["readout"].length`. | Also sets the extent of the acquisition window (`readout_time` sweep axis runs `0` to this value) — must be long enough to include both a clean pre-pulse baseline and the settled pulse plateau for the threshold/offset estimates to be meaningful. |

> **Docstrings vs. actual defaults, three separate mismatches:** `time_of_flight_in_ns`'s docstring reads "Default is 28 ns," but the actual field default is `None` (which means "leave whatever is currently configured," not 28 ns specifically — 28 ns is only QOP's own factory-default TOF, not this parameter's default). `readout_amplitude_in_v`'s docstring reads "Default is 0.1 V," but the actual default is `0.03`. `readout_length_in_ns`'s docstring reads "Default is the pulse predefined pulse length," implying `None`/inherited, but the actual default is a fixed `1000`, which **always** overrides whatever length the `readout` pulse was otherwise configured with. Trust the field defaults in `calibration_utils/time_of_flight/parameters.py`, not the docstrings, for all three.

## Outputs

**Measured:** per-qubit averaged and single-shot raw ADC traces (`adcI`/`adcQ`, `adc_single_runI`/`adc_single_runQ`, in volts) over the `readout_time` axis, plus `IQ_abs`.

| Fitted quantity | Unit | Written to state? | Description |
|---|---|---|---|
| `tof_to_add` | ns | ✅ (as a correction, see State Updates) | Threshold-crossing time of the pulse edge in the acquired trace, rounded to the nearest 4 ns. |
| `offset_I_to_add` | V | ✅ (conditionally, see State Updates) | Per-controller mean DC offset on the `I` analog input, shared across all qubits on that controller. |
| `offset_Q_to_add` | V | ✅ (conditionally, see State Updates) | Per-controller mean DC offset on the `Q` analog input, shared across all qubits on that controller. |

**Success criterion:** not all of `delay`/`offsets_I_mean`/`offsets_Q_mean` are NaN, **and** both `|offsets_I_mean| < 0.5 V` and `|offsets_Q_mean| < 0.5 V`. Checked per-qubit in `fit_raw_data`.

## State Updates

`update_state` first **reverts** every temporary override applied in `create_qua_program` (`tracked_resonator.revert_changes()` for each entry in `node.namespace["tracked_resonators"]`) — so the TOF/length/amplitude values actually used *during* the measurement are explicitly not what ends up persisted; only the values written below persist.

| Attribute | New value | Mode | Condition |
|---|---|---|---|
| `qubit.resonator.time_of_flight` | `time_of_flight_in_ns + tof_to_add` | **replace** | outcome successful **and** `time_of_flight_in_ns` was explicitly set (not `None`) |
| `qubit.resonator.time_of_flight` | previous value `+ tof_to_add` | **increment (`+=`)** | outcome successful **and** `time_of_flight_in_ns` was `None` (the default) |
| `qubit.resonator.opx_input_I.offset` | previous value `+ offset_I_to_add` (or set directly if previously `None`) | increment/replace | outcome successful **and** this qubit's controller is still in `controllers_to_update` (see below) |
| `qubit.resonator.opx_input_Q.offset` | previous value `+ offset_Q_to_add` (or set directly if previously `None`) | increment/replace | same as `opx_input_I` above |

`controllers_to_update` starts as the list of **unique** controller IDs present in the fitted dataset (`np.unique(ds_fit.con.values)`). The qubit loop iterates in `node.namespace["qubits"]` order; the **first** qubit encountered for a given controller has its `opx_input_I`/`opx_input_Q` offsets updated and that controller ID is then **removed** from `controllers_to_update` — every subsequent qubit sharing that same controller is skipped for the offset update (its `tof_to_add` is still applied individually, since TOF updates are not controller-scoped). This is intentional deduplication, not a bug: since the offset fit already averages to one shared value per controller, applying it more than once per controller would be redundant — but it does mean *which* qubit's `opx_input_I`/`opx_input_Q` object is the one QUAM records the update under depends on iteration order, not on which qubit you "expected."

## Troubleshooting

See also: [General troubleshooting](_general_troubleshooting.md#general-troubleshooting) for issues common to most nodes in this library. Below are issues specific to this node.

1. **A parameter you passed (e.g. `readout_amplitude_in_v`) doesn't seem to affect the persisted state afterward** → this is expected for `time_of_flight_in_ns`/`readout_length_in_ns`/`readout_amplitude_in_v`: they only control the *measurement*, via `tracked_updates(..., auto_revert=False)`, and are explicitly reverted in `update_state` before any final value is written. Only `qubit.resonator.time_of_flight` (and conditionally the input offsets) persist — the length/amplitude used during acquisition are always reverted, never saved.
2. **An input offset you expected to be corrected wasn't** → check whether another qubit sharing the same `opx_input_I.controller_id` was processed earlier in `node.namespace["qubits"]` order in `update_state`; that qubit "claims" the controller-wide offset update and removes the controller from `controllers_to_update`, so every other qubit on the same controller is skipped for the offset write (though not for the TOF write, which is per-qubit). This is normal when qubits genuinely share a controller; if you didn't expect two qubits' resonators to share a controller at all, check the QUAM wiring — that assumption may be wrong for your hardware.
3. **Fit fails (`success=False`) with `offsets_I_mean`/`offsets_Q_mean` near or beyond $\pm 0.5\,\text{V}$** → the ADC is saturating or close to its $\pm 0.5\,\text{V}$ full-scale input range. Check for an upstream gain/attenuation misconfiguration in the signal chain feeding this controller's analog input.
4. **`RuntimeError: <resonator> doesn't have its two outputs connected to the same controller`** (`_check_resonator_inputs`) → this node's controller-scoped offset averaging assumes I/Q share one controller; if your QUAM wiring genuinely splits a resonator's I and Q inputs across two controllers, this node cannot calibrate that resonator's offsets as written — fix the wiring/QUAM config, or treat this specific qubit's offset calibration as unsupported.
5. **You re-ran with a different `readout_length_in_ns` than a previous run and got a different `tof_to_add`, even though nothing physical changed** → the threshold in `fit_raw_data` is computed from the head/tail means of *this run's* trace window, which shifts if the window length changes what fraction of the trace is pre- vs. post-pulse. For repeatable comparisons across runs, keep `readout_length_in_ns` fixed.
6. **Downstream nodes (e.g. `02a_resonator_spectroscopy`) show a systematically wrong or noisy readout right after this node reports success** → double-check that the *reverted* readout pulse length/amplitude (i.e. what's actually configured in QUAM state after this node's `update_state` runs) match what you intend for real measurements — this node's own overrides during the run (often a long, low `readout_length_in_ns`/`readout_amplitude_in_v` tuned for a clean TOF fit) are not what persists, but if some other override elsewhere in your workflow *did* persist unexpectedly, that's the first place to check.

## Parameter Tuning Heuristics

See also: [General parameter tuning heuristics](_general_troubleshooting.md#general-parameter-tuning-heuristics) for guidance common to most nodes in this library. Below is guidance specific to this node.

1. **Fit fails (`success=False`) with `offsets_I_mean`/`offsets_Q_mean` near or beyond $\pm 0.5\,\text{V}$** → the ADC is saturating or close to its $\pm 0.5\,\text{V}$ full-scale input range (see Troubleshooting #3 for a non-parameter cause). Lower `readout_amplitude_in_v` to bring the trace back within range.
2. **`tof_to_add` comes out as 0, or clearly wrong (e.g. picks up an artifact rather than the true pulse edge)** → the Savitzky–Golay-filtered threshold crossing depends on a clean baseline-to-plateau step; if `readout_length_in_ns` is too short to show a stable pre-pulse baseline and a settled post-edge plateau, or if `readout_amplitude_in_v` is too low for the edge to rise clearly above noise, the threshold (the midpoint between head and tail means) can sit in the wrong place. Increase `readout_length_in_ns` and/or `readout_amplitude_in_v` and re-run.
3. **Re-running this node repeatedly without ever passing `time_of_flight_in_ns` causes `time_of_flight` to keep drifting further from a stable value** → with the default `None`, every successful run *increments* `time_of_flight` by `tof_to_add` rather than replacing it. If the pulse edge detection has any residual bias (e.g. from the 4 ns rounding), repeated blind re-runs can accumulate drift. Periodically pass an explicit `time_of_flight_in_ns` (the current known-good value) to force a **replace** instead of an increment, resetting any accumulated drift.

## Next Steps

Not part of the automated bring-up calibration graphs (both `80_calibration_graph_bringup_flux_tunable_transmon.py` and `90_calibration_graph_bringup_fixed_frequency_transmon.py` start at `02a_resonator_spectroscopy`) — this is a manual pre-flight step, run once TOF and analog-input offsets are believed correct. Next: `02a_resonator_spectroscopy`, which is the first node in both bring-up graphs and depends on a correctly time-aligned, offset-free readout.

## References

**[GRTW2021]** Y. Y. Gao, M. A. Rol, S. Touzard, and C. Wang, "Practical guide for building superconducting quantum devices," *PRX Quantum*, vol. 2, p. 040202, 2021.
