# Hello QUA

[`00_hello_qua.py`](../../../../../calibrations/1Q_calibrations/00_hello_qua.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Plays a minimal QUA program on every active qubit to confirm basic program compilation, hardware connectivity, and execution before attempting any real calibration.

## Purpose

Before trusting any quantitative calibration result, it is worth confirming the much more basic claim that the stack even works end to end: that the QUAM configuration compiles into valid QUA, that the QOP accepts and executes the resulting job, and that the OPX is reachable and not stuck in another session. This node exists purely to answer that yes/no question — it is infrastructure, not physics, and (unlike every other node in this batch's physical siblings) has no `calibration_utils.<module>.analysis` or `.plotting` submodule at all: there is no fit, no figure, and no state update anywhere in the source.

## Mechanism

From `create_qua_program` in `calibrations/1Q_calibrations/00_hello_qua.py`:

1. Initialize the flux point (`node.machine.initialize_qpu`) for each qubit in each batch, and reset the XY drive's digital oscillator frequency to baseband (`qubit.xy.update_frequency(0)`).
2. `align()` across all elements.
3. Loop `num_shots` times; inside each shot, sweep a QUA `fixed` amplitude-scale variable `a` over **11 hardcoded points**, `np.linspace(-1, 1, 11)` (`calibrations/1Q_calibrations/00_hello_qua.py:49`) — this sweep is not configurable via any node parameter.
4. At each amplitude point, for every qubit: play the `x180` pulse on `qubit.xy` with `amplitude_scale=a`, then `qubit.wait(250 * u.ns)` (also hardcoded).
5. `align()` at the end of each shot.

Notably, **no readout is performed and no I/Q data is streamed**: the commented-out lines in the source (`qubit.z.play("const", ...)`, `I_st[0].buffer(...).save("I1")`, etc.) confirm this was deliberately stripped down — the only thing streamed and saved is the shot counter `n` (`n_st.save("n")`), used solely to drive the progress bar. There is consequently no `analyse_data` or `plot_data` run action in this node at all — only `create_qua_program`, `simulate_qua_program`, `execute_qua_program`, and `save_results`.

## Prerequisites

- A loadable QUAM state (`Quam.load()` at node instantiation) — i.e. `quam_config/populate_quam_state_*.py` must have already been run at least once for the target QOP.
- Reachable QOP/cluster, with no other session holding the OPX resources needed (see `00_close_other_qms` if a prior run left a QM open).
- No calibration prerequisites in the physics sense: this node runs before mixer/Octave calibration, time-of-flight, or any readout calibration, and does not depend on any of them being correct — it only exercises `qubit.xy`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md) — plus the one node-specific parameter below (`calibration_utils/hello_qua/parameters.py`). Note this node's `Parameters` class does subclass `QubitsExperimentNodeParameters`, so `qubits`, `multiplexed`, `use_state_discrimination`, and `reset_type` are all technically present — but none of `use_state_discrimination` or `reset_type` are referenced anywhere in `create_qua_program`, since the sequence never resets or reads out the qubit.

| Parameter | Type | Default | Unit | Description | Effect |
|---|---|---|---|---|---|
| `num_shots` | `int` | 100 | – | Number of times the 11-point amplitude sweep is repeated. | Purely affects run time and how many progress-bar updates are shown; since no data is fetched or fit, it has no effect on any measurable outcome. |

> **Declared-but-unused common parameters:** `use_state_discrimination` and `reset_type` are inherited (via `QubitsExperimentNodeParameters`) but never read in `create_qua_program` — there is no reset call and no readout, so setting either has zero effect on this node's behavior. `multiplexed` **is** honored implicitly through `qubits.batch()`, same as other nodes.

## Outputs

**None.** No measured quantities, no fit, no `node.outcomes`. The only artifact of a run is whatever `simulate_qua_program` produces if `simulate=True` (a waveform-simulation figure/report), or — on real hardware — the printed `job.execution_report()` and the shot-count progress bar. There is no `ds_raw`/`ds_fit` populated with physically meaningful content beyond the trivial `n` stream.

## State Updates

**None.** No `update_state` run action exists in the source, and no QUAM attribute is read or written.

## Troubleshooting

1. **Program fails to compile** (error raised inside `create_qua_program` or immediately on `qm.execute`) → this points to a malformed or inconsistent QUAM configuration (e.g. `x180` not defined on `qubit.xy.operations`, or `node.machine.generate_config()` producing an invalid config) rather than anything physical. Fix the QUAM state/config before trying any other node — every other node in the library depends on the same `generate_config()` path.
2. **`qm.execute` hangs, or `qm_session` eventually raises `TimeoutError`** → the OPX resources are already claimed by another open QM (a crashed previous session, or a colleague). Run `00_close_other_qms` first, then retry; if that doesn't help, check `timeout` (default 120 s) isn't simply too short for a busy/slow-to-release cluster.
3. **Execution succeeds (`job.execution_report()` shows no errors, progress bar completes) but you see no waveform on a scope/spectrum analyzer at the `xy` output** → since this node performs no readout, "success" here only means the *digital* program ran — it says nothing about the analog output actually reaching the expected physical port. Check the config's port mapping for `qubit.xy` (I/Q output channels, Octave upconverter assignment) and any output attenuation/switch state in the signal path; a silently-misrouted or over-attenuated output will still report a clean `execution_report()`.
4. **Simulated waveforms (`simulate=True`) look correct in `simulate_and_plot`'s figure, but hardware execution behaves differently** → the simulator only reflects the QUA program and the parts of `config` it models; it cannot catch real-world issues like a disconnected cable, wrong LO frequency on the Octave, or an attenuator set incorrectly. Treat a clean simulation as confirmation of program *logic* only, not of the physical signal chain — use `01a_mixer_calibration` and a spectrum analyzer for that.
5. **`job.execution_report()` reports a runtime error mid-sequence despite a clean compile** → since the sequence is trivial (play + wait, no conditionals, no readout-dependent branching), a runtime error here almost always indicates a hardware-level problem (e.g. a channel not connected, an Octave not locked) rather than a QUA logic bug — unlike in more complex nodes where a runtime error could stem from feedback logic.
6. **You changed `num_shots` expecting to see a different waveform shape** → it won't change anything visible; `num_shots` only repeats the same fixed 11-point amplitude sweep more times for a longer/more stable progress-bar run. If you need a different amplitude range or step count, those are hardcoded (`np.linspace(-1, 1, 11)`) and require editing the node source, not passing a parameter.

## Parameter Tuning Heuristics

This node produces no measured/fitted result and has no sweep/acquisition parameter whose value affects output quality — its only parameter, `num_shots`, only changes run time and progress-bar granularity (see Troubleshooting #6) — so there are no parameter-tuning heuristics for this node.

## Next Steps

Not part of the automated bring-up calibration graphs (both `80_calibration_graph_bringup_flux_tunable_transmon.py` and `90_calibration_graph_bringup_fixed_frequency_transmon.py` start at `02a_resonator_spectroscopy`). As a manual pre-flight step, the natural next node is `01a_mixer_calibration` — now that basic execution and connectivity are confirmed, calibrate the Octave mixers feeding `qubit.resonator` and `qubit.xy` before any frequency-domain measurement.

## References

None — this node is pure infrastructure with no physics content.
