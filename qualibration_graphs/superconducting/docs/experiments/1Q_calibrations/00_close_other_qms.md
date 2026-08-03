# Close Other Quantum Machines

[`00_close_other_qms.py`](../../../../../calibrations/1Q_calibrations/00_close_other_qms.py) · **Targets:** qubits · **Category:** 1Q_calibrations

Frees up OPX resources by force-closing every open Quantum Machine on the cluster before a calibration session starts.

## Purpose

This is **not a physics experiment** — it is a housekeeping utility. An OPX's resources (analog/digital I/O, elements) can only be claimed by one open Quantum Machine (QM) at a time. If a previous script crashed, a Jupyter kernel was left running, or another user's session didn't clean up after itself, its QM stays open and holds those resources, so any subsequent `qmm.open_qm(...)` call (which every other node performs, via `qm_session`) blocks or times out. This node's only job is to release such stuck resources by calling `qmm.close_all_qms()` before running the actual calibration graph.

## Mechanism

The entire node body is a single `@node.run_action`:

1. Connect to the Quantum Machines Manager: `qmm = node.machine.connect()`.
2. Close every currently open quantum machine on that cluster: `qmm.close_all_qms()`.

There is no QUA program, no sweep, no measurement, and no `save_results` action calling `node.save()` — the source file has no such call, so this node produces no `data.json`/state snapshot the way physics nodes do.

## Prerequisites

- None beyond a reachable QOP/cluster — this node has no dependency on prior calibration state. It is typically the very first thing run in a session, before even `00_hello_qua`.

## Parameters

Inherits the common parameter set — see [Common node parameters](_common_parameters.md). There are **no node-specific parameters**: the node is instantiated with a bare `NodeParameters()` (`calibrations/1Q_calibrations/00_close_other_qms.py:10`), not a custom `Parameters` subclass, so only the two common groups (`CommonNodeParameters`) apply — and of those, only `timeout` and `simulate`/`simulation_duration_ns`/`use_waveform_report` are even defined; none are read anywhere in this node's single run action, and `qubits`/`multiplexed`/`reset_type`/`use_state_discrimination` from `QubitsExperimentNodeParameters` are not part of this node's `Parameters` at all since it doesn't subclass that group.

## Outputs

**None.** No measurement is performed, no dataset is produced, and no `save_results` action exists in the source. Success is simply "the function returned without raising."

## State Updates

**None.** No QUAM attribute is read or written. `close_all_qms()` operates purely on the QOP/cluster's live session state (which QMs are currently open), not on the QUAM machine configuration (`state.json`).

## Troubleshooting

See also: [General troubleshooting](_general_troubleshooting.md#general-troubleshooting) for issues common to most nodes in this library. Below are issues specific to this node.

1. **A calibration node hangs at "Opening QM" / eventually raises `TimeoutError` from `qm_session`** → another QM (yours from a crashed kernel, or a colleague's) is still holding the OPX resources, so `qmm.open_qm(...)` keeps retrying inside `qm_session`'s poll loop until `timeout` (default 120 s, `CommonNodeParameters.timeout`) elapses. Run `00_close_other_qms` first, then retry the original node.
2. **You're about to run this on a shared cluster and worry about impact on colleagues** → `qmm.close_all_qms()` closes **every** open QM on the cluster indiscriminately, not just ones owned by your session. If someone else has a job running, this will interrupt it. Coordinate before running this on shared hardware; on a dedicated single-user OPX this is a non-issue.
3. **`qmm.connect()` itself raises a connection error (e.g. cannot reach cluster host/port)** → this is a network/QOP-availability problem, not something `close_all_qms()` can fix. Verify the network config in the QUAM state (host, cluster name, port) and that the QOP software is up before re-running.
4. **Running this node doesn't actually unblock a subsequent node** → the stuck resource may not be a QM at all but a still-*running job* inside a QM that legitimately needs to finish (e.g. a long calibration graph another process is mid-way through). Closing the QM while a job is running will abort that job — check `qm.get_running_job()`/cluster job status before assuming a hang is due to a leftover session rather than an active one.
5. **You expect this to reset instrument state (DC offsets, Octave calibration, etc.) and it doesn't** → it doesn't touch any of that. Closing a QM releases resource *locks*, it does not revert any hardware settings (e.g. `keep_dc_offsets_when_closing` behavior on the QM side) or QUAM state. Don't use this node as a substitute for re-running an actual calibration.
6. **This node appears to do nothing (no error, no visible effect) when nothing was actually stuck** → this is expected and harmless: `close_all_qms()` on an already-clean cluster is a no-op. It is safe to run defensively at the start of every session regardless of whether anything is actually stuck.

## Parameter Tuning Heuristics

See also: [General parameter tuning heuristics](_general_troubleshooting.md#general-parameter-tuning-heuristics) for guidance common to most nodes in this library. Below is guidance specific to this node.

This node has no sweep/acquisition parameters and nothing to tune — it is a single unconditional `qmm.close_all_qms()` call with no node-specific `Parameters` — so there are no parameter-tuning heuristics for this node.

## Next Steps

Not part of the automated bring-up calibration graphs (`80_calibration_graph_bringup_flux_tunable_transmon.py` / `90_calibration_graph_bringup_fixed_frequency_transmon.py`, both of which start at `02a_resonator_spectroscopy`) — this and `00_hello_qua` are manual pre-flight steps run standalone before any graph. Typical next step: `00_hello_qua`, to confirm the now-freed OPX actually accepts and executes a QUA program before moving on to real calibrations.

## References

None — this node is pure infrastructure with no physics content.
