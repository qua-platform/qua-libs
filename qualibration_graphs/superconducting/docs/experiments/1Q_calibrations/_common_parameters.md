# Common node parameters

Every calibration node in `calibrations/1Q_calibrations/` inherits these two parameter groups on top of its own node-specific ones. They are documented once here; each node's own doc links back to this page instead of repeating them.

**Important caveat:** these fields are *declared* on every node, but not every node's QUA sequence necessarily *honors* all of them — some nodes hardcode behavior that ignores one of these knobs. Always check the node's own "Parameters" section for a note before assuming a common parameter does what its name implies for that specific node.

## `CommonNodeParameters`

Source: `qualibration_libs/parameters/common.py`

| Name | Type | Default | Description |
|---|---|---|---|
| `simulate` | `bool` | `False` | Simulate the waveforms on the OPX instead of executing on real hardware. |
| `simulation_duration_ns` | `int` | `50_000` | Duration (ns) over which the simulation collects samples. Only relevant if `simulate=True`. |
| `use_waveform_report` | `bool` | `True` | Whether to generate the interactive waveform report during simulation. Only relevant if `simulate=True`. |
| `timeout` | `int` | `120` | Seconds to wait for OPX resources to become available before giving up. |
| `load_data_id` | `Optional[int]` | `None` | If set, re-analyzes a previous run's dataset (by QUAlibrate run index) instead of taking new hardware data — skips `create_qua_program`/`execute_qua_program` entirely. |

## `QubitsExperimentNodeParameters` (single-qubit nodes)

Source: `qualibration_libs/parameters/experiment.py`. Qubit-pair-flavored nodes (e.g. `02d`, `03c`, `09b`, `16b`, `19`) use the analogous `QubitPairExperimentNodeParameters` (`qubit_pairs` instead of `qubits`) — not covered here.

| Name | Type | Default | Description |
|---|---|---|---|
| `multiplexed` | `bool` | `False` | `True`: play control pulses, readout pulses, and active/thermal reset simultaneously for all targeted qubits. `False`: run the sequence fully sequentially, one qubit at a time. |
| `use_state_discrimination` | `bool` | `False` | `True`: return the on-the-fly discriminated qubit `state` (0/1) instead of raw demodulated `I`/`Q`. Requires IQ-blob calibration (`07_iq_blobs`) to already be done. |
| `reset_type` | `Literal["thermal", "active", "active_gef"]` | `"thermal"` | Qubit initialization method between shots. `"thermal"`: just wait; `"active"`/`"active_gef"`: measure-and-feedback reset, requires the corresponding reset method to be calibrated first. |
| `qubits` | `Optional[List[str]]` | `None` | Names of qubits to run on. `None` → all of `machine.active_qubits`. |

## How these interact with node-specific behavior

- `multiplexed` changes crosstalk/timing characteristics of the measurement — a node that's clean when run one-qubit-at-a-time can show artifacts when multiplexed, and vice versa.
- `reset_type="active"`/`"active_gef"` is only as good as the underlying reset calibration; using it before that calibration exists can silently degrade fidelity rather than raise an error.
- `load_data_id` bypasses hardware entirely — useful for iterating on analysis/fit code without re-measuring, but any "current" QUAM state referenced during analysis (e.g. `qubit.xy.RF_frequency` used to convert a swept detuning into an absolute frequency) reflects the state *now*, not the state at the time of the original run.
