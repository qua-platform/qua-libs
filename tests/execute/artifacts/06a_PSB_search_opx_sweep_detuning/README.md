# 06a_PSB_search_opx_sweep_detuning

## Description


PAULI SPIN BLOCKADE SEARCH — Sweep detuning (OPX)

This node searches for the Pauli Spin Blockade (PSB) region by sweeping the
inter-dot detuning and measuring the sensor response during a PSB readout window.
Each sweep point plays a voltage sequence (prepare → ramp → measure) using OPX 
fast-line channels, and acquires per-shot I/Q data.

Prerequisites
-------------
- QUAM configured and loaded (``quam_config/populate_quam_state_*.py``).
- Sensor-dot readout calibrated (resonator calibration nodes completed).
- Detuning axis and voltage points (empty / initialize / measure) defined on the dot pair.

Datasets
--------
- ``ds_raw``: shot-level ``I`` and ``Q`` vs ``detuning`` (dims: ``qubit_pair``, ``n_runs``, ``detuning``).
- ``ds_fit``: readout metrics and optimum detuning per pair (from ``iq_sweep`` analysis).
- ``fit_results``: per-pair scalar results (serialized dataclass) for logging and state updates.

Results
-------
For each qubit pair, the node identifies an optimal detuning (by fidelity or visibility)
and extracts the readout axis and threshold used for the PSB readout discrimination.

Figures
-------
- Fidelity and visibility vs detuning
- Sweep summary (fidelity + visibility on twin axes)
- Shot histograms vs detuning (projected readout axis)
- Rotated IQ density at the optimal detuning with the chosen threshold

State update
------------
Updates the dot pair ``MEASURE`` voltage point detuning and stores the readout threshold
for the selected optimal detuning (only for successful pairs).


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `operation` | `readout` | Type of operation to perform. Default is "readout". |
| `sweep_name` | `detuning` | Name of the swept coordinate in ds_raw (e.g. "detuning", "integration_time"). |
| `optimization_metric` | `fidelity` | Metric used to pick the optimal sweep value for state updates.
Both the fidelity and visibility optima are always recorded regardless of this choice. |
| `labeled_states` | `False` | Whether ds_raw contains labelled S/T preparations (Ig,Qg,Ie,Qe, as in a
Rabi-style IQ-blob experiment) or a single mixed-state acquisition (I,Q,
as in a PSB search where loading is random). Determines whether a confusion
matrix is computed. Default False = mixed-state mode. |
| `plot_kde` | `True` | For the resulting figure, optionally plot the kernel density estimation. If False, this plots 
the raw scatter plot. |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `reset_type` | `thermal` | The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or
"active_gef". Default is "thermal". |
| `qubit_pairs` | `['q1_q2']` | A list of qubit pair names which should participate in the execution of the node. Default is None. |
| `num_shots` | `2` | Number of shots to acquire per detuning point. Default is 100. |
| `qubit_pair_to_initialize` | `None` | Initialize the qubit pair. If None, it will default to the same pair as the qubit pair for measurement. |
| `qubit_to_pulse` | `None` | Optionally apply a pi pulse to the qubit. |
| `barrier_gate_voltage` | `0.0` | Barrier Gate Voltage to pulse to with the detuning. Default zero. |
| `detuning_min` | `-0.05` | Minimum detuning value for the sweep in volts. Default is -0.1 V. |
| `detuning_max` | `0.05` | Maximum detuning value for the sweep in volts. Default is 0.1 V. |
| `detuning_points` | `3` | Number of detuning points to sweep. Default is 21. |
| `ramp_duration` | `40` | Ramp duration to ramp to the measurement point. |
| `buffer_duration` | `16` | Buffer duration at the measurement point before readout pulse. |
| `initialization_macro` | `empty` | Which dot-pair macro runs for the preparation step (formerly ``dot_pair.initialize()``).
Both ``empty`` and ``initialize`` must exist on ``dot_pair.macros``. |
| `use_simulated_data` | `False` | If True, skip QUA compile/execute and build synthetic shot-by-shot I/Q
(Barthel-style forward model) for offline analysis. Default False. |
| `simulate` | `False` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `40000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |

## Metadata

| Key | Value |
|-----|-------|
| Timestamp | 2026-08-10T16:19:48 UTC |
| Node | 06a_PSB_search_opx_sweep_detuning |
| Duration | 0.4s |
| Status | completed with errors |
| Error | `QmServerDetectionError: Failed to detect to QuantumMachines server, failed to connect to cluster 'CS_3'. Tried connecting to 172.16.33.115:80.` |

---
*Generated by execute test infrastructure*
