# 06a_PSB_search_opx_sweep_detuning

## Description


        PAULI SPIN BLOCKADE SEARCH - Sweep Detuning
The goal of this sequence is to find the Pauli Spin Blockade (PSB) region.
To do so, the following triangle in voltage space (empty - random initialization - measurement) is applied using OPX
channels on the fast lines of the bias-tees while sweeping the "measure" voltage point along the detuning axis.

The OPX measures the response via RF reflectometry or DC current sensing during the readout window
(last segment of the triangle). A single-point averaging is performed and the data is extracted while
the program is running to display the results.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage on average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time, unless a compensation pulse is played.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.

State update:
    - The optimal detuning value for PSB readout, as the voltage point associated with the .measure macro.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `qubit_pairs` | `['q1_q2']` | A list of qubit pair names which should participate in the execution of the node. Default is None. |
| `num_shots` | `2` | Number of shots to acquire per detuning point. Default is 100. |
| `detuning_min` | `-0.05` | Minimum detuning value for the sweep in volts. Default is -0.1 V. |
| `detuning_max` | `0.05` | Maximum detuning value for the sweep in volts. Default is 0.1 V. |
| `detuning_points` | `3` | Number of detuning points to sweep. Default is 21. |
| `ramp_duration` | `40` | Ramp duration to ramp to the measurement point. |
| `buffer_duration` | `16` | Buffer duration at the measurement point before readout pulse. |
| `operation` | `readout` | Type of resonator operation whose readout parameters are optimised. Default "readout". |
| `sweep_name` | `detuning` | Name of the swept coordinate in ds_raw (fixed to "detuning" here but kept
explicit so iq_sweep analysis remains generic). |
| `optimization_metric` | `fidelity` | Metric used to pick the optimal detuning for state updates.
Both fidelity and visibility optima are recorded regardless of this choice. |
| `labeled_states` | `False` | Whether ds_raw contains labelled S/T preparations (Ig,Qg,Ie,Qe) or a
single mixed-state acquisition (I,Q). PSB search uses random loading, so
defaults to False. Set True only if you explicitly prepare S and T. |
| `simulate` | `False` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `40000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |

## Execution Output

![Fidelity Vs Detuning](fidelity_vs_detuning.png)
![Visibility Vs Detuning](visibility_vs_detuning.png)
![Sweep Summary](sweep_summary.png)


## Fit Results

### q1_q2
| Parameter | Value |
|-----------|-------|
| `sweep_name` | `detuning` |
| `sweep_values` | `[-0.05, -0.01666666666666667, 0.016666666666666663]` |
| `optimal_sweep_value` | `nan` |
| `optimal_sweep_index` | `-1` |
| `optimal_sweep_value_fidelity` | `nan` |
| `optimal_sweep_index_fidelity` | `-1` |
| `optimal_sweep_value_visibility` | `nan` |
| `optimal_sweep_index_visibility` | `-1` |
| `iw_angle` | `nan` |
| `I_threshold` | `nan` |
| `ge_threshold` | `nan` |
| `readout_threshold` | `nan` |
| `readout_projector` | `{'wI': nan, 'wQ': nan, 'offset': nan}` |
| `readout_fidelity` | `nan` |
| `visibility` | `nan` |
| `confusion_matrix` | `[[nan, nan], [nan, nan]]` |
| `success` | `False` |


## Metadata

| Key | Value |
|-----|-------|
| Timestamp | 2026-04-17T03:11:03 UTC |
| Node | 06a_PSB_search_opx_sweep_detuning |
| Duration | 11.1s |
| Status | completed |

---
*Generated by execute test infrastructure*
