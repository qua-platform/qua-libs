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

| Parameter | Value |
|-----------|-------|
| `buffer_duration` | `16` |
| `detuning_max` | `0.1` |
| `detuning_min` | `-0.1` |
| `detuning_points` | `5` |
| `labeled_states` | `False` |
| `load_data_id` | `None` |
| `model_computed_fields` | `{}` |
| `model_config` | `{'extra': 'forbid', 'use_attribute_docstrings': True}` |
| `model_extra` | `None` |
| `model_fields` | `{'multiplexed': FieldInfo(annotation=bool, required=False, default=False, description='Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)\nor to play the experiment sequentially for each qubit (False). Default is False.'), 'use_state_discrimination': FieldInfo(annotation=bool, required=False, default=False, description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated\nquadratures 'I' and 'Q'. Default is False."), 'reset_wait_time': FieldInfo(annotation=int, required=False, default=5000, description='The wait time for qubit reset.'), 'qubit_pairs': FieldInfo(annotation=Union[List[str], NoneType], required=False, default=None, description='A list of qubit pair names which should participate in the execution of the node. Default is None.'), 'num_shots': FieldInfo(annotation=int, required=False, default=100, description='Number of shots to acquire per detuning point. Default is 100.'), 'detuning_min': FieldInfo(annotation=float, required=False, default=-0.1, description='Minimum detuning value for the sweep in volts. Default is -0.1 V.'), 'detuning_max': FieldInfo(annotation=float, required=False, default=0.1, description='Maximum detuning value for the sweep in volts. Default is 0.1 V.'), 'detuning_points': FieldInfo(annotation=int, required=False, default=21, description='Number of detuning points to sweep. Default is 21.'), 'ramp_duration': FieldInfo(annotation=int, required=False, default=40, description='Ramp duration to ramp to the measurement point.'), 'buffer_duration': FieldInfo(annotation=int, required=False, default=16, description='Buffer duration at the measurement point before readout pulse.'), 'operation': FieldInfo(annotation=Literal['readout', 'readout_QND'], required=False, default='readout', description='Type of resonator operation whose readout parameters are optimised. Default "readout".'), 'sweep_name': FieldInfo(annotation=str, required=False, default='detuning', description='Name of the swept coordinate in ds_raw (fixed to "detuning" here but kept\nexplicit so iq_sweep analysis remains generic).'), 'optimization_metric': FieldInfo(annotation=Literal['fidelity', 'visibility'], required=False, default='fidelity', description='Metric used to pick the optimal detuning for state updates.\nBoth fidelity and visibility optima are recorded regardless of this choice.'), 'labeled_states': FieldInfo(annotation=bool, required=False, default=False, description='Whether ds_raw contains labelled S/T preparations (Ig,Qg,Ie,Qe) or a\nsingle mixed-state acquisition (I,Q). PSB search uses random loading, so\ndefaults to False. Set True only if you explicitly prepare S and T.'), 'simulate': FieldInfo(annotation=bool, required=False, default=False, description='Simulate the waveforms on the OPX instead of executing the program. Default is False.'), 'simulation_duration_ns': FieldInfo(annotation=int, required=False, default=50000, description='Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns.'), 'use_waveform_report': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use the interactive waveform report in simulation. Default is True.'), 'timeout': FieldInfo(annotation=int, required=False, default=120, description='Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s.'), 'load_data_id': FieldInfo(annotation=Union[int, NoneType], required=False, default=None, description='Optional QUAlibrate node run index for loading historical data. Default is None.')}` |
| `model_fields_set` | `{'detuning_points', 'labeled_states', 'detuning_max', 'optimization_metric', 'detuning_min', 'simulate', 'num_shots'}` |
| `multiplexed` | `False` |
| `num_shots` | `200` |
| `operation` | `readout` |
| `optimization_metric` | `fidelity` |
| `qubit_pairs` | `None` |
| `ramp_duration` | `40` |
| `reset_wait_time` | `5000` |
| `simulate` | `False` |
| `simulation_duration_ns` | `50000` |
| `sweep_name` | `detuning` |
| `targets` | `None` |
| `targets_name` | `qubits` |
| `timeout` | `120` |
| `use_state_discrimination` | `False` |
| `use_waveform_report` | `True` |

## Fit Results

| qubit_pair | optimal_detuning | F* @ detuning | V* @ detuning | F (%) | V | success |
|------------|------------------|---------------|---------------|-------|---|---------|
| q1_q2 | 0.05 | 0.05 | 0.05 | 93.8 | 0.877 | True |
| q1_q2_alias_1 | 0 | 0 | 0 | 93.2 | 0.863 | True |
| q1_q2_alias_2 | -0.05 | -0.05 | -0.05 | 95.0 | 0.900 | True |

## Figures

![fidelity_vs_detuning](fidelity_vs_detuning.png)
![visibility_vs_detuning](visibility_vs_detuning.png)
![sweep_summary](sweep_summary.png)
