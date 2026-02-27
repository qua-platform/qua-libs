# 02b_resonator_spectroscopy_vs_power

## Description


        RESONATOR SPECTROSCOPY VERSUS READOUT POWER
This sequence involves measuring the resonator by sending a readout pulse and
demodulating the signals to extract the 'I' and 'Q' quadratures for all resonators
simultaneously. This is done across various readout frequencies and amplitudes.
Based on the results, one can then adjust the readout amplitude, choosing a
readout amplitude value just before the observed frequency splitting.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a_resonator_spectroscopy.py).
    - Having instantiated a starting readout amplitude.

State update:
    - The readout power: sensor.readout_resonator.set_output_power()
    - The readout frequency for the optimal readout power.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `buffer_from_crossing_threshold_in_dbm` | `1` | Buffer from the crossing threshold in dBm - the optimal readout power will be set to be this number in Db below
the threshold. Default is 1 dBm. |
| `derivative_crossing_threshold_in_hz_per_dbm` | `-50000` | Threshold for derivative crossing in Hz/dBm. Default is -50000 Hz/dBm. |
| `derivative_smoothing_window_num_points` | `10` | Size of the window in number of points corresponding to the rolling average (number of points). Default is 10. |
| `frequency_span_in_mhz` | `4.0` | Span of frequencies to sweep in MHz. Default is 15 MHz. |
| `frequency_step_in_mhz` | `2.0` | Step size for frequency sweep in MHz. Default is 0.1 MHz. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |
| `max_amp` | `0.1` | Maximum readout amplitude for the experiment. Default is 0.1. |
| `max_power_dbm` | `-25` | Maximum power level in dBm. Default is -25 dBm. |
| `min_power_dbm` | `-50` | Minimum power level in dBm. Default is -50 dBm. |
| `model_computed_fields` | `{}` |  |
| `model_config` | `{'extra': 'forbid', 'use_attribute_docstrings': True}` |  |
| `model_extra` | `None` |  |
| `model_fields` | `{'multiplexed': FieldInfo(annotation=bool, required=False, default=False, description='Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)\nor to play the experiment sequentially for each qubit (False). Default is False.'), 'use_state_discrimination': FieldInfo(annotation=bool, required=False, default=False, description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated\nquadratures 'I' and 'Q'. Default is False."), 'reset_wait_time': FieldInfo(annotation=int, required=False, default=5000, description='The wait time for qubit reset.'), 'num_shots': FieldInfo(annotation=int, required=False, default=100, description='Number of averages to perform. Default is 100.'), 'sensor_names': FieldInfo(annotation=Union[List[str], NoneType], required=False, default=None, description='The list of sensor dot names to be included in the measurement. '), 'frequency_span_in_mhz': FieldInfo(annotation=float, required=False, default=15, description='Span of frequencies to sweep in MHz. Default is 15 MHz.'), 'frequency_step_in_mhz': FieldInfo(annotation=float, required=False, default=0.1, description='Step size for frequency sweep in MHz. Default is 0.1 MHz.'), 'min_power_dbm': FieldInfo(annotation=int, required=False, default=-50, description='Minimum power level in dBm. Default is -50 dBm.'), 'max_power_dbm': FieldInfo(annotation=int, required=False, default=-25, description='Maximum power level in dBm. Default is -25 dBm.'), 'num_power_points': FieldInfo(annotation=int, required=False, default=100, description='Number of points of the readout power axis. Default is 100.'), 'max_amp': FieldInfo(annotation=float, required=False, default=0.1, description='Maximum readout amplitude for the experiment. Default is 0.1.'), 'derivative_crossing_threshold_in_hz_per_dbm': FieldInfo(annotation=int, required=False, default=-50000, description='Threshold for derivative crossing in Hz/dBm. Default is -50000 Hz/dBm.'), 'derivative_smoothing_window_num_points': FieldInfo(annotation=int, required=False, default=10, description='Size of the window in number of points corresponding to the rolling average (number of points). Default is 10.'), 'moving_average_filter_window_num_points': FieldInfo(annotation=int, required=False, default=10, description='Size of the moving average filter window (number of points). Default is 5.'), 'buffer_from_crossing_threshold_in_dbm': FieldInfo(annotation=int, required=False, default=1, description='Buffer from the crossing threshold in dBm - the optimal readout power will be set to be this number in Db below\nthe threshold. Default is 1 dBm.'), 'simulate': FieldInfo(annotation=bool, required=False, default=False, description='Simulate the waveforms on the OPX instead of executing the program. Default is False.'), 'simulation_duration_ns': FieldInfo(annotation=int, required=False, default=50000, description='Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns.'), 'use_waveform_report': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use the interactive waveform report in simulation. Default is True.'), 'timeout': FieldInfo(annotation=int, required=False, default=120, description='Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s.'), 'load_data_id': FieldInfo(annotation=Union[int, NoneType], required=False, default=None, description='Optional QUAlibrate node run index for loading historical data. Default is None.')}` |  |
| `model_fields_set` | `{'max_amp', 'reset_wait_time', 'num_shots', 'derivative_crossing_threshold_in_hz_per_dbm', 'timeout', 'sensor_names', 'derivative_smoothing_window_num_points', 'min_power_dbm', 'max_power_dbm', 'simulation_duration_ns', 'num_power_points', 'use_state_discrimination', 'buffer_from_crossing_threshold_in_dbm', 'moving_average_filter_window_num_points', 'load_data_id', 'frequency_step_in_mhz', 'multiplexed', 'simulate', 'frequency_span_in_mhz', 'use_waveform_report'}` |  |
| `moving_average_filter_window_num_points` | `10` | Size of the moving average filter window (number of points). Default is 5. |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `num_power_points` | `100` | Number of points of the readout power axis. Default is 100. |
| `num_shots` | `1` | Number of averages to perform. Default is 100. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `sensor_names` | `None` | The list of sensor dot names to be included in the measurement.  |
| `simulate` | `True` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `20000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `targets` | `None` |  |
| `targets_name` | `qubits` |  |
| `timeout` | `30` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*
