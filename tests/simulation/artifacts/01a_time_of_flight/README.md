# 01a_time_of_flight

## Description


        TIME OF FLIGHT - OPX+ & LF-FEM
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation
      delay of the readout pulse. This value is utilized to offset the acquisition window relative
      to when the readout pulse is dispatched.

    - Analog Inputs Offset: Due to minor impedance mismatches, the signals captured by
      the OPX might exhibit slight DC offsets.

    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates
      the ADC, the variable gain of the OPX analog input, ranging from -12 dB to 20 dB,
      can be modified to fit the signal within the ADC range of +/-0.5V.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).

State update:
    - The time of flight: sensor.readout_resonator.time_of_flight.
    - The analog input offsets: sensor.readout_resonator.opx_input.offset.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |
| `model_computed_fields` | `{}` |  |
| `model_config` | `{'extra': 'forbid', 'use_attribute_docstrings': True}` |  |
| `model_extra` | `None` |  |
| `model_fields` | `{'multiplexed': FieldInfo(annotation=bool, required=False, default=False, description='Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)\nor to play the experiment sequentially for each qubit (False). Default is False.'), 'use_state_discrimination': FieldInfo(annotation=bool, required=False, default=False, description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated\nquadratures 'I' and 'Q'. Default is False."), 'reset_wait_time': FieldInfo(annotation=int, required=False, default=5000, description='The wait time for qubit reset.'), 'num_shots': FieldInfo(annotation=int, required=False, default=100, description='Number of averages to perform. Default is 100.'), 'time_of_flight_in_ns': FieldInfo(annotation=Union[int, NoneType], required=False, default=28, description='Time of flight in nanoseconds. Default is 28 ns.'), 'readout_amplitude_in_v': FieldInfo(annotation=Union[float, NoneType], required=False, default=0.1, description='Readout amplitude in volts. Default is 0.1 V.'), 'readout_length_in_ns': FieldInfo(annotation=Union[int, NoneType], required=False, default=1000, description='Readout length in nanoseconds. Default is 1µs.'), 'sensor_names': FieldInfo(annotation=Union[list[str], NoneType], required=False, default=None, description='List of sensor names. Default is None.'), 'simulate': FieldInfo(annotation=bool, required=False, default=False, description='Simulate the waveforms on the OPX instead of executing the program. Default is False.'), 'simulation_duration_ns': FieldInfo(annotation=int, required=False, default=50000, description='Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns.'), 'use_waveform_report': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use the interactive waveform report in simulation. Default is True.'), 'timeout': FieldInfo(annotation=int, required=False, default=120, description='Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s.'), 'load_data_id': FieldInfo(annotation=Union[int, NoneType], required=False, default=None, description='Optional QUAlibrate node run index for loading historical data. Default is None.')}` |  |
| `model_fields_set` | `{'multiplexed', 'reset_wait_time', 'simulation_duration_ns', 'simulate', 'use_waveform_report', 'timeout', 'load_data_id', 'time_of_flight_in_ns', 'readout_length_in_ns', 'num_shots', 'use_state_discrimination', 'sensor_names', 'readout_amplitude_in_v'}` |  |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `num_shots` | `10` | Number of averages to perform. Default is 100. |
| `readout_amplitude_in_v` | `0.1` | Readout amplitude in volts. Default is 0.1 V. |
| `readout_length_in_ns` | `1000` | Readout length in nanoseconds. Default is 1µs. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `sensor_names` | `None` | List of sensor names. Default is None. |
| `simulate` | `True` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `10000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `targets` | `None` |  |
| `targets_name` | `qubits` |  |
| `time_of_flight_in_ns` | `28` | Time of flight in nanoseconds. Default is 28 ns. |
| `timeout` | `30` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*
