# 10a_ramsey_parity_diff

## Description


        RAMSEY PARITY DIFFERENCE
This sequence performs a Ramsey measurement with parity difference to characterize the qubit frequency
and the qubit Ramsey dephasing time T2*. The measurement involves sweeping the idle time of the qubit between
two π/2 rotations. PSB is used to measure the parity of the resulting state.

The sequence uses voltage sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each pulse duration,
the parity is measured before (P1) and after (P2) the qubit pulse, and the parity difference (P_diff) is
calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The parity difference signal reveals Ramsey oscillations as a function of pulse duration, which can be used
to extract the qubit coupling strength, coherence time, and optimal pulse parameters.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

State update:
    - The qubit Larmor frequency.
    - The qubit  T2* (Ramsey) time.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `frequency_detuning_in_mhz` | `1.0` | Frequency detuning in MHz. Default is 1.0 MHz. |
| `gap_wait_time_in_ns` | `1024` | Wait time between initialization and first X90 pulse in nanoseconds. Default is 128 ns. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |
| `model_computed_fields` | `{}` |  |
| `model_config` | `{'extra': 'forbid', 'use_attribute_docstrings': True}` |  |
| `model_extra` | `None` |  |
| `model_fields` | `{'multiplexed': FieldInfo(annotation=bool, required=False, default=False, description='Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)\nor to play the experiment sequentially for each qubit (False). Default is False.'), 'use_state_discrimination': FieldInfo(annotation=bool, required=False, default=False, description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated\nquadratures 'I' and 'Q'. Default is False."), 'reset_type': FieldInfo(annotation=Literal['thermal', 'active', 'active_gef'], required=False, default='thermal', description='The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or\n"active_gef". Default is "thermal".'), 'qubits': FieldInfo(annotation=Union[List[str], NoneType], required=False, default=None, description='A list of qubit names which should participate in the execution of the node. Default is None.'), 'num_shots': FieldInfo(annotation=int, required=False, default=100, description='Number of averages to perform. Default is 100.'), 'tau_min': FieldInfo(annotation=int, required=False, default=16, description='Minimum idle time in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns.'), 'tau_max': FieldInfo(annotation=int, required=False, default=10000, description='Maximum idle time in nanoseconds. Default is 10000 ns (10 us).'), 'tau_step': FieldInfo(annotation=int, required=False, default=16, description='Step size for the idle time sweep in nanoseconds. Default is 16 ns.'), 'frequency_detuning_in_mhz': FieldInfo(annotation=float, required=False, default=1.0, description='Frequency detuning in MHz. Default is 1.0 MHz.'), 'gap_wait_time_in_ns': FieldInfo(annotation=int, required=False, default=2048, description='Wait time between initialization and first X90 pulse in nanoseconds. Default is 128 ns.'), 'simulate': FieldInfo(annotation=bool, required=False, default=False, description='Simulate the waveforms on the OPX instead of executing the program. Default is False.'), 'simulation_duration_ns': FieldInfo(annotation=int, required=False, default=50000, description='Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns.'), 'use_waveform_report': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use the interactive waveform report in simulation. Default is True.'), 'timeout': FieldInfo(annotation=int, required=False, default=120, description='Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s.'), 'load_data_id': FieldInfo(annotation=Union[int, NoneType], required=False, default=None, description='Optional QUAlibrate node run index for loading historical data. Default is None.')}` |  |
| `model_fields_set` | `{'num_shots', 'use_state_discrimination', 'simulate', 'gap_wait_time_in_ns', 'simulation_duration_ns', 'tau_min', 'load_data_id', 'multiplexed', 'use_waveform_report', 'qubits', 'frequency_detuning_in_mhz', 'timeout', 'tau_step', 'tau_max', 'reset_type'}` |  |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `num_shots` | `10` | Number of averages to perform. Default is 100. |
| `qubits` | `None` | A list of qubit names which should participate in the execution of the node. Default is None. |
| `reset_type` | `thermal` | The qubit reset method to use. Must be implemented as a method of Quam.qubit. Can be "thermal", "active", or
"active_gef". Default is "thermal". |
| `simulate` | `True` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `20000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `targets` | `None` |  |
| `targets_name` | `qubits` |  |
| `tau_max` | `10000` | Maximum idle time in nanoseconds. Default is 10000 ns (10 us). |
| `tau_min` | `500` | Minimum idle time in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns. |
| `tau_step` | `500` | Step size for the idle time sweep in nanoseconds. Default is 16 ns. |
| `timeout` | `30` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*
