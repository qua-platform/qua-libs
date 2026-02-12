# 09b_time_rabi_chevron_parity_diff

## Description


        TIME RABI CHEVRON PARITY DIFFERENCE
This sequence performs a 2D chevron measurement with parity difference to characterize qubit coherence and
coupling as a function of both pulse duration and frequency detuning. The measurement involves sweeping both
the duration of a qubit control pulse (typically an X180 pulse) and the frequency detuning while measuring
the parity state before and after the pulse using charge sensing via RF reflectometry or DC current sensing.

The sequence uses voltage gate sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each combination
of pulse duration and frequency detuning, the parity is measured before (P1) and after (P2) the qubit pulse,
and the parity difference (P_diff) is calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The 2D chevron pattern in the parity difference signal reveals the qubit coupling strength as a function
of both time and frequency, creating a characteristic chevron shape. This measurement is particularly useful
for characterizing two-qubit gates, understanding the dynamics of coupled quantum dots, and identifying
optimal operating points for qubit control.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement), including sensor the dot bias.
    - Rough guess of the qubit pulse calibration (X180 pulse amplitude and frequency).

State update:
    - The qubit x180 operation duration and frequency.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `frequency_span_in_mhz` | `100.0` | Span of frequencies to sweep in MHz. Default is 2 MHz. |
| `frequency_step_in_mhz` | `1.0` | Step size for the frequency detuning sweep in MHz. Default is 0.025 MHz. |
| `gap_wait_time_in_ns` | `128` | Wait time between initialization and X180 pulse in nanoseconds. Default is 128 ns. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |
| `max_wait_time_in_ns` | `400` | Maximum pulse duration in nanoseconds. Default is 10000 ns (10 us). |
| `min_wait_time_in_ns` | `0` | Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns. |
| `model_computed_fields` | `{}` |  |
| `model_config` | `{'extra': 'forbid', 'use_attribute_docstrings': True}` |  |
| `model_extra` | `None` |  |
| `model_fields` | `{'multiplexed': FieldInfo(annotation=bool, required=False, default=False, description='Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)\nor to play the experiment sequentially for each qubit (False). Default is False.'), 'use_state_discrimination': FieldInfo(annotation=bool, required=False, default=False, description="Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated\nquadratures 'I' and 'Q'. Default is False."), 'reset_wait_time': FieldInfo(annotation=int, required=False, default=5000, description='The wait time for qubit reset.'), 'qubits': FieldInfo(annotation=Union[List[str], NoneType], required=False, default=None, description='A list of qubit names which should participate in the execution of the node. Default is None.'), 'num_shots': FieldInfo(annotation=int, required=False, default=100, description='Number of averages to perform. Default is 100.'), 'min_wait_time_in_ns': FieldInfo(annotation=int, required=False, default=16, description='Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns.'), 'max_wait_time_in_ns': FieldInfo(annotation=int, required=False, default=10000, description='Maximum pulse duration in nanoseconds. Default is 10000 ns (10 us).'), 'time_step_in_ns': FieldInfo(annotation=int, required=False, default=52, description='Step size for the pulse duration sweep in nanoseconds. Default is 52 ns.'), 'frequency_span_in_mhz': FieldInfo(annotation=float, required=False, default=2, description='Span of frequencies to sweep in MHz. Default is 2 MHz.'), 'frequency_step_in_mhz': FieldInfo(annotation=float, required=False, default=0.025, description='Step size for the frequency detuning sweep in MHz. Default is 0.025 MHz.'), 'gap_wait_time_in_ns': FieldInfo(annotation=int, required=False, default=128, description='Wait time between initialization and X180 pulse in nanoseconds. Default is 128 ns.'), 'operation': FieldInfo(annotation=str, required=False, default='x180', description="Name of the qubit operation to perform. Default is 'x180'."), 'simulate': FieldInfo(annotation=bool, required=False, default=False, description='Simulate the waveforms on the OPX instead of executing the program. Default is False.'), 'simulation_duration_ns': FieldInfo(annotation=int, required=False, default=50000, description='Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns.'), 'use_waveform_report': FieldInfo(annotation=bool, required=False, default=True, description='Whether to use the interactive waveform report in simulation. Default is True.'), 'timeout': FieldInfo(annotation=int, required=False, default=120, description='Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s.'), 'load_data_id': FieldInfo(annotation=Union[int, NoneType], required=False, default=None, description='Optional QUAlibrate node run index for loading historical data. Default is None.')}` |  |
| `model_fields_set` | `{'min_wait_time_in_ns', 'frequency_span_in_mhz', 'num_shots', 'frequency_step_in_mhz', 'max_wait_time_in_ns'}` |  |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `num_shots` | `4` | Number of averages to perform. Default is 100. |
| `operation` | `x180` | Name of the qubit operation to perform. Default is 'x180'. |
| `qubits` | `None` | A list of qubit names which should participate in the execution of the node. Default is None. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `simulate` | `False` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `50000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `targets` | `None` |  |
| `targets_name` | `qubits` |  |
| `time_step_in_ns` | `52` | Step size for the pulse duration sweep in nanoseconds. Default is 52 ns. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |

## Analysis Output

![Analysis simulation](simulation.png)

---
*Generated by analysis test infrastructure (virtual_qpu)*
