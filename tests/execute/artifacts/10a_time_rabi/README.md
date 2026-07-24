# 10a_time_rabi

## Description


        TIME RABI PARITY DIFFERENCE
This sequence performs a time Rabi measurement with parity difference to characterize qubit coherence and
coupling. The measurement involves sweeping the duration of a qubit control pulse (typically an X180 pulse)
while measuring the parity state before and after the pulse using charge sensing via RF reflectometry or DC
current sensing.

The sequence uses voltage gate sequences to navigate through a triangle in voltage space (empty -
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each pulse duration,
the parity is measured before (P1) and after (P2) the qubit pulse, and the parity difference (P_diff) is
calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The parity difference signal reveals Rabi oscillations as a function of pulse duration, which can be used
to extract the qubit coupling strength, coherence time, and optimal pulse parameters.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X180 pulse amplitude and frequency).

State update:
    - The qubit x180 operation duration.


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `analysis_signal` | `E_p2_given_p1_0` | Which conditional expectation to use for fitting.
E_p2_given_p1_0: P(second=1 | first=0) — post-select on empty dot.
E_p2_given_p1_1: P(second=1 | first=1) — post-select on loaded dot. |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `qubits` | `['q1', 'q2']` | A list of qubit names which should participate in the execution of the node. Default is None. |
| `num_shots` | `10` | Number of averages to perform. Default is 100. |
| `min_wait_time_in_ns` | `16` | Minimum pulse duration in nanoseconds. Must be larger than 4 clock cycles. Default is 16 ns. |
| `max_wait_time_in_ns` | `10000` | Maximum pulse duration in nanoseconds. Default is 10000 ns (10 us). |
| `time_step_in_ns` | `100` | Step size for the pulse duration sweep in nanoseconds. Default is 52 ns. |
| `operation` | `x180` | Name of the qubit operation to perform. Default is 'x180'. |
| `simulate` | `False` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `40000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |

## Execution Output

![Figure](figure.png)


## Fit Results

### virtual_dot_1
| Parameter | Value |
|-----------|-------|
| `optimal_duration` | `nan` |
| `rabi_frequency` | `nan` |
| `decay_rate` | `nan` |
| `success` | `False` |
| `_fft_diag` | `{'fft_freqs': array([0.    , 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
       0.0008, 0.0009, 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
       0.0016, 0.0017, 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023,
       0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031,
       0.0032, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
       0.004 , 0.0041, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047,
       0.0048, 0.0049, 0.005 ]), 'fft_magnitude': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'peak_curve': None}` |
| `_sinusoid_fit` | `None` |

### virtual_dot_2
| Parameter | Value |
|-----------|-------|
| `optimal_duration` | `nan` |
| `rabi_frequency` | `nan` |
| `decay_rate` | `nan` |
| `success` | `False` |
| `_fft_diag` | `{'fft_freqs': array([0.    , 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007,
       0.0008, 0.0009, 0.001 , 0.0011, 0.0012, 0.0013, 0.0014, 0.0015,
       0.0016, 0.0017, 0.0018, 0.0019, 0.002 , 0.0021, 0.0022, 0.0023,
       0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003 , 0.0031,
       0.0032, 0.0033, 0.0034, 0.0035, 0.0036, 0.0037, 0.0038, 0.0039,
       0.004 , 0.0041, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047,
       0.0048, 0.0049, 0.005 ]), 'fft_magnitude': array([2.66453526e-15, 2.89827849e+00, 3.94514378e+00, 1.75253594e+00,
       2.87829045e+00, 4.05774760e-01, 1.16976297e+00, 2.11369105e+00,
       3.96115788e+00, 1.24109593e+00, 2.25117390e+00, 4.38980690e+00,
       7.74684755e-01, 3.66091484e+00, 4.26970833e+00, 2.15016197e+00,
       2.36571241e+00, 4.05647729e-01, 1.71389073e+00, 2.33753977e+00,
       2.35162294e+00, 1.64973495e+00, 1.01413691e+00, 1.96049883e+00,
       1.86125006e+00, 1.61614128e+00, 2.11030772e+00, 2.32566903e+00,
       1.62517238e+00, 3.70438298e-01, 2.13741346e+00, 8.06043688e-01,
       3.01868998e+00, 1.93914495e+00, 1.61100366e+00, 5.00368610e-01,
       1.05728072e+00, 1.64099449e+00, 3.41481639e+00, 1.33735372e+00,
       1.51895057e+00, 9.78149640e-01, 2.69473660e+00, 1.67969410e+00,
       2.07139185e+00, 3.56196537e+00, 1.85954502e+00, 4.67220553e-01,
       1.56425553e+00, 8.73190517e-01, 2.15952381e+00]), 'peak_curve': None}` |
| `_sinusoid_fit` | `None` |


## Metadata

| Key | Value |
|-----|-------|
| Timestamp | 2026-04-29T00:44:14 UTC |
| Node | 10a_time_rabi |
| Duration | 9.6s |
| Status | completed |

---
*Generated by execute test infrastructure*
