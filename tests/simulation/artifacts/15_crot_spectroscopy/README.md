# 15_crot_spectroscopy

## Description


        CROT (CONTROLLED-ROTATION) SPECTROSCOPY - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the exchange coupling J between two spin qubits and identify
the conditional resonance frequencies required for implementing a CROT (controlled-rotation) gate.

In exchange-coupled spin qubits, the resonance frequency of a target qubit depends on the spin state
of a control qubit. When the control qubit is in |↓⟩ vs |↑⟩, the target qubit's resonance frequency
shifts by the exchange coupling strength J. This state-dependent frequency shift enables conditional
quantum operations - the foundation of two-qubit gates in the Loss-DiVincenzo architecture.

This measurement performs a 2D sweep of drive frequency vs virtual barrier gate (or virtual exchange voltage)
to map out the exchange coupling as a function of the inter-dot tunnel coupling.

The QUA program sequence:
    1) Start at the initialization point.
    2) Step to the two-qubit exchange point using sticky elements (virtual barrier/exchange voltage).
    3) Apply the RF drive pulse to the target qubit while sweeping the drive frequency.
    4) Step to the initialization (operation) point.
    5) Step to the measurement point and read out the two-qubit state via parity readout.

The CROT spectroscopy works by:
    - At each virtual barrier/exchange voltage, the exchange coupling J varies.
    - The target qubit resonance splits into two frequencies (f_↓ and f_↑) separated by J.
    - Sweeping frequency vs barrier voltage produces a chevron-like pattern showing J(V_barrier).
    - The optimal exchange point and CROT drive frequencies can be extracted from this 2D map.

The CROT gate is equivalent to a CNOT gate up to single-qubit rotations. For high-fidelity CROT gates,
the Zeeman energy difference between qubits must be much larger than the exchange coupling J, ensuring
addressability and avoiding off-resonant rotations.

Prerequisites:
    - Having calibrated single-qubit gates (π and π/2 pulses) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having set the appropriate flux/gate voltages to enable exchange coupling between the qubits.

Before proceeding to the next node:
    - Extract the exchange coupling J from the frequency shift between the two resonance peaks.
    - Identify the conditional resonance frequencies f_↓ and f_↑ for CROT gate implementation.
    - Verify that J is sufficiently large for the desired gate speed but small enough for addressability.

State update:
    - exchange_coupling_J
    - crot_frequency_down
    - crot_frequency_up


## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `analysis_signal` | `E_p2_given_p1_0` | Which conditional expectation to use for fitting.
E_p2_given_p1_0: P(second=1 | first=0) — post-select on empty dot.
E_p2_given_p1_1: P(second=1 | first=1) — post-select on loaded dot. |
| `parity_pre_measurement` | `False` | Whether to use parity pre measurement. Default is False. |
| `multiplexed` | `False` | Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
or to play the experiment sequentially for each qubit (False). Default is False. |
| `use_state_discrimination` | `False` | Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
quadratures 'I' and 'Q'. Default is False. |
| `reset_wait_time` | `5000` | The wait time for qubit reset. |
| `qubit_pairs` | `['q1_q2']` | A list of qubit pair names which should participate in the execution of the node. Default is None. |
| `num_shots` | `1` | Number of averages to perform. Default is 100. |
| `exchange_min` | `-0.1` | Minimum virtual barrier / exchange voltage (V). Default is 0.0. |
| `exchange_max` | `0.1` | Maximum virtual barrier / exchange voltage (V). Default is 0.5. |
| `exchange_points` | `3` | Number of points in the exchange voltage sweep. Default is 50. |
| `esr_frequency_min` | `5000000000.0` | Minimum ESR drive frequency offset (Hz). Default is -50 MHz. |
| `esr_frequency_max` | `5500000000.0` | Maximum ESR drive frequency offset (Hz). Default is 50 MHz. |
| `esr_frequency_points` | `2` | Number of points in the ESR frequency sweep. Default is 100. |
| `duration` | `1048` | CROT drive pulse duration in nanoseconds. Default is 1024 ns. |
| `simulate` | `True` | Simulate the waveforms on the OPX instead of executing the program. Default is False. |
| `simulation_duration_ns` | `40000` | Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns. |
| `use_waveform_report` | `True` | Whether to use the interactive waveform report in simulation. Default is True. |
| `timeout` | `120` | Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s. |
| `load_data_id` | `None` | Optional QUAlibrate node run index for loading historical data. Default is None. |

## Simulation Output

![Simulation](simulation.png)

---
*Generated by simulation test infrastructure*

## Area Under Curve (Mean Voltage per Channel)

| Controller | Port | Mean Voltage (V) |
|------------|------|------------------|
| con1 | 1-1-1 | 1.955745e-03 |
| con1 | 5-1 | -2.087586e-04 |
| con1 | 5-2 | -2.087586e-04 |
| con1 | 5-3 | -4.214383e-06 |
| con1 | 5-4 | 0.000000e+00 |
| con1 | 5-5 | -2.224729e-16 |
