Bayesian Frequency Estimation for Quantum Qubits
================================================

This experiment implements a Bayesian frequency estimation protocol to measure and track
the frequency of quantum qubits over time. The method uses sequential Bayesian updates
to estimate the qubit frequency based on measurement outcomes, providing real-time
frequency tracking and noise characterization.

Overview
--------
The experiment performs the following sequence for each qubit:
1. Applies a flux pulse to shift the qubit frequency
2. Executes a Ramsey-like sequence with variable idle time
3. Measures the qubit state
4. Updates the Bayesian probability distribution P(f) over frequency using measurement outcomes
5. Estimates the most likely frequency from the posterior distribution
6. Repeats the process to track frequency evolution over time

The results include:
- Time-resolved Bayesian probability distributions P(f|t)
- Estimated frequency trajectories
- Single-shot measurement data (optional)

Prerequisites
-------------
Before running this experiment, ensure the following calibrations are complete:
- Time of flight calibration (offsets and gains)
- IQ mixer calibration for the readout line
- Resonator spectroscopy (resonance frequency identification)
- Readout pulse amplitude and duration configuration
- Resonator depletion time specification in the state
- SPAM (State Preparation and Measurement) calibration (confusion matrix)

Parameters
----------
The experiment is configured through the Parameters class. Key parameters include:

Qubit Selection:
    qubits : Optional[List[str]]
        List of qubit names to measure. Default: None
        Example: ["qC1", "qC2"] to measure multiple qubits

Experiment Parameters:
    num_repetitions : int
        Number of measurement repetitions per time point. Default: 5000
        Higher values improve frequency resolution but increase measurement time.

    detuning : int
        Nominal detuning frequency in Hz. Default: 2 MHz
        This is the frequency offset used in the Ramsey sequence.

    physical_detuning : int
        Physical detuning applied via flux bias in Hz. Default: 5 MHz
        This creates the actual frequency shift that we want to measure.

    min_wait_time_in_ns : int
        Minimum idle time in the Ramsey sequence (nanoseconds). Default: 36 ns
        The shortest time between the two π/2 pulses.

    max_wait_time_in_ns : int
        Maximum idle time in the Ramsey sequence (nanoseconds). Default: 8000 ns
        The longest time between the two π/2 pulses.

    wait_time_step_in_ns : int
        Step size for the idle time sweep (nanoseconds). Default: 120 ns
        Controls the resolution of the time axis.

Bayesian Estimation Parameters:
    f_min : float
        Minimum frequency in the Bayesian search range (MHz). Default: 1.05 MHz
        Lower bound of the frequency prior distribution.

    f_max : float
        Maximum frequency in the Bayesian search range (MHz). Default: 1.25 MHz
        Upper bound of the frequency prior distribution.
        Note: Frequency range should be between 0 and 8 MHz due to QUA fixed variable limitations.

    df : float
        Frequency resolution step size (MHz). Default: 0.002 MHz
        Smaller values improve frequency resolution but increase computational overhead.
        The number of frequency bins is (f_max - f_min) / df.

Data Collection:
    keep_shot_data : bool
        Whether to save individual shot measurement outcomes. Default: True
        If False, only the Bayesian distributions and estimated frequencies are saved.
        Setting to False reduces memory usage for long experiments.

Execution Parameters:
    simulate : bool
        Run in simulation mode instead of hardware. Default: False
        When True, the program is compiled but not executed on hardware.

    simulation_duration_ns : int
        Duration for simulation mode (nanoseconds). Default: 2500 ns
        Only used when simulate=True.

    timeout : int
        Maximum execution time in seconds. Default: 100
        The experiment will timeout if not completed within this time.

    load_data_id : Optional[int]
        Load previously saved data instead of running new experiment. Default: None
        If provided, loads data from the specified experiment ID instead of executing.

    multiplexed : bool
        Whether to use multiplexed readout. Default: False
        Currently not fully implemented.

Output Data
----------
The experiment returns an xarray Dataset with the following variables:

    Pf : (qubit, repetition, vf)
        Bayesian probability distribution over frequency for each repetition.
        Normalized such that sum(Pf) = 1 for each repetition.

    estimated_frequency : (qubit, repetition)
        Most likely frequency estimate (MHz) for each repetition, computed as argmax(Pf).

    state : (qubit, repetition, t) [if keep_shot_data=True]
        Single-shot measurement outcomes (0 or 1) for each time point and repetition.

    time_stamp : (qubit, repetition)
        Timestamp for each measurement (seconds since experiment start).

Figures
-------
The experiment generates several visualization figures:

    PF_figure : Bayesian probability distribution P(f|t) as a function of time
    state_figure : Single-shot measurement outcomes (if keep_shot_data=True)
    estimated_frequency_figure : Estimated frequency trajectory over time

Usage Example
-------------
```python
from qualibrate import QualibrationNode

# Create node with custom parameters
node = QualibrationNode(
    name="FrequencyBayes",
    parameters=Parameters(
        qubits=["qC2"],
        num_repetitions=10000,
        f_min=1.0,
        f_max=1.5,
        df=0.001,
        keep_shot_data=True
    )
)

# Run the experiment
node.run()
```

Notes
-----
- The Bayesian update uses SPAM parameters (alpha, beta) from the qubit's confusion matrix
- Frequency estimation is limited to 0-8 MHz range due to QUA fixed variable constraints
- The flux shift is automatically calculated based on physical_detuning and qubit properties
- Frame rotation is reset after each measurement to prevent phase accumulation
- Normalization prevents numerical underflow in the Bayesian updates

References
----------
Berritta, F. et al. Real-time two-axis control of a spin qubit. Nat Commun 15, (2024).

This implementation is based on Bayesian parameter estimation techniques for quantum
metrology, adapted for real-time frequency tracking in superconducting qubits.
