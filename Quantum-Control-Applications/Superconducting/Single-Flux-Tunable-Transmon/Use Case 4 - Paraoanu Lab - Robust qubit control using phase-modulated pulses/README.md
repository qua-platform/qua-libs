# Robust qubit control by implementing phase-modulated pulses

This project contains code to generate robust phase-modulated waveforms for qubit experiments and perform a Rabi amplitude-frequency sweeps. 

You can find more information about this technique in:

- [Research Paper](https://doi.org/10.1103/PhysRevResearch.6.013188): Kuzmanović, M., et al. "High-fidelity robust qubit control by phase-modulated pulses." *Phys. Rev. Research* 6, 013188 (2024).
- [Blog Post](link)



## Overview

The code is organized into two main sections:

1. **Waveform Generation:** This section defines a configuration dictionary and functions to generate various types of waveforms, including rectangular, super-Gaussian, and robust phase-modulated pulses. The generated waveforms are stored in the configuration file.

2. **QUA Program:** This section defines a QUA program to perform a Rabi amplitude and frequency sweep. The program sweeps over different amplitude and detuning values, measures the resulting I (in-phase) and Q (quadrature) components of the signal, and processes the data to obtain averaged results.

## Getting Started

### 1. **Waveform Generation**

The first part of the code is responsible for generating and configuring waveforms for the quantum experiment.

- **`config` Dictionary:** A configuration dictionary is defined to store the pulse and waveform configurations, which will be updated based on the specific waveforms generated.

- **Waveform Functions:**
  - `supergaussian(length, order, cutoff)`: Generates a super-Gaussian envelope.
  - `robust_wf(amp, length, mod=40e6, order=4, cutoff=1e-2)`: Generates a robust waveform with phase modulation.

- **Waveform Generation:**
  - The `pulse_flag` is used to select the type of pulse to generate (rectangular, super-Gaussian, or robust phase-modulated pulse).
  - The generated pulse is then added to the configuration dictionary and visualized using Matplotlib.

### 2. **QUA Program: Rabi Amplitude-Frequency Sweep**

The second part of the code defines a QUA program that performs a Rabi amplitude and frequency sweep.

- **Parameters:**
  - `n_detuning`: Number of detuning points.
  - `detuning_span`: Total span of detuning values.
  - `n_a`: Number of amplitude points.
  - `a_array`: Array of amplitude values.

- **QUA Program:**
  - A QUA program named `rabi_amp_freq` is created to perform the sweep.
  - The program iterates over different amplitude and detuning values, applies the corresponding pulse to the qubit, and measures the resulting I and Q components.
  - The data is streamed, buffered, averaged, and saved for analysis.



## References

The code was provided by Kuzmanović, M. 





