# Two Fixed Frequency Coupled Transmon Superconducting Qubit

<img align="right" src="Two Flux Tunable Transmon Setup.PNG" alt="drawing" width="400"/>

## Experimental setup and context

This repository includes a collection of scripts for performing various experiments on two fixed-frequency transmons,
encompassing qubit calibration, cross-resonance (CR) gate tuning, and CNOT gate formation.
Each transmon is equipped with individual qubit drive line and a shared readout transmission line.

These scripts have been tested in a real experimental setup, but they are provided as-is without any guarantees.

While they can serve as a foundation for new labs or experiments, users should be prepared to make necessary adaptations to fit their specific setups. Please use them with caution.


## Script Descriptions

0. [Hello QUA](00_hello_qua.py) - A script used for introductory experiments with QUA.

1. [Time of Flight](01_time_of_flight.py) - Measures the time delay between qubit control and readout.

2. [Single Resonator Spectroscopy](02_resonator_spectroscopy_single.py) - Performs a frequency sweep on a single resonator.

3. [Multiplexed Resonator Spectroscopy](03_resonator_spectroscopy_multiplexed.py) - Conducts a simultaneous frequency sweep on multiple resonators.

4. [Resonator Spectroscopy vs Amplitude](04_resonator_spectroscopy_vs_amplitude.py) - Analyzes resonator response as a function of readout power.

5. [Qubit Spectroscopy](05_qubit_spectroscopy.py) - Executes a frequency sweep on qubits while measuring the resonator.

6. [Rabi Chevron](06_rabi_chevron.py) - Performs a 2D frequency vs qubit drive amplitude sweep to obtain the Rabi chevron.

7. [Power Rabi](07_power_rabi.py) - A Rabi experiment sweeping the amplitude of the microwave pulse.

8. [Ramsey Chevron](08_ramsey_chevron.py) - Conducts a 2D sweep (detuning vs idle time) to generate a Ramsey chevron pattern.

9. **Readout Optimization:**
    - [Frequency Optimization](09a_readout_optimization_freq.py) - Optimizes readout frequency by scanning frequency and calculating the signal-to-noise ratio (SNR).
    - [Amplitude Optimization](09b_readout_optimization_amp.py) - Measures readout fidelity for different readout powers.
    - [Duration Optimization](09c_readout_optimization_duration.py) - Analyzes SNR as a function of readout pulse duration.
    - [Weight Optimization](09d_readout_weight_optimization.py) - Computes normalized optimal readout weights from demodulated state trajectories.

10. [IQ Blobs](10_IQ_blobs.py) - Creates IQ blobs for the qubit in ground and excited states, providing parameters for active reset.

11. [T1 Measurement](11_T1.py) - Measures the qubit's relaxation time T1.

12. [T2 Echo Measurement](12_T2echo.py) - Measures the T2 echo time of the qubit.

13. [T2 Ramsey Measurement](13_T2ramsey.py) - Measures the T2 Ramsey time using Ramsey interference.

14. **Calibration:**
    - [AC Stark Calibration](14a_ac_stark_calibration.py) - Calibrates the AC Stark shift for a qubit.
    - [DRAG Calibration](14b_drag_calibration.py) - Performs calibration for DRAG (Derivative Removal by Adiabatic Gate).

15. [All XY Experiment](15_allxy.py) - Estimates gate imperfections using the ALL XY protocol.

16. **Randomized Benchmarking:**
    - [Single Qubit Randomized Benchmarking](16a_single_qubit_RB.py) - Measures single-qubit gate fidelity through randomized benchmarking.
    - [Interleaved Single Qubit Randomized Benchmarking](16b_single_qubit_RB_interleaved.py) - Conducts interleaved randomized benchmarking for more robust fidelity estimation.

17. **Cross-Resonance Calibration:**
    - [CR Time Rabi with 1Q QST](17a_CR_time_rabi_1q_QST.py) - Measures QST on the target qubit by applying CR drive on the control qubit.
    - [Cancel CR Time Rabi with 1Q QST](17b_cancelCR_time_rabi_1q_QST.py) - Measures QST on the target qubit by applying CR drive on the control qubit along with cancel drive on the target qubit.
    - [Echoed CR Time Rabi with 1Q QST](17c_echoCR_time_rabi_1q_QST.py) - Measures QST on the target qubit by applying echoed CR drive on the control qubit along with cancel drive on the target qubit.

18. **Hamiltonian Tomography:**
    - [Unit Hamiltonian Tomography](18a_CR_calib_unit_hamiltonian_tomography.py) - Calibrates CR gates by estimating Hamiltonian parameters through state tomography.
    - [CR Drive Amplitude Calibration](18b_CR_calib_cr_drive_amplitude.py) - Calibrates the amplitude of the CR drive.
    - [CR Drive Phase Calibration](18c_CR_calib_cr_drive_phase.py) - Calibrates the phase of the CR drive and CR cancel drive.
    - [CR Cancel Phase Calibration](18d_CR_calib_cr_cancel_phase.py) - Calibrates the phase of the CR cancel drive.
    - [CR Cancel Amplitude Calibration](18e_CR_calib_cr_cancel_amplitude.py) - Calibrates the amplitude of the CR cancel drive.
    - [CR Driven Ramsey Measurement](18f_CR_calib_cr_driven_ramsey_RCVersion.py) - Measures the phase shift due to the AC Stark shift.

19. [CNOT Gate Experiment](19_CNOT.py) - Demonstrates a controlled-NOT operation between two fixed frequency transmons.

## Miscellaneous

- [macros.py](macros.py) - Collection of useful macros for the experiment.
- [CR Hamilotonian Tomography Functions](cr_hamiltonian_tomography.py) - Implements analysis and plotting functions used for Hamiltonian tomography.

## Configuration Files

- [configuration.py](configuration.py) - Base configuration file for the experiment setup.
- [configuration_lf_fem.py](configuration_lf_fem.py) - Configuration file for Low-Frequency FEM setup.
- [configuration_lf_fem_and_octave.py](configuration_lf_fem_and_octave.py) - Configuration for Low-Frequency FEM with Octave integration.
- [configuration_mw_fem.py](configuration_mw_fem.py) - Configuration file for Microwave FEM setup.
- [configuration_with_octave.py](configuration_with_octave.py) - Example configuration that integrates with Octave.

## Set-ups with Octave

The configuration included in this folder correspond to a set-up without Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_clock_and_calibration.py](octave_clock_and_calibration.py): A file __to execute__ in order to configure the Octave's clock and calibrate the Octave.
3. [set_octave.py](set_octave.py): A helper function to ease the octave initialization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).

