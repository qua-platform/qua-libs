# Two flux tunable transmons with the standard configuration

<img align="right" src="Two Flux Tunable Transmon Setup.PNG" alt="drawing" width="400"/>

## Experimental setup and context

These files showcase various experiments that can be done on Two flux tunable transmons with individual qubit drive lines 
and a single readout transmission line.
The readout pulses are sent through an IQ mixer and down-converted through an IQ mixer. 
Qubit addressing is being done with IQ mixers.

These files were tested in a real setup shown on the right, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

## Basic Files
0. [Hello QUA](00_hello_qua.py) - A script used for playing with QUA.
1. [Mixer Calibration](01_manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances.
2. [Raw ADC Traces](02_raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets and check the multiplexed readout levels.
3. [time_of_flight](03_time_of_flight.py) - A script to measure the ADC offsets and calibrate the time of flight.
4. [Resonator Spectroscopy](04_resonator_spectroscopy_single.py) - Performs a 1D frequency sweep on a given resonator.
5. [Multiplexed Resonator Spectroscopy](05_resonator_spectroscopy_multiplexed.py) - Performs a 1D frequency sweep on the two resonators simultaneously.
6. **2D resonator spectroscopy:**
    * [Resonator Spectroscopy vs readout power](06_resonator_spectroscopy_vs_amplitude.py) - Performs the resonator spectroscopy versus readout power to find the maximum desired readout amplitude.
7. [Qubit Spectroscopy](07_qubit_spectroscopy.py) - Performs a 1D frequency sweep on the qubits, measuring the resonator.
8. [Rabi Chevron](09_rabi_chevron.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
9. **1D Rabi** - Calibrate a $\pi$ pulse:
    * [Power Rabi](10_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse. Can also apply multiple pi pulses to better estimate the pi amplitude.
    * [Time Rabi](10_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
10. [IQ Blobs](11_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for active reset.
11. [Active Reset](12_IQ_blobs_active_reset.py) - Script for performing a single shot discrimination and active reset. ![care](https://img.shields.io/badge/to_be_tested_on_a_real_device-use_with_care-red)
12. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](13a_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](13b_readout_frequency_optimization.py) - The script performs amplitude scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Duration optimization](13c_readout_frequency_optimization.py) - The script performs duration scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Weights optimization](13d_readout_frequency_optimization.py) - The script returns to you the demodulated time traces of the |g> and |e> states. As a result you can find the optimal weights for discrimination.
13. [T1](14_T1.py) - Measures T1.
14. [Ramsey Chevron](15_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
15. [Ramsey with virtual Z rotations](16_Ramsey.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second pi/2 pulse to apply a virtual Z rotation.
16. [ALLXY](17_allxy.py) - Performs an ALLXY experiment to estimate gates imperfection
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
17. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Single Qubit Randomized Benchmarking](18a_single_qubit_RB.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout.
    * [Single Qubit Interleaved Randomized Benchmarking](18b_single_qubit_RB.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout.
18. **CR time rabi** - Performs cross-resonance gate while varying the cross-resonance pulse duration; includes single-qubit tomography on target qubit.
    * [CR_square_time_rabi](19a_CR_time_rabi_1q_QST.py) - CR gate with rectangular pulse of varying length.
    * [CR_echo_square_time_rabi](19b_CR_time_rabi_1q_QST.py) - echo CR gate with rectangular pulse of varying length.
    * [CR_echo_flat_top_gaussian_time_rabi](19c_echoCR_flattop_time_rabi_1q_QST.py) - echo CR gate with flat top pulse with gaussian rise and fall edges of vaying length.
19. **CR power rabi** - Performs cross-resonance gate while varying the cross-resonance pulse amplitude; includes single-qubit tomography on target qubit.
    * [CR_square_time_rabi](20a_CR_power_rabi_1q_QST.py) - CR gate with rectangular pulse of varying length.
    * [CR_echo_square_time_rabi](20b_CR_power_rabi_1q_QST.py) - echo CR gate with rectangular pulse of varying length.
    * [CR_echo_flat_top_gaussian_time_rabi](20c_echoCR_flattop_power_rabi_1q_QST.py) - echo CR gate with flat top pulse with gaussian rise and fall edges of vaying length.
20. **CR two-qubit state tomography** CR gate with two qubit state tomography to reconstruct the full density matrix.
    * [CR_echo_flat_top_gaussian_time_rabi_2q_QST](21_echoCR_flattop_time_rabi_2q_QST.py) - Prepares the control qubit in a superposition and then performs echo CR with a flat top pulse with gaussian rise and fall, and captures two-qubit state tomography that will be used to reconstruct the density matrix.

## Set-ups with Octave

The configuration included in this folder correspond to a set-up without Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_clock_and_calibration.py](octave_configuration.py): A file __to execute__ in order to set the clock and/or calibrate the Octave.
3. [set_octave.py](set_octave.py): A set of helper function to ease the octave parametrization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).

