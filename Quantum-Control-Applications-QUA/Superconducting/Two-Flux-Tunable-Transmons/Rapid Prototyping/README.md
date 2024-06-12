# Two flux tunable transmons with the rapid prototyping


## Experimental setup and context

These files showcase various experiments that can be done on Two flux tunable transmons with individual qubit drive lines 
and a single readout transmission line.
The readout pulses are sent through an IQ mixer and down-converted through an IQ mixer. 
Qubit addressing is being done with IQ mixers.

These files were tested in a real setup, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

## QuAM and Rapid Prototyping
The goal of the rapid prototyping is to give users the possibility to define their own structure (JSON file) to store 
the different parameters describing the state of the quantum system (LO and IF frequencies, wiring, flux points, 
pulse amplitudes and durations...).
Additionally, the QuAM SDK creates a python class out of the user-defined JSON structure that can then be used in the 
calibration scripts to set and get the qubit parameters.

The configuration dictionary that the OPX expects is then built using a one-to-one mapping between the config fields and 
the values from the user-defined structure as shown below.

A step-by-step [tutorial with best practices](https://github.com/qua-platform/qua-libs/tree/main/Tutorials/intro-to-quam-rapid-prototyping/README.md) details the basics of this framework and shows how to create your own QuAM (Quantum Abstract Machine).

**Note that two quam examples are given for two different set-ups**:
* 1 OPX+ connected to two flux tunable transmons with individuals qubit addressing and a common readout line: [QuAM for 2 qubits](./quam_for_2_qubits).
* A cluster of 2 OPX+ and 1 Octave connected to five flux tunable transmons with individuals qubit addressing and a common readout line: [QuAM for 5 qubits and octave](./quam_for_5_qubits_and_octave).

You can just copy and paste the relevant files that correspond to your set-up to the main folder with the calibration scripts.

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
    * [Resonator Spectroscopy vs flux](06_resonator_spectroscopy_vs_flux.py) - Performs the resonator spectroscopy versus flux to find the desired flux points.
7. [Qubit Spectroscopy](07_qubit_spectroscopy.py) - Performs a 1D frequency sweep on the qubits, measuring the resonator.
8.  * [Qubit Spectroscopy vs flux](08_qubit_spectroscopy_vs_flux.py) - Performs the qubit spectroscopy versus flux, measuring the resonators.
9. **Rabi chevrons** - Quickly find the qubit for a given pulse amplitude or duration:
    * [duration vs frequency](09_rabi_chevron_duration.py) - Performs a 2D sweep (frequency vs qubit drive duration) to acquire the Rabi chevron.
    * [amplitude vs frequency](09_rabi_chevron_amplitude.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
10. **1D Rabi** - Calibrate a $\pi$ pulse:
    * [Power Rabi](10_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse. Can also apply multiple pi pulses to better estimate the pi amplitude.
    * [Time Rabi](10_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
11. [IQ Blobs](11_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for active reset.
12. [Active Reset](12_IQ_blobs_active_reset.py) - Script for performing a single shot discrimination and active reset. ![care](https://img.shields.io/badge/to_be_tested_on_a_real_device-use_with_care-red)
13. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](13a_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](13b_readout_amplitude_optimization.py) - The script measures the readout fidelity for different readout powers.
    * [Duration optimization](13d_readout_duration_optimization.py) - The script performs accumulated demodulation for a given frequency, amplitude, and total duration of readout pulse, and plots the SNR as a function of readout time.
    * [Integration Weights optimization](13e_readout_weights_optimization.py) - Performs sliced.demodulation to obtain the trajectories of the |e> and |g> states, and calculates the normalized optimal readout weights.
14. [T1](14_T1.py) - Measures T1.
15. [Ramsey Chevron](15_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
16. [Ramsey with virtual Z rotations](16_ramsey.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second pi/2 pulse to apply a virtual Z rotation.
17. [Echo](17_echo.py) - Measures T2 by apply an echo pulse.
18. [ALLXY](18_allxy.py) - Performs an ALL XY experiment to estimate gates imperfection
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
19. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 40ns](19c_single_qubit_RB_interleaved.py) - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity  for gates longer than 40ns.
    * [Single Qubit Randomized Benchmarking for gates > 40ns](19a_single_qubit_RB.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates longer than 40ns.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 20ns](19d_single_qubit_RB_interleaved_20ns.py) - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity for gates as short as 20ns (currently limited to a depth of 1000 Clifford gates).
    * [Single Qubit Randomized Benchmarking for gates > 20ns](19b_single_qubit_RB_20ns.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates as short as 20ns (currently limited to a depth of 2600 Clifford gates).
20. **Cryoscope**: Cryoscope measurement to estimate the distortion on the flux lines based on [Appl. Phys. Lett. 116, 054001 (2020)](https://pubs.aip.org/aip/apl/article/116/5/054001/38884/Time-domain-characterization-and-correction-of-on)
    * [Cryoscope_amplitude_calibration](20_cryoscope_amplitude_calibration.py) - Performs the detuning vs flux pulse amplitude calibration prior to the cryoscope measurement. This gives the relation between the qubit detuning and flux pulse amplitude which should be quadratic.
    * [Cryoscope](20_cryoscope.py) - Performs the cryoscope measurement.
21. **SWAP spectroscopy** by driving the energy exchange |10> <--> |01>:
    * [iSWAP](21a_iSWAP.py) - Performs the iSWAP spectroscopy by scanning the OPX dc offset.
    * [iSWAP pulsed](21b_iSWAP_1ns.py) - Performs the iSWAP spectroscopy by scanning the flux pulse with 1ns resolution using the baking tool.
22. **CZ spectroscopy** by driving the energy exchange |11> <--> |02>:
    * [CZ](22a_CZ.py) - Performs the CZ spectroscopy by scanning the OPX dc offset.
    * [CZ pulsed](22b_CZ_1ns.py) - Performs the CZ spectroscopy by scanning the flux pulse with 1ns resolution using the baking tool.
