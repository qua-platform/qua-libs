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
    * [Resonator Spectroscopy vs flux](06_resonator_spectroscopy_vs_flux.py) - Performs the resonator spectroscopy versus flux to find the desired flux points.
7. [Qubit Spectroscopy](07_qubit_spectroscopy.py) - Performs a 1D frequency sweep on the qubits, measuring the resonator.
8. **Qubit spectroscopy versus flux:**
    * [Qubit Spectroscopy vs flux](08_qubit_spectroscopy_vs_flux.py) - Performs the qubit spectroscopy versus individual flux, measuring the resonator.
    * [Qubit Spectroscopy vs flux simultaneous](08_qubit_spectroscopy_vs_flux_simultaneous.py) - Performs the qubit spectroscopy while sweeping the two flux levels, measuring the resonator.
9. [Rabi Chevron](09_rabi_chevron.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
10. **1D Rabi** - Calibrate a $\pi$ pulse:
    * [Power Rabi](10_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse. Can also apply multiple pi pulses to better estimate the pi amplitude.
    * [Time Rabi](10_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
11. [IQ Blobs](11_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for active reset.
12. [Active Reset](12_IQ_blobs_active_reset.py) - Script for performing a single shot discrimination and active reset. ![care](https://img.shields.io/badge/to_be_tested_on_a_real_device-use_with_care-red)
13. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](13a_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](13b_readout_amp_optimization.py) - The script measures the readout fidelity for different readout powers.
    * [Duration optimization](13c_readout_duration_optimization.py) - The script performs accumulated demodulation for a given frequency, amplitude, and total duration of readout pulse, and plots the SNR as a function of readout time.
    * [Integration Weights optimization](13d_readout_weight_optimization.py) -Performs sliced.demodulation to obtain the trajectories of the |e> and |g> states, and calculates the normalized optimal readout weights.
14. [T1](14_T1.py) - Measures T1.
15. [Ramsey Chevron](15_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
16. [Ramsey with virtual Z rotations](16_Ramsey.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second pi/2 pulse to apply a virtual Z rotation.
17. [ALL XY](17_allxy.py) - Performs an ALL XY experiment to estimate gates imperfection
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
18. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Single Qubit Randomized Benchmarking](18_single_qubit_RB.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout.
19. **Cryoscope**: Cryoscope measurement to estimate the distortion on the flux lines based on [Appl. Phys. Lett. 116, 054001 (2020)](https://pubs.aip.org/aip/apl/article/116/5/054001/38884/Time-domain-characterization-and-correction-of-on)
    * [Cryoscope with 1ns resolution](19_cryoscope_1ns.py) - Performs the cryoscope measurement with 1ns resolution using the baking tool, but limited to 260ns flux pulses.
    * [Cryoscope with 4ns resolution](19_cryoscope_4ns.py) - Performs the cryoscope measurement with 4ns granularity but no limitation of the flux pulse duration. ![care](https://img.shields.io/badge/to_be_tested_on_a_real_device-use_with_care-red)
20. ** SWAP spectroscopy ** by driving the energy exchange |10> <--> |01>:
    * [iSWAP](20_iSWAP.py) - Performs the iSWAP spectroscopy by scanning the flux pulse with a 4ns granularity.
    * [iSWAP pulsed](20_iSWAP_1ns.py) - Performs the iSWAP spectroscopy by scanning the flux pulse with 1ns resolution using the baking tool.



21. ** CZ spectroscopy ** by driving the energy exchange |11> <--> |20>: ![care](https://img.shields.io/badge/to_be_tested_on_a_real_device-use_with_care-red)
    * [CZ](21_CZ.py) - Performs the CZ spectroscopy by scanning the flux pulse with a 4ns granularity.
    * [CZ pulsed](21_CZ_1ns.py) - Performs the CZ spectroscopy by scanning the flux pulse with 1ns resolution using the baking tool.
    

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Two qubit gate optimization with cryoscope](../Use%20Case%201%20-%20Two%20qubit%20gate%20optimization%20with%20cryoscope)
The goal of this use-case is to perform SWAP spectroscopy and improve the SWAP fidelity by correcting for the flux pulse 
distortion using Cryoscope and the OPX IIR and FIR filters..
* [Two-qubit randomized benchmarking](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Use%20Case%202%20-%20Two-Qubit-Randomized-Benchmarking#two-qubit-randomized-benchmarking).
This use-case showcases an implementation of two-qubit randomized benchmarking with the OPX.

## Set-ups with Octave

The configuration included in this folder correspond to a set-up without Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_clock_and_calibration.py](octave_clock_and_calibration.py): A file __to execute__ in order to configure the Octave's clock and calibrate the Octave.
3. [set_octave.py](set_octave.py): A helper function to ease the octave initialization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).

