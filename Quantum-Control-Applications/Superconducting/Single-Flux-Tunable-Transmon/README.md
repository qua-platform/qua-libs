# Single flux tunable transmon

<img align="right" src="Single Flux Tunable Transmon Setup.PNG" alt="drawing" width="400"/>

## Experimental setup and context

These files showcase various experiments that can be done on an single flux tunable transmon.
The readout pulses are sent through an IQ mixer and down-converted through an IQ mixer. 
Qubit addressing is being done with and IQ mixer.

These files were tested in a real setup shown on the right, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

## Basic Files
0. [Hello QUA](00_hello_qua.py) - A script used for playing with QUA.
1. [Mixer Calibration](01_manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances.
2. [Raw ADC Traces](02_raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets.
3. [time_of_flight](03_time_of_flight.py) - A script to measure the ADC offsets and calibrate the time of flight.
4. [Resonator Spectroscopy](04_resonator_spec.py) - Performs a 1D frequency sweep on the resonator.
5. **2D resonator spectroscopy:**
    * [Resonator Spectroscopy vs readout power](05_resonator_spec_vs_amplitude.py) - Performs the resonator spectroscopy versus readout power to find the maximum desired readout amplitude.
    * [Resonator Spectroscopy vs flux](05_resonator_spec_vs_flux.py) - Performs the resonator spectroscopy versus flux to find the desired flux points.
6. **Qubit spectroscopy**
    * [Qubit Spectroscopy](06_qubit_spec.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator.
    * [Qubit Spectroscopy vs flux](06_qubit_spec_vs_flux.py) - Performs the qubit spectroscopy versus flux, measuring the resonator.
7. [Rabi Chevron](07_rabi_chevron.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
8. **1D Rabi** - Calibrate a $\pi$ pulse:
    * [Power Rabi](08_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse.
    * [Power Rabi with error amplification](08_power_rabi_error_amplification.py) - A multi-pulse Rabi experiment sweeping the amplitude of the MW pulses to better estimate the pi pulse amplitude.
    * [Time Rabi](08_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
9. [IQ Blobs](09_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for active reset.
    * [Resonator Emptying Time](09_resonator_emptying_time.py) - Measure the resonator depletion time using a fixed time Ramsey sequence to know how long one needs to wait after measuring the resonator for active reset protocols.
    * [Active Reset](09_active_reset.py) - Script for performing a single shot discrimination and active reset.
10. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](10_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](10_readout_amp_optimization.py) - The script measures the readout fidelity for different readout powers.
    * [Duration optimization](10_readout_duration_optimization.py) - The script performs accumulated demodulation for a given frequency, amplitude, and total duration of readout pulse, and plots the SNR as as a function of readout time.
    * [Integration Weights optimization](10_readout_weight_optimization.py) -Performs sliced.demodulation to obtain the trajectories of the |e> and |g> states, and from them it calculates the normalized optimal readout weights.
11. [T1](11_T1.py) - Measures T1.
13. [Ramsey Chevron](12_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
12. **1D Ramsey** - Measures T2*.
    * [Ramsey with detuning](13_ramsey_w_detuning.py) - Perform a Ramsey measurement by scanning the idle time with a given detuning.
    * [Ramsey with virtual Z rotations](13_ramsey_w_virtual_rotation.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second pi/2 pulse to apply a virtual Z rotation.
14. [Echo](14_echo.py) - Measures T2 by apply an echo pulse.
15. [ALLXY](15_allxy.py) - Performs an ALLXY experiment to estimate gates imperfection
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
16. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 40ns](16_randomized_benchmarking_interleaved.py) <span style="color:red">_to be tested on a real device, use with care_</span> - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity  for gates longer than 40ns.
    * [Single Qubit Randomized Benchmarking for gates > 40ns](16_randomized_benchmarking.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates longer than 40ns.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 20ns](16_randomized_benchmarking_interleaved_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity for gates as short as 20ns (currently limited to a depth of 1000 Cliffords).
    * [Single Qubit Randomized Benchmarking for gates > 20ns](16_randomized_benchmarking_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates as short as 20ns (currently limited to a depth of 2600 Cliffords).
17. **Cryoscope**: Cryoscope measurement to estimate the distortion on the flux lines based on [Appl. Phys. Lett. 116, 054001 (2020)](https://pubs.aip.org/aip/apl/article/116/5/054001/38884/Time-domain-characterization-and-correction-of-on) 
    * [Cryoscope_amplitude_calibration](17_cryoscope_amplitude_calibration.py) - Performs the detuning vs flux pulse amplitude calibration prior to the cryoscope measurement. This gives the relation between the qubit detuning and flux pulse amplitude which should be quadratic.
    * [Cryoscope](17_cryoscope.py) - Performs the cryoscope measurement.
18. **DRAG calibration** - Calibrates the DRAG coefficient `$\alpha$` and AC-Stark shift:
    * [Google method](18_DRAG_calibration_Google.py) - Performs `x180` and `-x180` pulses to obtain 
the DRAG coefficient `$\alpha$`.
    * [Yale method](18_DRAG_calibration_Yale.py) - Performs `x180y90` and `y180x90` pulses to obtain 
the DRAG coefficient `$\alpha$`.
19. **DRAG calibration** - Calibrates the AC-Stark shift:
    * [2D](19_AC_Stark_2Dcalibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by plotting the 2D map DRAG pulse detuning versus number of iterations.
    * [1D](19_AC_Stark_1Dcalibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by scanning the DRAG pulse detuning for a given number of pulses.
20. **Tomography:**
    * [State Tomography](20_state_tomography.py) - A template to perform state tomography.
20. [Calibration](calibrations.py) <span style="color:red">_to be tested on a real device, use with care_</span> - Uses an API to perform several single qubit calibrations easily from a single file.

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Paraoanu Lab - Cryoscope](./Use%20Case%201%20-%20Paraoanu%20Lab%20-%20Cryoscope)
The goal of this use-case is to implement Cryoscope.
* [DRAG coefficient calibration](./Use%20Case%202%20-%20DRAG%20coefficient%20calibration) 
The goal of this experiment is to calibrate the DRAG coefficient and AC Start shift
to increase the single qubit gate fidelity as well as to minimize the leakage out of the
computational space.

## Set-ups with Octave

The configuration included in this folder correspond to a set-up without Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_configuration.py](octave_configuration.py): A file __to execute__ in order to configure and/or calibrate the Octave.
3. [set_octave.py](set_octave.py): A set of helper function to ease the octave parametrization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).
