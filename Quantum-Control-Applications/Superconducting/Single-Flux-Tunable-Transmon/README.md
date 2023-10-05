# Single flux tunable transmon

<img align="right" src="Single Flux Tunable Transmon Setup.PNG" alt="drawing" width="400"/>

## Experimental setup and context

These files showcase various experiments that can be done on a single flux tunable transmon.
The readout pulses are sent through an IQ mixer and down-converted through an IQ mixer. 
Qubit addressing is being done with and IQ mixer.

These files were tested in a real setup shown on the right, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

## Basic Files
0. [Hello QUA](./Single-Flux-Tunable-Transmon/00_hello_qua.py) - A script used for playing with QUA.
1. [Mixer Calibration](./Single-Flux-Tunable-Transmon/01_manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances.
2. [Raw ADC Traces](./Single-Flux-Tunable-Transmon/02_raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets.
3. [time_of_flight](./Single-Flux-Tunable-Transmon/03_time_of_flight.py) - A script to measure the ADC offsets and calibrate the time of flight.
4. [Resonator Spectroscopy](./Single-Flux-Tunable-Transmon/04_resonator_spectroscopy.py) - Performs a 1D frequency sweep on the resonator.
5. **2D resonator spectroscopy:**
    * [Resonator Spectroscopy vs readout power](./Single-Flux-Tunable-Transmon/05a_resonator_spectroscopy_vs_amplitude.py) - Performs the resonator spectroscopy versus readout power to find the desired readout amplitude and check if the qubit is coupled to the resonator.
    * [Resonator Spectroscopy vs flux](./Single-Flux-Tunable-Transmon/05b_resonator_spectroscopy_vs_flux.py) - Performs the resonator spectroscopy versus flux.
6. **Qubit spectroscopy**
    * [Qubit Spectroscopy](./Single-Flux-Tunable-Transmon/06a_qubit_spectroscopy.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator.
    * [Qubit Spectroscopy vs flux](./Single-Flux-Tunable-Transmon/06b_qubit_spectroscopy_vs_flux.py) - Performs the qubit spectroscopy versus flux, measuring the resonator.
    * [Qubit Spectroscopy Wide Range](./Single-Flux-Tunable-Transmon/06c_qubit_spectroscopy_wide_range_outer_loop.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source in the outer loop.
7. **Rabi chevrons** - Quickly find the qubit for a given pulse amplitude or duration:
    * [duration vs frequency](./Single-Flux-Tunable-Transmon/07a_rabi_chevron_duration.py) - Performs a 2D sweep (frequency vs qubit drive duration) to acquire the Rabi chevron.
    * [amplitude vs frequency](./Single-Flux-Tunable-Transmon/07b_rabi_chevron_amplitude.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
8. **1D Rabi** - Precisely calibrate a $\pi$ pulse: 
    * [Time Rabi](./Single-Flux-Tunable-Transmon/08a_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
    * [Power Rabi](./Single-Flux-Tunable-Transmon/08b_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse.
    * [Power Rabi with error amplification](./Single-Flux-Tunable-Transmon/08c_power_rabi_error_amplification.py) - A multi-pulse Rabi experiment sweeping the amplitude of the MW pulses to better estimate the $\pi$ pulse amplitude.
9. [IQ Blobs](./Single-Flux-Tunable-Transmon/09a_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for state discrimination and active reset.
    * [Resonator Depletion Time](./Single-Flux-Tunable-Transmon/09b_resonator_depletion_time.py) - Measure the resonator depletion time using a fixed time Ramsey sequence to know how long one needs to wait after measuring the resonator for active reset protocols.
    * [Active Reset](./Single-Flux-Tunable-Transmon/09c_active_reset.py) - Script for performing a single shot discrimination and active reset.
10. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](./Single-Flux-Tunable-Transmon/10a_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](./Single-Flux-Tunable-Transmon/10b_readout_amplitude_optimization.py) - The script measures the readout fidelity for different readout powers.
    * [Duration optimization](./Single-Flux-Tunable-Transmon/10c_readout_duration_optimization.py) - The script performs accumulated demodulation for a given frequency, amplitude, and total duration of readout pulse, and plots the SNR as a function of readout time.
    * [Integration Weights optimization](10d_readout_weights_optimization.py) - Performs sliced.demodulation to obtain the trajectories of the |e> and |g> states, and calculates the normalized optimal readout weights.
11. [T1](./Single-Flux-Tunable-Transmon/11_T1.py) - Measures T1.
13. [Ramsey Chevron](./Single-Flux-Tunable-Transmon/12_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
12. **1D Ramsey** - Measures T2*.
    * [Ramsey with virtual Z rotations](./Single-Flux-Tunable-Transmon/13a_ramsey_w_virtual_rotation.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second $\pi/2$ pulse to apply a virtual Z rotation.
    * [Ramsey with detuning](./Single-Flux-Tunable-Transmon/13b_ramsey_w_detuning.py) - Perform a Ramsey measurement by scanning the idle time with a given detuning.
14. [Echo](./Single-Flux-Tunable-Transmon/14_echo.py) - Measures T2 by apply an echo pulse.
15. [ALL XY](./Single-Flux-Tunable-Transmon/15_allxy.py) - Performs an ALL XY experiment to estimate gates imperfection.
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
16. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 40ns](./Single-Flux-Tunable-Transmon/16b_randomized_benchmarking_interleaved.py) - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity  for gates longer than 40ns.
    * [Single Qubit Randomized Benchmarking for gates > 40ns](./Single-Flux-Tunable-Transmon/16a_randomized_benchmarking.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates longer than 40ns.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 20ns](./Single-Flux-Tunable-Transmon/16d_randomized_benchmarking_interleaved_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity for gates as short as 20ns (currently limited to a depth of 1000 Clifford gates).
    * [Single Qubit Randomized Benchmarking for gates > 20ns](./Single-Flux-Tunable-Transmon/16c_randomized_benchmarking_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates as short as 20ns (currently limited to a depth of 2600 Clifford gates).
17. **Cryoscope**: Cryoscope measurement to estimate the distortion on the flux lines based on [Appl. Phys. Lett. 116, 054001 (2020)](https://pubs.aip.org/aip/apl/article/116/5/054001/38884/Time-domain-characterization-and-correction-of-on) 
    * [Cryoscope_amplitude_calibration](./Single-Flux-Tunable-Transmon/17_cryoscope_amplitude_calibration.py) - Performs the detuning vs flux pulse amplitude calibration prior to the cryoscope measurement. This gives the relation between the qubit detuning and flux pulse amplitude which should be quadratic.
    * [Cryoscope with 1ns resolution](./Single-Flux-Tunable-Transmon/17_cryoscope_1ns.py) - Performs the cryoscope measurement with 1ns resolution using the baking tool, but limited to 260ns flux pulses.
    * [Cryoscope with 4ns resolution](./Single-Flux-Tunable-Transmon/17_cryoscope_4ns.py) - Performs the cryoscope measurement with 4ns granularity but no limitation of the flux pulse duration.
18. **DRAG calibration** - Calibrates the DRAG coefficient $`\alpha`$ and AC-Stark shift:
    * [Google method](./Single-Flux-Tunable-Transmon/18_DRAG_calibration_Google.py) - Performs `x180` and `-x180` pulses to obtain 
the DRAG coefficient $`\alpha`$.
    * [Yale method](./Single-Flux-Tunable-Transmon/18_DRAG_calibration_Yale.py) - Performs `x180y90` and `y180x90` pulses to obtain 
the DRAG coefficient $`\alpha`$.
    * [AC Stark-shift calibration](./Single-Flux-Tunable-Transmon/19_AC_Stark_calibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by plotting the 2D map DRAG pulse detuning versus number of iterations.
20. **Tomography:**
    * [State Tomography](./Single-Flux-Tunable-Transmon/20_state_tomography.py) - A template to perform state tomography.
21.  [Advanced calibration prototype](./Single-Flux-Tunable-Transmon/advanced_calibration_prototype.py) <span style="color:red">_to be tested on a real device, use with care_</span> - Uses an API to perform several single qubit calibrations easily from a single file.

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Ma Lab - Parametric Drive between flux-tunable-qubit and qubit-coupler](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Single-Flux-Tunable-Transmon/Use%20Case%203%20-%20Ma%20Lab%20-%20Parametric%20Drive%20iSWAP#parametric-drive-between-flux-tunable-qubit-and-qubit-coupler) 
  In this use-case, the parametric drive is demonstrated through the red- and blue-sideband transitions between a superconducting resonator and a flux-tunable transmon.
* [Paraoanu Lab - Cryoscope](./Use%20Case%201%20-%20Paraoanu%20Lab%20-%20Cryoscope)
The goal of this use-case is to implement Cryoscope.
* [DRAG coefficient calibration](./Use%20Case%202%20-%20DRAG%20coefficient%20calibration) 
The goal of this experiment is to calibrate the DRAG coefficient and AC Start shift
to increase the single qubit gate fidelity as well as to minimize the leakage out of the
computational space.

## Set-ups with Octave

The configuration included in this folder correspond to a set-up without Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](./Single-Flux-Tunable-Transmon/configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_clock_and_calibration.py](./Single-Flux-Tunable-Transmon/octave_clock_and_calibration.py): A file __to execute__ in order to configure the Octave's clock and calibrate the Octave.
3. [set_octave.py](./Single-Flux-Tunable-Transmon/set_octave.py): A helper function to ease the octave initialization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).
