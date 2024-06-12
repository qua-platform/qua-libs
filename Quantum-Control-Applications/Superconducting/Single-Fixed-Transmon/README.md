# Single Fixed Transmon Superconducting Qubit

<img align="right" src="Single Fixed Frequency Transmon Setup.PNG" alt="drawing" width="400"/>

## Basic Files
These files showcase various experiments that can be done on a Single Fixed Transmon Superconducting Qubit.
These files were tested on real qubits, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

0. [Hello QUA](./Single-Fixed-Transmon/00_hello_qua.py) - A script used for playing with QUA.
1. [Mixer Calibration](./Single-Fixed-Transmon/01_manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances.
2. [Raw ADC Traces](./Single-Fixed-Transmon/02_raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets.
3. [time_of_flight](./Single-Fixed-Transmon/03_time_of_flight.py) - A script to measure the ADC offsets and calibrate the time of flight.
4. [Resonator Spectroscopy](./Single-Fixed-Transmon/04_resonator_spectroscopy.py) - Performs a 1D frequency sweep on the resonator.
5. [Resonator Spectroscopy vs readout power](./Single-Fixed-Transmon/05_resonator_spectroscopy_vs_amplitude.py) - Performs the resonator spectroscopy versus readout power to find the desired readout amplitude and check if the qubit is coupled to the resonator.
6. [Qubit Spectroscopy](./Single-Fixed-Transmon/06a_qubit_spectroscopy.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator.
   * [Qubit Spectroscopy Wide Range](./Single-Fixed-Transmon/06b_qubit_spectroscopy_wide_range_outer_loop.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source in the outer loop.
   * [Qubit Spectroscopy Wide Range Inner Loop](./Single-Fixed-Transmon/06c_qubit_spectroscopy_wide_range_inner_loop.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source in the inner loop.
7. **Rabi chevrons** - Quickly find the qubit for a given pulse amplitude or duration:
    * [duration vs frequency](./Single-Fixed-Transmon/07a_rabi_chevron_duration.py) - Performs a 2D sweep (frequency vs qubit drive duration) to acquire the Rabi chevron.
    * [amplitude vs frequency](./Single-Fixed-Transmon/07b_rabi_chevron_amplitude.py) - Performs a 2D sweep (frequency vs qubit drive amplitude) to acquire the Rabi chevron.
8. **1D Rabi** - Precisely calibrate a $\pi$ pulse: 
    * [Time Rabi](./Single-Fixed-Transmon/08a_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse.
    * [Power Rabi](./Single-Fixed-Transmon/08b_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse.
    * [Power Rabi with error amplification](./Single-Fixed-Transmon/08c_power_rabi_error_amplification.py) - A multi-pulse Rabi experiment sweeping the amplitude of the MW pulses to better estimate the $\pi$ pulse amplitude.
9. [IQ Blobs](./Single-Fixed-Transmon/09a_IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for state discrimination and active reset.
    * [Resonator Depletion Time](./Single-Fixed-Transmon/09b_resonator_depletion_time.py) - Measure the resonator depletion time using a fixed time Ramsey sequence to know how long one needs to wait after measuring the resonator for active reset protocols.
    * [Active Reset](./Single-Fixed-Transmon/09c_active_reset.py) - Script for performing a single shot discrimination and active reset.
10. **Readout optimization** - The optimal separation between the |g> and |e> blobs lies in a phase spaced of amplitude, duration, and frequency of the readout pulse:
    * [Frequency optimization](./Single-Fixed-Transmon/10a_readout_frequency_optimization.py) - The script performs frequency scanning and from the results calculates the SNR between |g> and |e> blobs. As a result you can find the optimal frequency for discrimination.
    * [Amplitude optimization](./Single-Fixed-Transmon/10b_readout_amplitude_optimization.py) - The script measures the readout fidelity for different readout powers.
    * [Duration optimization](./Single-Fixed-Transmon/10c_readout_duration_optimization.py) - The script performs accumulated demodulation for a given frequency, amplitude, and total duration of readout pulse, and plots the SNR as a function of readout time.
    * [Integration Weights optimization](10d_readout_weights_optimization.py) - Performs sliced.demodulation to obtain the trajectories of the |e> and |g> states, and calculates the normalized optimal readout weights.
11. [T1](./Single-Fixed-Transmon/11_T1.py) - Measures T1.
13. [Ramsey Chevron](./Single-Fixed-Transmon/12_ramsey_chevron.py) - Perform a 2D sweep (detuning versus idle time) to acquire the Ramsey chevron pattern.
12. **1D Ramsey** - Measures T2*.
    * [Ramsey with virtual Z rotations](./Single-Fixed-Transmon/13a_ramsey_w_virtual_rotation.py) - Perform a Ramsey measurement by scanning the idle time and dephasing the second $\pi/2$ pulse to apply a virtual Z rotation.
    * [Ramsey with detuning](./Single-Fixed-Transmon/13b_ramsey_w_detuning.py) - Perform a Ramsey measurement by scanning the idle time with a given detuning.
14. [Echo](./Single-Fixed-Transmon/14_echo.py) - Measures T2 by apply an echo pulse.
15. [ALL XY](./Single-Fixed-Transmon/15_allxy.py) - Performs an ALL XY experiment to estimate gates imperfection.
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details).
16. **Single Qubit Randomized Benchmarking** - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 40ns](./Single-Fixed-Transmon/16b_randomized_benchmarking_interleaved.py) - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity  for gates longer than 40ns.
    * [Single Qubit Randomized Benchmarking for gates > 40ns](./Single-Fixed-Transmon/16a_randomized_benchmarking.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates longer than 40ns.
    * [Interleaved Single Qubit Randomized Benchmarking for gates > 20ns](./Single-Fixed-Transmon/16d_randomized_benchmarking_interleaved_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity for gates as short as 20ns (currently limited to a depth of 1000 Clifford gates).
    * [Single Qubit Randomized Benchmarking for gates > 20ns](./Single-Fixed-Transmon/16c_randomized_benchmarking_20ns.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity with or without single shot readout for gates as short as 20ns (currently limited to a depth of 2600 Clifford gates).
17. **DRAG calibration** - Calibrates the DRAG coefficient $`\alpha`$ and AC-Stark shift:
    * [Google method](./Single-Fixed-Transmon/17_DRAG_calibration_Google.py) - Performs `x180` and `-x180` pulses to obtain 
the DRAG coefficient $`\alpha`$.
    * [Yale method](./Single-Fixed-Transmon/17_DRAG_calibration_Yale.py) - Performs `x180y90` and `y180x90` pulses to obtain 
the DRAG coefficient $`\alpha`$.
    * [AC Stark shift](./Single-Fixed-Transmon/18_AC_Stark_calibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by plotting the 2D map DRAG pulse detuning versus number of iterations.
19. **Tomography:**
    * [State Tomography](./Single-Fixed-Transmon/19_state_tomography.py) - A template to perform state tomography of an arbitrary state.
20.  [Frequency Tracking](./Single-Fixed-Transmon/20_frequency_tracking.py) <span style="color:red">__to be tested on a real device, use with care__</span> - Script to implement tracking of the qubit frequency over time. More details about the usage and the method can be found in [Schuster Lab - Qubit Frequency Tracking](./Use%20Case%201%20-%20Schuster%20Lab%20-%20Qubit%20Frequency%20Tracking).
21.  [Advanced calibration prototype](./Single-Fixed-Transmon/advanced_calibration_prototype.py) <span style="color:red">_to be tested on a real device, use with care_</span> - Uses an API to perform several single qubit calibrations easily from a single file.

## Use Cases
These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Schuster Lab - Qubit Frequency Tracking](./Use%20Case%201%20-%20Schuster%20Lab%20-%20Qubit%20Frequency%20Tracking)
The goal of this measurement is to track the frequency fluctuations of the transmon qubit, and update the frequency of the qubit element accordingly using a closed-loop feedback.
* [Optimized Readout with Optimal Weights](./Use%20Case%202%20-%20Optimized%20readout%20with%20optimal%20weights) 
The goal of this experiment is to optimize the information obtained from the readout of a superconducting resonator by deriving the optimal integration weights. With the usage of optimal integration weights we maximize the separation of the IQ blobs when the ground and excited state are measured.


## Set-ups with Octave

The configuration included in this folder correspond to a set-up without an Octave. 
However, a few files are there to facilitate the integration of the Octave:
1. [configuration_with_octave.py](./Single-Fixed-Transmon/configuration_with_octave.py): An example of a configuration including the octave. You can replace the content of the file called `configuration.py` by this one so that it will be imported in all the scripts above.
2. [octave_clock_and_calibration.py](./Single-Fixed-Transmon/octave_clock_and_calibration.py): A file __to execute__ in order to configure the Octave's clock and calibrate the Octave.
3. [set_octave.py](./Single-Fixed-Transmon/set_octave.py): A helper function to ease the octave initialization.

If you are a new Octave user, then it is recommended to start with the [Octave tutorial](https://github.com/qua-platform/qua-libs/blob/main/Tutorials/intro-to-octave/README.md).
