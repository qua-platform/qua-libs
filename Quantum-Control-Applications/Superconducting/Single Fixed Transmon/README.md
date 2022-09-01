# Single Fixed Transmon Superconducting Qubit

## Basic Files
These files showcase various experiments that can be done on a Single Fixed Transmon Superconducting Qubit.
These files were tested on real qubits, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

1. [Hello QUA](hello_qua.py) - A script used for playing with QUA
2. [Mixer Calibration](manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances
3. [Raw ADC Traces](raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC 
is not saturated, correct for DC offsets and define the time of flight
4. [Resonator Spectroscopy](resonator_spec.py) - Performs a 1D frequency sweep on the resonator
5. [Qubit Spectroscopy](qubit_spec.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator
   * [Qubit Spectroscopy Wide Range](qubit_spec_wide_range.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source in the outer loop
   * [Qubit Spectroscopy Wide Range Inner Loop](qubit_spec_wide_range_inner_loop.py) - Performs a 1D frequency sweep on the qubit, measuring the resonator while also sweeping an external LO source in the inner loop
6. [Time Rabi](time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse
7. [Power Rabi](power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse
8. [IQ Blobs](IQ_blobs.py) - Measure the qubit in the ground and excited state to create the IQ blobs. If the separation
and the fidelity are good enough, gives the parameters needed for active reset
9. [Ramsey](ramsey.py) - Measures T2*
10. [Echo](echo.py) - Measures T2
11. [T1](T1.py) - Measures T1
12. [ALLXY](allxy.py) - Performs an ALLXY experiment to correct for gates imperfections
(see [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details)
13. **DRAG calibration** -  Calibrates the DRAG coefficient `$\alpha$` and AC-Stark shift:
    * [Google method](DRAG_calibration_Google.py) - Performs `x180` and `-x180` pulses to obtain 
the DRAG coefficient `$\alpha$`
    * [Yale method](DRAG_calibration_Yale.py) - Performs `x180y90` and `y180x90` pulses to obtain 
the DRAG coefficient `$\alpha$`
    * [2D](AC_Stark_2Dcalibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by plotting the 2D map DRAG pulse detuning versus number of iterations.
    * [1D](AC_Stark_1Dcalibration_Google.py) - Calibrates the AC Stark shift using a sequence of `x180` and `-x180` pulses by scanning the DRAG pulse detuning for a given number of pulses.

15. [Single Qubit Randomized Benchmarking](rb.py) - Performs a 1 qubit randomized benchmarking to measure the 1 qubit gate
fidelity
    * [Interleaved Randomized Benchmarking](interleaved_rb.py) - Performs a single qubit interleaved randomized benchmarking to measure a specific single qubit gate fidelity.
    * [Randomized Benchmarking without Single Shot readout](rb_without_singleshot_readout.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity without single shot readout.
16. [State Tomography](state_tomography.py) - A template to perform state tomography
17. [Calibration](calibrations.py) - Uses an API to perform several single qubit calibrations easily from a single file. 

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Schuster Lab - Qubit Frequency Tracking](./Use%20Case%201%20-%20Schuster%20Lab%20-%20Qubit%20Frequency%20Tracking)
The goal of this measurement is to track the frequency fluctuations of the transmon qubit, and update the frequency of the qubit element accordingly using a closed-loop feedback.
* [Optimized Readout with Optimal Weights](./Use%20Case%202%20-%20Optimized%20readout%20with%20optimal%20weights) 
The goal of this experiment is to optimize the information obtained from the readout of a superconducting resonator by deriving the optimal integration weights. With the usage of optimal integration weights we maximize the separation of the IQ blobs when the ground and excited state are measured.
