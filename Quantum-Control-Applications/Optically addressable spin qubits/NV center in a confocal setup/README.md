# Single NV center in a confocal setup

## Basic Files
These files showcase various experiments that can be done on an NV center in a confocal setup with an SPCM and an AOM
which is controlled via a digital channel.
These files were tested in a real setup, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

0. [Hello QUA](00_hello_qua.py) - A script used for playing with QUA
1. [Mixer Calibration](01_manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances
2. [Raw ADC Traces](02_raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC is
not saturated and defining the threshold for time tagging
3. [Counter](03_counter.py) - Starts a counter which reports the current counts from the SPCM
4. [Calibrate Delays](04a_calibrate_delays.py) - Plays a MW pulse during a laser pulse, while performing time tagging 
throughout the sequence. This allows measuring all the delays in the system, as well as the NV initialization duration
    * [Calibrate Delays Python Histogram](04b_calibrate_delays_python_histogram.py) - This version processes the data in 
Python, which makes it slower but works better when the counts are high.
5. ODMR - Counts photons while sweeping the frequency of the applied MW
   * [CW ODMR](05a_cw_odmr.py) - Plays microwave and laser readout simultaneously
   * [pulsed ODMR](05b_pulsed_odmr.py) - Plays microwave pulse and laser readout consecutively
6. [Time Rabi](06_time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse
7. [Power Rabi](07_power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse
8. [Ramsey](08_ramsey.py) - Measures T2*
9. T2 measurement and dynamical decoupling
   * [Hahn Echo](09a_hahn_echo.py) - Measures T2
   * [XY8-N tau sweep](09b_xy8_tau.py) - An XY8-N dynamical decoupling measurement with fixed order N and sweeping the interpulse spacing
   * [XY8 order sweep](09c_xy8_order.py) - An XY8 dynamical decoupling measurement with fixed interpulse spacing and sweeping the order N
10. [T1](10_T1.py) - Measures T1. Can measure the decay from either |1> or |0>
11. [State TOmography](11_state_tomography.py) - Get the state of the qubit by measuring the three projections.
12. [Randomized Benchmarking](12_randomized_benchmarking.py) - Performs a single qubit randomized benchmarking to measure the single qubit gate fidelity for gates longer than 40ns.