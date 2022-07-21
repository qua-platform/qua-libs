# Single NV center in a confocal setup

## Basic Files
These files showcase various experiments that can be done on an NV center in a confocal setup with an SPCM and an AOM
which is controlled via a digital channel.
These files were tested in a real setup, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

1. [Hello QUA](hello_qua.py) - A script used for playing with QUA
2. [Mixer Calibration](manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances
3. [Raw ADC Traces](raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC is
not saturated and defining the threshold for time tagging
4. [Counter](counter.py) - Starts a counter which reports the current counts from the SPCM
5. [CW ODMR](cw_odmr.py) - Counts photons while sweeping the frequency of the applied MW
6. [Calibrate Delays](calibrate_delays.py) - Plays a MW pulse during a laser pulse, while performing time tagging 
throughout the sequence. This allows measuring all the delays in the system, as well as the NV initialization duration
    * [Calibrate Delays Python Histogram](calibrate_delays_python_histogram.py) - This version process the data in 
Python, which makes it slower but works better when the counts are high.
7. [Time Rabi](time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse
8. [Power Rabi](power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse
9. [Ramsey](ramsey.py) - Measures T2*
10. [Hahn Echo](hahn_echo.py) - Measures T2
11. [T1](T1.py) - Measures T1. Can measure the decay from either |1> or |0>