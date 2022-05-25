# Single NV center in a confocal setup

## Basic Files
These files showcase various experiments that can be done on an NV center in a confocal setup with an SPCM and an AOM
which is controlled via a digital channel.
These files were tested in a real setup, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

1. [Time Tagging Calibrations](time_tagging_calibrations.py) - A script used to look at the raw ADC data, this allows 
checking that the ADC is not saturated and defining the threshold for time tagging.
2. [Counter](counter.py) - Starts a counter which reports the current counts from the SPCM
3. [CW ODMR](cw_odmr.py) - Counts photons while sweeping the frequency of the applied MW
4. [Calibrate Delays](calibrate_delays.py) - Plays a MW pulse during a laser pulse, while performing time tagging 
throughout the sequence. This allows measuring all the delays in the system, as well as the NV initialization duration.
5. [Time Rabi](time_rabi.py) - A Rabi experiment sweeping the duration of the MW pulse 
6. [Power Rabi](power_rabi.py) - A Rabi experiment sweeping the amplitude of the MW pulse
7. [Ramsey](ramsey.py) - Measures T2*
8. [Hahn Echo](hahn_echo.py) - Measures T2
9. [T1](T1.py) - Measures T1. Can measure the decay from either |1> or |0>