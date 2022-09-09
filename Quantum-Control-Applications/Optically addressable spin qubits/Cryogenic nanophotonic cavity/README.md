# Single Yb center in a cyrogenic nanophotonic cavity

## Basic Files
These files showcase various experiments that can be done on a Yb center in cryogenic nanophotonic cavity with a SNSPD 
and an AOM which is controlled via a digital channel.

These files were tested in a real setup, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

The configuration and QUA programs were inspired by the following papers J. M. Kindem *et al.* [[1]](#1) and A. Ruskuc *et al.* [[2]](#2).

1. [Hello QUA](hello_qua.py) - A script used for playing with QUA
2. [Mixer Calibration](manual_mixer_calibration.py) - A script used to calibrate the corrections for mixer imbalances
3. [Raw ADC Traces](raw_adc_traces.py) - A script used to look at the raw ADC data, this allows checking that the ADC is
not saturated and defining the threshold for time tagging
4. [Counter](counter.py) - Starts a counter which reports the current counts from the SNSPD
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
12. [Randomized Benchmarking](rb.py) - Performs single qubit gates randomized benchmarking

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Faraon Lab - Sub-Nanosecond Time-Tagging](./Use%20case%201%20-%20Faraon%20Lab%20-%20sub-ns%20timetagging)
The goal of this measurement is to resolve the amplitude modulation of a laser light oscillating at ~ `535 MHz`
by time tagging photons hitting a superconducting nanowire single photon detector (SNSPD).

## References

<a id="1">[1]</a> https://www.nature.com/articles/s41586-020-2160-9

<a id="2">[2]</a> https://www.nature.com/articles/s41586-021-04293-6