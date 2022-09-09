# Electron Spin Resonance (ESR) experiments

## Basic Files
These ESR protocols can be used in a variety of ensemble of defects in solids
such as NV.

These files showcase various ESR experiments that can be done on an ensemble.
These files were tested in a real setup, but are given as-is with no guarantee.
While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

1. [Hello QUA](hello_qua.py) - A script used for playing with QUA
2. [Mixer Calibration](mixer_cal.py) - A script used to calibrate the corrections for mixer imbalances
3. [Input calibration](input_calibration.py) - A script to measure the analog signal when no drive is applied. Allows you to correct for offsets
4. [Signal test](signal_test.py) - A script that mimics a `pi/2 - pi` pulse sequence but with arbitrary pulse duration. Helps you check if signal is being generated from your setup
5. [Pi pulse calibration](pi_pulse_calibration.py) - A script that changes the duration of the pulses send to the ensemble to determine which pulse duration maximizes the echo amplitude
6. [Time Rabi](time_rabi.py) - Having calibrated roughly a `pi` pulse this script allows you fix the `pi` pulse duration and change the duration of the first pulse to obtain Rabi oscillations 
throughout the sequence. This allows measuring all the delays in the system, as well as the NV initialization duration
7. [T1](T1.py) - Measures T1 either from |0> or |1> to the thermal state, i.e., prior to initialization
8. [T2](T2.py) - A script that measures T2 after initialization of the ensemble
9. [CPMG](cpmg.py) - A script that measures the echo amplitude for a wide range of delays between `pi` pulses in a CPMG pulse sequence

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. 

Current use-cases:

* [Sekhar Lab - CPMG](./Use%20case%201%20-%20Sekhar%20Lab%20-%20CPMG)
The goal of this measurement is to obtain the decoherence time of an ensemble of NV centers with a CPMG-based dynamical decoupling
sequence.
