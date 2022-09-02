# Single flux tunable transmon with SSB mixer for readout

<img align="right" src="Single Flux Tunable Transmon Setup.PNG" alt="drawing" width="400"/>

## Experimental setup and context

These files showcase various experiments that can be done on an single flux tunable transmon.
The readout pulses are sent through a SSB mixer but are downconverted through an IQ mixer. 
Qubit addressing is being done with and IQ mixer.

These files were tested in a real setup shown on the right, but are given as-is with no guarantee.

While these can serve as a template for new labs or for new experiments, certain adaptations will probably have to be made.
Use with care.

## Basic Files
1. [Hello_qua](hello_qua.py) - A script used for basic qua program demonstration.
2. [raw_adc_traces](raw_adc_traces.py) - A script for acquiring raw ADC traces from inputs 1 and 2 and check ADC saturation and time of flight.
3. [Resonator_spec](resonator_spec.py) - Performs the 1D and 2D (with flux amplitude sweep) resonator spectroscopy.
4. [Rabi_amp_freq](rabi_amp_freq.py) - Acquires the 2D (pulse amplitude & frequency sweeps) Rabi oscillations.
5. [ramsey_freq_duration](ramsey_freq_duration.py) - Acquires the 2D (idle time & pulse frequency sweeps) Ramsey oscillations.
6. [Resonator_spec_g_e](resonator_spec_g_e.py) -  Performs the 1D resonator spectroscopy for a ground and excited qubit (with IO values).
7. [IQ_blobs](IQ_blobs.py) - Performs a single shot discrimination and active reset.
8. [Tomography](tomography.py) - Performs the qubit tomography by scanning the phase of the 2nd pi/2 pulse.
9. [Cryoscope_amplitude_calibration](cryoscope_amplitude_calibration.py) - Performs the detuning vs flux pulse amplitude calibration prior to the cryoscope measurement. This gives the relation between the qubit detuning and flux pulse amplitude which should be quadratic.
10. [Cryoscope](cryoscope.py) - Performs the cryoscope measurement.

## Use Cases

These folders contain various examples of protocols made with the OPX, including the results. The scripts are tailored to
a specific setup and would require changes to run on different setups. Current use-cases:

* [Paraoanu Lab - Cryoscope](./Use%20Case%201%20-%20Paraoanu%20Lab%20-%20Cryoscope)
The goal of this use-case is to implement Cryoscope.
* [DRAG coefficient calibration](./Use%20Case%202%20-%20DRAG%20coefficient%20calibration) 
The goal of this experiment is to calibrate the DRAG coefficient and AC Start shift
to increase the single qubit gate fidelity as well as to minimize the leakage out of the
computational space.
