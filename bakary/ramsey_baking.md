## Generating short Ramsey pulse sequences with waveform baking

This tutorial presents a use case for the waveform baking tool, which facilitates the generation of pulse samples that are shorter than 16 ns, which would usually have to be manually modified to upload it to the OPX.

Using the baking environment before launching the QUA program allows the pulse to be seamlessly integrated in the configuration file, without having to account for the restrictions of pulse length imposed by QUA.

It also provides a simpler interface to generate one single waveform that can contain several play statements (preserving program memory).

The experiment is as follows : 
We have a superconducting qubit (controlled using the quantum element 'Drive' in the configuration file) coupled to a readout resonator ('Resonator') with which we would like to apply sequences of two short Gaussian pulses spaced with a varying time duration, followed directly by a probe coming from the resonator (the measurement procedure should start immediately after the second Gaussian pulse was played by the Drive element).

The baking environment is used here to synthesize without any effort a waveform resulting from delayed superposition of two Gaussian pulses (a simple play followed by a play_at at a varying delay).
Note that we also use an initial delay for waiting time to ensure that there is a perfect synchronization between the end of the Ramsey sequence and the trigger of the resonator for probing the qubit state.

Within the QUA program, what remains to do is simply launching the created baking objects within a Python for loop and use all appropriate commands related to the resonator to build your experiment. Note that there is also another QUA for_ loop that iterates on another delay called *d*. This additional delay might also be used within the baking to add an extra delay between the two Gaussian pulses.

