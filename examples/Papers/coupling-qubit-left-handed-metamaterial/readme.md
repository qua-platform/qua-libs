# Addressing left-handed metamaterial resonator with the OPX
These scripts demonstrate usage of QUA to implement work published in https://doi.org/10.1103/PhysRevApplied.14.064033

The system under consideration is a flux tuneable superconducting qubit that is coupled to a specially engineered readout resonator. 
The readout resonator is built such that the refractive index of one part of it is negative. This is called a Left Handed resonator or Left Handed Transmission Line (LHTL). The interaction between different types of resonators creates a tightly spaced resonator mode structure. One important feature of LHTL is that for a specific frequency range, we have a very particular dispersion relation regime : the wavelength associated to the resonator actually *increases* with the frequency.

The paper demonstrates how strong interaction between resonator modes and the qubit is achieved. 

### The configuration file
The setup presented is as follows : we have a flux tuneable superconducting qubit coupled to the left handed metamaterial readout resonator. 
We hence define three quantum elements in the configuration : one for addressing control on the qubit directly, one for the readout resonator, and one tuning up the flux bias to the fluxline connected to the qubit.

Available pulses are all of the constant types, but can easily be set to arbitrary shapes depending on the actual hardware configuration.

### The scripts
We introduce multiple QUA programs aiming to reproduce experiments conducted in the paper referenced above.
- *transmission_spectrum.py* contains a QUA script allowing the realization of the plot done in Figure 1(i), which consists in performing a wide frequency sweep for probing all the modes of the resonator. 
 In QUA, we have to sweep over the intermediate frequency, but the sweep range attainable is only of the order of few hundred MHz, whereas the range desired is of the order of few GHz. We hence have to change the LO frequency source interactively to reach all desired frequencies with IQ mixer. This change of LO frequency is done using the mock LO_source object. 
There is therefore an interaction between the OPX and the client PC which updates manually the LO frequency to run in QUA a full frequency sweep. This interaction is controlled via the use of pause() and job.resume() statements available respectively in QUA and in the Quantum Machine API.
- *mock_LO_source.py* defines an abstraction of what could be the LO source object, tha would normally be controllable using a driver in Python. Here this mock class just bears a frequency attribute that can be setup in Python.
- *time_rabi.py* simply reproduces the time Rabi experiment conducted in Figure 3(c). This experiment conducted on many configurations in QUA (see for example digital_control_SC scripts), remains easily done regardless of the configuration settings. 