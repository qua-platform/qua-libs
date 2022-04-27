---
title: Coupling to left-handed metamaterial resonator
sidebar_label: Left-handed resonator
slug: ./
id: index
---

These scripts demonstrate usage of QUA to implement work published in [this paper](https://doi.org/10.1103/PhysRevApplied.14.064033).

The system under consideration is a flux tunable superconducting qubit that is coupled to a specially engineered readout resonator. 
The readout resonator is built such that the refractive index of one 
part of it is negative. This is called a Left Handed resonator or Left Handed Transmission Line (LHTL).
The interaction between different types of resonators creates a tightly
spaced resonator mode structure. 
One important feature of LHTL is that for a specific frequency range, 
we have a very particular dispersion relation regime : the wavelength associated to the resonator actually *increases* with the frequency.

The paper provides a full report on the characterization of this unique resonator design, 
followed by measurements demonstrating the strong interaction of the transmon qubit with the cavity modes. 
To show this, the vacuum Rabi splitting in reflection spectra is measured as the qubit is tuned across different modes is measured. 
In addition, the interaction population decay rate (T1), and the stark shift are measured as the qubit tuning is controlled showing the enhanced 
decay due to the purcell effect.  

### The configuration file

As stated above, the experimental system is a flux tunable qubit coupled to a two resonators. We therefore define 
three quantum elements for control: one for control of the qubit, 
one for the readout resonator, and one the flux line.

The flux line element has a single input and operates at DC. The other two 
are mixed input elements, connected to IQ mixers. 

In this use case, the elements control fixed amplitude pulses. 
However, advanced pulse shaping can easily be incorporated. 

### The scripts

We did not implement all experiments presented in the paper. Instead, we focus on a subset which demonstrates unique or noteworthy 
abilities afforded by QUA. 

- *transmission_spectrum.py* contains a QUA script allowing the realization of the plot done in Figure 1(i). This is a
  characterisation of the unique resonator spectrum. The script demonstrates the ability of QUA to interface with external 
  lab devices and control them. In this case, the frequency of an LO source is set between loops. For each LO frequency, 
  the IF of the control quantum element is looped over, and the reflected signal is demodulated and recorded. The handover
  between FPGA control and python control of the LO source is seamless and automatic. We implemented a mock LO element for 
  demonstration purpose (*mock_LO_source.py*). 
  This type of measurement would more commonly be performed with a VNA, however it can be useful 
  to perform this wide sweep with the same instrument with which other parts of the 
  experiments are performed.  
  
- *time_rabi.py* reproduces the time Rabi experiment conducted in Figure 3(c). This experiment conducted on many configurations in QUA (see for example digital_control_SC scripts).
 This is a particularly short script, showcasing the simplicity of running this type of experiment. In principle, we could estimate the state in real time and do something with that, 
  as is done in active-reset scenarios. However, in this case it doesn't make much sense to do so as this is a _calibration_ and a pi-pulse in not yet defined at this stage. 
  

 
