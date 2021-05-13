---
title: Hahn echo
sidebar_label: Hahn echo
slug: ./
id: index
---

This tutorial presents a practical use of QUA to run a well known experiment known as the Hahn echo sequence.
#  Description of experiment
## Introduction

" Electron spin resonance (ESR) and nuclear magnetic resonance (NMR) are used in diverse branches of science, ranging from spectroscopic studies in biochemistry and materials science to imaging of internal organs in medicine.
In NMR and ESR, an ensemble of spins is typically placed within a resonator, controlled by the application of resonant pulses, and measured via emission of signals into a resonator mode. As the spin ensembles are typically inhomogeneous, a common solution is to use control pulses which refocus inhomogeneous interactions, reversing the time evolution of different spin packets to produce a spin echo or ‘Hahn echo’. "

Source : Mølmer et al., Self-stimulated pulse echo trains from inhomogeneously broadened spin ensembles(https://arxiv.org/pdf/2004.01116.pdf)

## Hahn echo sequence 
![](https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/SpinEcho_GWM_stills.jpg/1280px-SpinEcho_GWM_stills.jpg)
Source: Gavin Morley, Wikipedia page "Spin echo" (https://en.wikipedia.org/wiki/Spin_echo)

The Hahn echo sequence can be summarized as follows (when working in the rotating frame of the qubit):
1. Apply a $$\pi/2$$-pulse
2. Wait a certain time duration (echo time)
3. Apply a $$\pi$$-pulse 
4. Wait again for echo, the relaxation time gives a good indication of the $$T_2$$ relaxation
5. Optional: Apply again a $$\pi/2$$-pulse to come back to the z-axis of the Bloch sphere for deduction of $$T_1$$
6. Readout of the qubit


## Describing the QUA program
### The configuration file

The configuration file is composed of the following items :

- controllers :
We define the outputs and inputs of the OPX device, which will be of use for the experiment. In this case, we have two analog outputs for the qubit, and two others for its coupled readout resonator. We add an analog input which is the channel where will be sampled out the analog results of the readout operation.
- elements :
This defines the set of essential components of the quantum system interacting with the OPX. In this case, the two elements are the qubit and the coupled readout resonator. 
Here are specified the main characteristics of the element, such as its resonant frequency, its associated operations (i.e the operations the OPX can apply on the element).
- pulses : 
A description of the doable pulses introduced in the elements. Here is provided a description of the default pulse duration (length parameter), the associated waveform (which can be taken from an arbitrary array), the type of operation (e.g control or measurement)
- waveforms : 
Specification of the pulse shape based on pre-built arrays (either by the user in case the shape is arbitrary, or constant pulse otherwise)
- Integration weights :
Describe the demodulation process of the data 


### The QUA program
The goal of the program is to find the echo time for which the qubit's relaxation process is complete. We then perform a for_ iteration over a set of possible times (within an interval named as t_vec) and perform the Hahn echo sequence. We then retrieve the results using measurement features of QUA (here embedded into the function readout_QUA) and use plot tools to determine the echo time.

```python
with for_(tau, 4, tau < taumax, tau + dtau):
        with for_(n, 0, n < NAVG, n + 1):
            Hadamard("qubit")
            wait(tau, "qubit")
            Rx(π, "qubit")
            wait(tau, "qubit")
            Hadamard("qubit")
            align("rr", "qubit")
            measure_and_save_state("rr")
            wait(recovery_delay // 4, "qubit")
```

## Script 

[download script](hahn_echo.py)



















 

