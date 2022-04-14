---
title: Digital coherent control for superconducting qubits
sidebar_label: Digital control
slug: ./
id: index
---

Scripts in this folder demonstrate usage of QUA to implement measurements published in the paper 
"Digital coherent control for superconducting qubits"  https://arxiv.org/pdf/1806.07930.pdf
The device measured in this paper is a superconducting qubit which is interfaced to a DC-SFQ element which 
generates a train of pulses, each with 2 picosecond duration. Each pulse produces a 
finite and discrete rotation of the qubit, effectively providing the means for digital control of the qubit. 
The paper used this technique to demostrate coherent control of a qubit and characterization of the fidelity with which 
such digital operations can be performed. 

### Introducing Single Flux Quantum (SFQ) control

It is currently an open question in the theory of quantum control for superconducting 
qubits whether analog (usual microwave control encountered in many implementations) or digital
control will allow the best performances regarding scalability and fault tolerance. 

The objective of the paper mentioned above is to demonstrate the ability of digitizing the control of a flux tunable transmon qubit 
by using what is called the SFQ digital logic family.

In this framework, "classical bits of information are encoded in fluxons, propagating voltage pulses whose time integral 
is precisely quantized to $$h/2e \equiv \Phi_0$$, the superconducting flux quantum".

With the help of a dc/SFQ converter, one can think about digitalizing the control of the 
qubit by performing incremental rotations of the qubit state around the Bloch sphere, translated physically by 
very short pulse trains (of the order of 2 ps). 

## The configuration file

In the configuration, we hence define quantum elements according to the layout of the experiment shown in Fig 9. 
We have then one element for the SFQ trigger (driver circuit), one other for the voltage source inducing the bias 
current to be applied (called *SFQ_bias*), and usual quantum elements for probing the readout resonator and apply 
usual analog microwave control on the qubit.
Multiple waveforms are defined and can be tuned to match the experimental characteristics of the setup.

## The QUA programs

The QUA scripts available are built to showcase how the experiments summarized by the plots done in Figure 3 
in the reference above can be realized with the OPX.

The first script, *bias_current_sweep.py*, is the investigation conducted by Fig 3.a 
to find the optimal bias current to be applied on the SFQ driver circuit to determine the period 
of Rabi oscillations induced by the SFQ control. 

The QUA program then consists in three QUA *for_* loops, the outer one iterating to 
repeat the number of repetitions for sampling (stream processing then performs a direct averaging 
operation to retrieve easily experiment statistics). The two other loops iterate over bias voltages 
(corresponding to various current biases) and the duration of the pulse applied by the SFQ driver circuit 
(which basically corresponds to a Time Rabi experiment, for each bias current chosen). 

The two other scripts *frequency_sweep.py* and *ramsey_exp.py*, do a similar job but replace the 
iteration over bias currents into a sweeping of drive frequency applied by the SFQ circuit 
(realized by the QUA command update frequency, which updates the intermediate frequency in the configuration file). 







