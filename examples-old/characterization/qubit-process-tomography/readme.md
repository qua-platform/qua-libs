---
id: index
title: Process tomography
sidebar_label: Process tomography
slug: ./
---

# Reconstructing a single qubit process using samples from a QUA program
This program showcases the possibility to use a QUA program and the OPX to perform a process tomography experiment for a single qubit.

# Introduction
"Quantum operations provide a wonderful mathematical model for open quantum systems, and are conveniently visualized (at least for qubits) – but how do they relate to experimentally measurable quantities? What measurements should an experimentalist do if they wish to characterize the dynamics of a quantum system? For classical systems, this elementary task is known as system identification. Here, we show how its analogue, known as quantum process tomography, can be performed for a single qubit."

Source : Michael Nielsen & Isaac Chuang, "Quantum Computation and Quantum Information", 10th Anniversary edition (2010)

# 1. Theoretical reminder
A reminder on quantum state tomography can be found in another tutorial (this review is also of great use : https://arxiv.org/abs/1407.4759). 
In this tutorial, we review how state tomography can be used to perform generic qubit state tomography. The reference for the theory behind this tutorial, centered for a single qubit is generalized in Chapter 8.4.2 of the source above (especially Box 8.5 for qubit treatment, implemented in this script).

A quantum process is characterized by a superoperator (also trace preserving complete positive map, TPCP) $$\varepsilon$$ which maps a density matrix to another density matrix. For a single qubit, the density matrix consists in a 2 $$\times$$ 2 matrix which can be written in the traditional computational basis as :
$$\rho=\alpha |0\rangle\langle 0|+\beta |0\rangle\langle 1|+\gamma |1\rangle\langle 0|+\delta |1\rangle\langle 1|\equiv\alpha \rho_1+\beta \rho_2+\gamma \rho_3+\delta \rho_4$$.

The objective is to find a decomposition of the TPCP in a set of elementary operations $$E_i$$ such that :
$$\varepsilon(\rho)=\displaystyle \sum_{i=1}^4 E_i\rho E_i^\dagger$$.

To go from operatorial notation to actual numbers that result from sampling a quantum computer, we need to consider a fixed set of operators $$\tilde{E}_i$$, forming a basis for the set of operators on the state space, such that :
$$E_i=\displaystyle \sum_i e_{im}\tilde{E}_m$$,
where $$e_{im}$$ are some complex numbers.
Inserting this transformation into previous equation, we get : 
$$\varepsilon(\rho)=\displaystyle \sum_{m,n} \tilde{E}_m\rho \tilde{E}_n^\dagger \chi_{mn}$$, 
where $$\chi_{mn}=\displaystyle \sum_i e_{im}e_{in}^*$$ are the entries of a positive Hermitian matrix. 
This expression, known as the *chi matrix representation*, shows that the process can be fully characterized by a complex number matrix (a 4 $$\times$$4 matrix in the case of the single qubit) once the $$\tilde{E}_i$$ are fixed.

Experimentally, we would like to prepare each of the 4 states $$\varepsilon(|i\rangle \langle j|)$$, with $$\{i,j\}=\{0,1\}\times \{0,1\}$$, and perform state tomography on each of those states to express the process for an arbitrary input density matrix.

Preparing states $$|0\rangle \langle 0|$$ and $$|1\rangle \langle 1|$$ is easy since it is enough to initialize the quantum state of our qubit to the ground state or its excited state (latter is done by applying a simple X gate to input state $$|0\rangle$$).

For the two remaining, we can easily prepare the states $$|+\rangle=\frac{|0\rangle+|1\rangle}{\sqrt{2}}$$ and $$|-\rangle=\frac{|0\rangle+i|1\rangle}{\sqrt{2}}$$ and apply the process on those states. Recovering the original $$\varepsilon(|0\rangle \langle 1|)$$ and $$\varepsilon(|1\rangle \langle 0|)$$ can be done by using the following equality : 
$$\varepsilon(|i\rangle \langle j|)=\varepsilon(|+\rangle \langle +|)+i\varepsilon(|-\rangle \langle -|)-\frac{1+i}{2}\varepsilon(|i\rangle \langle i|)-\frac{1+i}{2}\varepsilon(|j\rangle \langle j|)$$

We have finally 4 reconstructed states :
$$\rho_1'=\varepsilon(|0\rangle \langle 0|)$$,
$$\rho_4'=\varepsilon(|1\rangle \langle 1|)$$
$$\rho_2'=\varepsilon(|+\rangle \langle +|)-i\varepsilon(|-\rangle \langle -|)-\frac{1-i}{2}(\rho_1'+\rho_4')$$,
$$\rho_3'=\varepsilon(|+\rangle \langle +|)+i\varepsilon(|-\rangle \langle -|)-\frac{1+i}{2}(\rho_1'+\rho_4')$$

where $$\rho_j'=\varepsilon(\rho_j)$$.

For one single qubit we would set $$\tilde{E}_1=I, \tilde{E}_2=X, \tilde{E}_3=-iY, \tilde{E}_4=Z$$ since we know that the Pauli operators do form a basis for the space of density matrices associated to the state of one single qubit. Furthermore, this choice is justified because of the convenient commutation relations which allow a reduction of the problem to a simple matrix multiplication.
It can be shown that with this particular basis choice, the *chi* matrix can be written as :
$$\chi=\Lambda \begin{pmatrix}\rho_1' & \rho_2' \\ \rho_3' & \rho_4'\end{pmatrix}\Lambda$$, with $$\Lambda=\frac{1}{2}\begin{pmatrix} I & X \\X & -I
\end{pmatrix}$$



# 2. The QUA program
### 2.1 The configuration file

The configuration file is architectured around the following items :

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

In this example, we inspire ourselves from the configuration of a traditional superconducting quantum computer with one single qubit coupled to a readout resonator, and the Gaussian pulses available are supposedly calibrated priorhand to perform $$\pi/2$$ rotations around X-axis or Y axis. Drag Pulses are implemented from the input data provided at the beginning of the script *configuration.py*.


## 2.2 The QUA program
The QUA program takes elements from the script done for Qubit state tomography, and takes back QUA macros to synthesize elementary single qubit gates.
It simply encapsulates four qubit state tomography experiments (for each of the input state described in the theory above) by using Python functions applying appropriate tomography, according to the axis desired, and the process described (chosen arbitrarily as a QUA macro).

Results are saved and retrieved using the features of stream_processing. The classical post processing of the data is done in a very similar way to original qubit state tomography. Tomography is done using two different methods for each of the input states, and the *chi* matrix is simply calculated according to the last equation of previous section.