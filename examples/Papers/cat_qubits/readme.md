---
title: Cat-qubits for Quantum Computation
sidebar_label: Cat-qubit paradigm
slug: ./
id: index
---

# Introducing Schrödinger cat qubit paradigm for QC using QUA

### Reference papers :
- [1] Cat-qubits for quantum computation: https://www.sciencedirect.com/science/article/pii/S1631070516300627?via%3Dihub
- [2] Hardware-Efficient Autonomous Quantum Memory Protection: https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.111.120501
- [3] Exponential suppression of bit-flips in a qubit encoded in an oscillator: https://www.nature.com/articles/s41567-020-0824-x
- [4] Measuring the photon number parity in a cavity: from light quantum jumps to the tomography of non-classical field states: https://www.tandfonline.com/doi/full/10.1080/09500340701391118
- [5] Building a fault-tolerant quantum computer using concatenated cat codes :https://arxiv.org/pdf/2012.04108.pdf
- [6] Dynamically protected cat-qubits:
a new paradigm for universal quantum computation: https://iopscience.iop.org/article/10.1088/1367-2630/16/4/045014
- [7] Repetition Cat Qubits for Fault-Tolerant Quantum Computation:  https://arxiv.org/abs/1904.09474
- [8] Bias-preserving gates with stabilized cat qubits: https://arxiv.org/abs/1905.00450


# Introduction

NISQ devices rely on qubits that are still subject to all sorts of noisy processes, destroying quantum information and therefore interfering with the coherent computations delivered by the promises of having fault tolerance. Among the multiple platforms available for quantum computing today, one stands as particularly interesting due to its capacity to perform partial "autonomous" quantum error correction. By autonomous, we mean that the system is engineered such that its natural time evolution protects the encoded information from Markovian errors.
This paradigm relies on the control of the so-called classical states of light, or "coherent" states of light. 
The goal of this overview is to provide the basics of this encoding, and how QUA can be used to perform calibration and control of the associated hardware, presented in reference [3] above, which will be the main paper of reference for the QUA scripts.

# Reminder on coherent states of lights

The cat qubit is defined upon specific states of lights characterizing the modes in a high-Q cavity that behaves like a quantum harmonic oscillator (QHO). Provided the eigenbasis $$\{|n\rangle, n\in \mathcal{N}\}$$ of the Hamiltonian characterizing the QHO ($$\hat{H}=\hbar\omega(\hat{a}^\dagger\hat{a}+\frac{1}{2})$$), we define the coherent state of complex amplitude $$\alpha$$ as :
$$|\alpha\rangle:=e^{-|\alpha|^2/2} \displaystyle \sum_n\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

Those states constitute eigenstates of the annihilation operator  $$\hat{a}$$ of the QHO, that is $$\hat{a}|\alpha\rangle=\alpha|\alpha\rangle$$. Note that $$|\alpha\rangle$$ is the unique eigenstate associated to the eigenvalue $$\alpha$$.
One can easily show that the average photon number $$\langle\hat{a}^{\dagger}\hat{a}\rangle\equiv\langle\alpha|\hat{a}^{\dagger}\hat{a}|\alpha\rangle$$ on this state is given by $$|\alpha|^2$$.
Moreover, the dot product between two coherent states $$|\alpha\rangle$$ and $$|\beta\rangle$$ decreases exponentially with the norm of the respective photon numbers : $$\langle\alpha|\beta\rangle=e^{-\frac{1}{2}(|\alpha|^2+|\beta|^2-2\alpha^*\beta)}$$
This means that $$|\alpha\rangle$$ and $$|-\alpha\rangle$$ are quasi-orthogonal if $$|\alpha|^2\gg 1$$: $$\langle\alpha|-\alpha\rangle=e^{-2|\alpha|^2}$$.
Finally, a coherent state can be obtained from the vacuum state of the QHO by applying the displacement operation:
$$\hat{D}(\alpha)|0\rangle=e^{\alpha\hat{a}^\dagger-\alpha^*\hat{a}}|0\rangle=|\alpha\rangle$$.

Another interesting property about those states is that they minimize the Heisenberg uncertainty relation, as it can be shown that $$\Delta x \Delta p=\hbar/2$$ for $$\alpha=x+ip$$. This fact justifies the name of classical states of light in the literature, as it reduces to the minimum amount the quantum character of the light.
Moreover, we call a quantum superposition of coherent states of opposite phases "Schrödinger cat states", because as the example of the cat, this is a quantum superposition of two macroscopic states usually very well observed in classical physics.


# Encoding the qubit state on a QHO: advantage of bosonic code
![Bloch sphere of the cat qubit [5]](images/cat_qubit_sphere.png)

The qubit state is encoded on the previously introduced coherent states, which are considered as orthogonal since the average number of photons in the cavity is always assumed to higher than 1, as shown on the Bloch sphere above.

This encoding is an efficient conversion of a continuous variable description of a quantum system into a logical two-level system. On a usual transmon qubit, a truncation of the Hilbert space 
associated to a QHO would be done to work with a subspace spanned by the ground and first excited states of the QHO. Here, the fact that we use coherent states (which are superpositions of Fock states) 
do allow a natural delocalization of the stored quantum information onto many energy levels. We are now going to see how this encoding can also be used as a strength to overcome noisy quantum channels characterizing a cavity by turning 
them into simple logical digital errors.
It turns out that the two main error channels that usually occur are respectively the photon loss (also called energy relaxation) and phase damping (or dephasing) channels.
What is done in [3] to overcome the photon loss channel is the engineering of a specific interaction that strongly couples the cavity encoding the cat qubit to another cavity called the "buffer" which focuses on the exchange of photons by pair. This dissipation engineering technique is used to control the entropy flow accumulated in the system by focusing on one type of dissipation, at the expense of the usual single photon loss channel rate.
The Master equation characterizing the system can be written as:
$$\frac{d \hat{\rho}(t)}{d t}=\kappa_{2} \mathcal{D}\left[\hat{a}^{2}-\alpha^{2}\right] \hat{\rho}(t)+\kappa_{1} \mathcal{D}[\hat{a}] \hat{\rho}(t)+\kappa_{\phi} \mathcal{D}\left[\hat{a}^{\dagger} \hat{a}\right] \hat{\rho}(t)$$, with $$\mathcal{D}[\hat{L}] \hat{\rho}:=\hat{L} \hat{\rho} \hat{L}^{\dagger}-\frac{1}{2}\left(\hat{L}^{\dagger} \hat{L} \hat{\rho}+\hat{\rho} \hat{L}^{\dagger} \hat{L}\right)$$. 
The first term describes the two photon dissipation channel, while second and third terms are respectively the single photon loss and dephasing channels. The objective is to design the system such that $$\kappa_2\gg\kappa_1$$.

 Focusing on this two-photon dissipation process is effective because of the following realization :
quantum states resulting from linear combinations of coherent states encoding our qubit (that is our subspace of interest) are steady states of the Master equation describing this process, that is:
$$\frac{d \hat{\rho}(t)}{d t}=0$$, $$\forall \hat{\rho}\in \text{span}\{|\alpha\rangle\langle\alpha|,|-\alpha\rangle\langle -\alpha|\}$$,
under the assumption that $$\kappa_1 = \kappa_\phi=0$$. This is powerful, as it indicates that this decay process confines the system in the manifold encoded by the initial codeword. 


# Configuration of experimental setup 
We propose here a series of QUA scripts that are meant to generate the data necessary to produce figures in the reference [3], which uses the cat qubit encoding to demonstrate an exponential suppression of bit flip errors at the expense of linear increase of phase flip errors with the average photon number in the cavity.
![Experimental setup [3].](images/experimental_setup.png)
We have the following quantum elements:

- *buffer_drive*, with varying drive amplitude to tune up the average number of photons at frequency $$\omega_b$$. The buffer is in the experiment in [3] another cavity carrying a very high photon loss rate.

- *transmon*, used to perform a parity measurement (equivalent to a readout of the cat qubit in the Hadamard basis)
- *RR*, readout resonator, coupled to the transmon qubit

- *ATS*, element pumped at frequency $$\omega_p=2\omega_a-\omega_b$$
- *storage*, the cavity where the cat qubit is encoded with eigenfrequency $$\omega_a$$, usually denoted as the storage mode

As one can see on the layout, the experimental setup is divided in two parts. On the right side, the combination transmon-readout resonator focuses on the realization of a QND measurement of the cat qubit state, whereas the combination of the buffer and the ATS are used for both control and stabilization of this very same cat qubit state. In fact, the latter combination allows the engineering of a Hamiltonian $$\hat{H}_{i} / \hbar=g_{2} \hat{a}^{\dagger 2} \hat{b}+g_{2}^{*} \hat{a}^{2} \hat{b}^{\dagger}$$, which yields the two-photon dissipation term in the Master equation above once the modes of the buffer are adiabatically eliminated (which can be done under the assumption that the average photon number in the buffer cavity is close to zero.)

In the experiment conducted in the paper, the cat qubit cavity is directly controllable using a microwave drive for realising Wigner tomography. The Wigner distribution of a quantum state described by a density matrix $$\rho$$ is given by :
$$W(x,p)=\frac{2}{\pi} \mathrm{Tr}[\hat{D}(-\alpha)\hat{\rho}\hat{D}(\alpha)\hat{\Pi}]$$
where $$\alpha=x+ip$$ and $$\hat{\Pi}=e^{i\pi\hat{a}^\dagger\hat{a}}$$ is the parity operator.

This relation means that, for a field described by the density operator $$\hat{\rho}$$, the Wigner function at an arbitrary point $$\alpha$$ in phase space is the expectation value of the parity operator of the translated field of density operator $$\hat{D}(-\alpha)\hat{\rho}\hat{D}(\alpha)$$, obtained by the action of the displacement $$\hat{D}(-\alpha)$$ on the original field.

The parity operator is the QND observable measured using the probing of the transmon qubit, thanks to a dispersive interaction between the cat qubit cavity and the transmon of the type $$H_{int}=-\chi|e\rangle\langle e| \otimes \hat{a}^\dagger\hat{a}$$. Performing a Ramsey experiment on the transmon qubit (two $$\pi/2$$ pulses separated by a wait duration $$\tau=\pi/\chi$$) allows the deduction of the parity by the following mapping: measuring the transmon in the ground state yields an even parity whereas a measurement of the excited state yields the odd parity [4].

# Protecting the codeword from bit-flip errors

The storage-buffer engineered interaction creates a dynamic that keeps refocusing the storage quantum state onto the manifold spanned by the two attractors $$|\alpha\rangle$$ and $$|-\alpha\rangle$$. Moreover, increasing the value of $$\alpha$$, which can be done by simply increasing the amplitude of the drive applied on the buffer ($$\alpha^2:=-\epsilon_d/g_2^*$$), tends to increase the distance in phase space between the two attractors. The demonstration done in [4] is the following: the combination of the attractor potential and the great distance in phase space between the logical $$|0\rangle_\alpha$$ and $$|1\rangle_\alpha$$ induces an exponential suppression of bit flip errors with the value of $$\alpha$$. This can be understood intuitively by considering that when noisy processes occuring on the cat are local in phase space, the consequence might be a temporary escape of the coherent states manifold while staying near the original attractor. 

Using the two photon dissipation process, the state of the cat eventually falls back to one of the two attractors, and the probability of falling back on the original one (that is the state the cat was in before escaping the manifold) is significantly improved if the other attractor is very far away in phase space, hence reducing the probability of having a bit flip error.

![Pseudo-potential induced by the two-photon dissipation engineered mechanism [3]](images/potential.png)

## What about phase flip errors?
Phase flip errors typically arise when considering the single photon loss channel. In fact, considering the annihilation operator as a jump operator, we have that:
$$\hat{a}|\pm\alpha\rangle=\pm\alpha|\alpha\rangle$$
Since the output state is renormalized after the jump and we can ignore a global phase ($$|\psi_{out}\rangle=\frac{\hat{a}|\pm\alpha\rangle}{|\hat{a}|\pm\alpha\rangle|}=\pm e^{i\theta}|\alpha\rangle\equiv\pm |\alpha\rangle, \alpha=re^{i\theta}$$), we indeed recover the action of a logical Z error on the codeword.
One could argue that the protection scheme described above does not help reducing the phase flip error rate, it might actually even increase it since single photon loss channel becomes more likely with an increasing number of photons in the cavity (i.e $$|\alpha|^2\gg 1$$). The argument made in the paper is that while we do benefit from an exponential suppression of bit-flip errors with the cat-size, the undesirable increase of the phase-flip error rate only scales linearly with the same parameter. 
In other words, it is advanteageous to bias the qubit error rate towards phase-flip errors only as the scales involved with the cat-size allow a full removal of bit-flip errors. What can hence be done to have a full protection is to perform active error correction, using the repetition code as depicted in [7], to protect the qubit against residual phase-flip errors while its natural dynamics protect it from bit-flip errors.

Since the question of the scalability of repetition codes or surface codes is often a major issue for reaching fault tolerance, this partial autonomous quantum error correction feature might contribute to reduce the overhead related to the required number of qubits to build a full error corrected scheme.

## Towards fault tolerance using hybrid autonomous/active QEC
### The notion of a bias-preserving gate
Performing active error correction with cat-qubits involve the possibility of performing efficient control and readout while preserving the error bias towards phase errors as depicted previously. In fact, bias-preserving gates, defined as quantum operations that do not modify the bit-flip rate while being played, do constitute a major area of research that create new directions in the field of reservoir engineering [8].
A two-qubit gate $$U$$ is said to be bias-preserving if:

$$\left[U Z_{1,2} U^{\dagger}, Z_{1,2}\right]=0$$

Let's see two examples to understand more concretely the definition.

Consider the gate: $$ZZ(\theta)=e^{i\theta Z_1 Z_2/2}$$.
This gate is usually computed by implementing a Hamiltionian of the form $$\hat{H}\propto -V Z_1 Z_2$$. 
The subsequent unitary evolution operator associated to this gate is therefore: 
$$\widehat{U}(t)=e^{iVtZ_1 Z_2}\implies ZZ(\theta)$$ is realized in a time $$T=\theta/2V$$. 
Now Assume a phase-flip error occurs at time $$0\leq\tau\leq T$$, we have the following evolution:
$$\widehat{U}_{\mathrm{e}}(T)=\widehat{U}(T-\tau) \hat{Z}_{1 / 2} \widehat{U}(\tau)=\hat{Z}_{1 / 2} \widehat{U}(T),$$
We realize here that the imperfect gate $$\widehat{U}_{\mathrm{e}}(T)$$ is equivalent to an error-free gate followed by a phase flip. We therefore did not change the bit flip error rate by playing this gate, it is therefore bias-preserving.

Consider now the CNOT (CX) gate, derived from an interaction of the form: $$\widehat{H}_{\mathrm{CX}}=V\left[\left(\frac{\hat{I}_{1}+\widehat{Z}_{1}}{2}\right) \otimes \hat{I}_{2}+\left(\frac{\hat{I}_{1}-\hat{Z}_{1}}{2}\right) \otimes \widehat{X}_{2}\right]$$,
leading to a unitary evolution $$\widehat{U}(t)=\exp \left(-i \widehat{H}_{\mathrm{CX}} t\right)$$.
Setting $$T$$ such that $$VT=\pi/2$$, and assuming that a phase flip error on qubit 2 (target) occurs at time $$0\leq\tau\leq T$$, we have:  
$$\widehat{U}_{\mathrm{e}}(T) =\widehat{U}(T-\tau) (\hat{I}_{1} \otimes \hat{Z}_{2}) \widehat{U}(\tau) =(\hat{I}_{1} \otimes \widehat{Z}_{2}) e^{i V(T-\tau)\left(\hat{I}_{1}-\widehat{Z}_{1}\right) \otimes \hat{X}_{2}} \widehat{U}(T)$$.
We see in this case that the CNOT gate has converted the original phase flip error into a rotation around the X axis for the target qubit, leading to an increase of the bit-flip rate. The CNOT gate is therefore not bias-preserving.

Based on this realization, it is clear that the set of available bias-preserving gates is not universal for quantum computation (a simple proof is available in section F of the Appendix of [8]). For the important use case of error correction, more specifically to measure X types stabilizers (aimed for phase error correction), being able to perform a CNOT gate is absolutely crucial. 
We shall hence derive below from [5] and [7] how we can compute a CNOT gate in a bias preserving way using the additional degrees of freedom associated to our cat-qubit encoding, and how this would translate into a physical pulse sequence written in QUA.

### Deriving a logical X gate
Let us start by describing the Hamiltonian and the subsequent harware implementation allowing the generation of a bias-preserving version of the X gate, from which we shall derive the implementation of the CNOT gate.

Recall that the combination of the ATS and the buffer cavity coupled the storage mode allow the tuning of a two-photon dissipation process of the form:
$$\frac{d \hat{\rho}(t)}{d t}=\kappa_{2} \mathcal{D}\left[\hat{a}^{2}-\alpha^{2}\right] \hat{\rho}(t)$$.
This dynamics does stabilize the manifold spanned by $$\{|\pm\alpha\rangle\langle\pm\alpha|\}$$. Now, the idea is that one can dynamically tune up the value of $$\alpha$$ and make it time dependent in order to temporarily escape the original cat qubit manifold. We hence do the following transformation:
$$\alpha\rightarrow \alpha(t)=\alpha e^{i\pi t/T}$$, $$\alpha(0)=\alpha$$, 
$$\alpha(T)=\alpha e^{i\pi}=-\alpha$$, 
$$t\in[0,T]$$
One can see here that we introduce a time dependence on the value of $$\alpha$$, such that we emulate an adiabatic deformation on the codespace, eventually leading to a flip of the original basis, such that we eventually have indeed transformed the logical $$|0\rangle_\alpha=|\alpha\rangle$$ to $$|1\rangle_\alpha=|-\alpha\rangle$$ and vice versa. 
Eventually, we recover the following dissipator:
$$\frac{d \hat{\rho}(t)}{d t}=\kappa_{2} \mathcal{D}\left[\hat{a}^{2}-(\alpha e^{i\pi t/T})^{2}\right] \hat{\rho}(t)$$.
The deformation must hence be adiabatic in order for the stabilization to occur at the same time as the position of the two attractors is being exchanged such that a basis flip is performed while temporarily exiting and going back to the original codespace. The diagram shown in [7] and displayed below summarizes the dynamics presented above.
![[7]](images/X_gate.png)

 The optimal "path" that the function $$\alpha(t)$$ should take in order to optimize the gate fidelity is currently an open question. In practise, tuning dynamically the value of $$\alpha$$ amounts to changing the relative phase of the weak buffer drive. Real time phase modulation can be realized in QUA by either defining a fixed detuning $$\Delta$$ from the resonant frequency $$\omega_d==\omega_b$$ (this will yield a linear change of the relative phase $$\phi(t)$$ as $$\frac{d\phi(t)}{dt}=\omega$$), or by using the frequency chirp, which allows the realization of a dynamic evolution of the frequency, which hence can define a variable rate for the relative phase. 
In QUA, those frequency changes can be implemented in a few lines, which can either be:

    with program() as prog:
        # Option 1
        detuning = declare(int, value = 1e6)
        update_frequency(freq)
        
        # Option 2
        freq_profile = declare(int, value=[25000, 0, 50000, -30000, 80000])
        play("drive", "buffer", chirp=(freq_profile, "Hz/nsec")    

While those options are the easiest to implement, it could be that the resolution available is not enough to comply with the required adiabaticity. Another way around is the usage of the baking tool, allowing to perform frame rotations and detuning settings at the resolution of the nanosecond. However, using this method for long pulses could be inefficient in terms of waveform memory management, it is therefore crucial to assess precisely the requirements to perform fast and reliable X gates and implement the best alternative with the QOP.

### Deriving the CNOT gate
The CNOT gate is a natural extension of the code deformation detailed above. It is essentially implementing the same Hamiltonian on a target qubit, but this time conditioned to the state of another control qubit. The gate is given by the following unitary [7]:
CNOT $$\approx
|\alpha\rangle\langle\alpha|\otimes I_{\alpha}+|-\alpha\rangle\langle-\alpha| \otimes X_{\alpha}$$
where $$I_{\alpha}=|\alpha\rangle\langle\alpha|+|-\alpha\rangle\langle-\alpha|$$ and $$X_{\alpha}=|\alpha\rangle\langle-\alpha|+|-\alpha\rangle\langle\alpha|$$. 
The approximation is exponentially precise in $$|\alpha|^{2}$$. 
To implement this gate, we use two dissipation channels $$\mathcal{L}_{\hat{a}}=\mathcal{D}[\hat{L}_{\hat{a}}]$$ and $$\mathcal{L}_{\hat{b}}=\mathcal{D}[\hat{L}_{\hat{b}}(t)]$$
with $$\hat{L}_{\hat{a}}=\hat{a}^2-\alpha^2$$,
$$\hat{L}_{\hat{b}}(t)=\hat{b}^{2}-\frac{1}{2} \alpha(\hat{a}+\alpha)+\frac{1}{2} \alpha e^{2 i(\pi / T) t}(\hat{a}-\alpha)$$.
As one can see, the control qubit has a usual stabilization scheme, whereas the target does have a dissipator dependent of the state of the control qubit. Typically, if the control is in state $$|\alpha\rangle\equiv |0\rangle_\alpha$$, meaning that $$\langle\hat{a}\rangle=\alpha$$, we recover the usual dissipator for the target qubit. Contrariwise, if the control is in state $$|-\alpha\rangle\equiv |1\rangle_\alpha$$, i.e $$\langle\hat{a}\rangle=-\alpha$$, we recover the dissipator associated to a X gate for the target.

The way to compute those two dissipators simultaneously has been outlined in [7], and requires frequency multiplexing on both the ATS and the buffer coupled to the two storage modes of interest. The multiplexing scheme consists in addressing different storage modes by selecting a specific resonant frequency for the drive corresponding to each individual storage mode. The scheme is detailed further in Section II.E in [5].
As one could have guessed by the form of the dissipator on the control, computing a CNOT involves performing a usual stabilization round for the control qubit. This is done by pumping the ATS at frequency $$\omega_{p_1}=2\omega_{a_1}-\omega_{b}$$ and driving the buffer at frequency $$\omega_d = \omega_b$$. Here, we assimilate $$\omega_{a_1}$$ as the eigenfrequency of the storage mode acting as the control qubit, as opposed to $$\omega_{a_2}$$ which corresponds to the equivalent frequency for the target qubit. 
Now, the dissipator on the target has to be implemented such that the control state is involved. First of all, we notice that regardless of the state of the control, the dissipator involves a stabilization scheme for the target mode (only the rotation in the IQ plane is conditional to the control state). This means that the ATS shall be also pumped at frequency $$\omega_{p_2}=2\omega_{a_2}-\omega_b$$.
The involvement of the control mode in the rotation of the IQ plane is realized by driving the buffer at another frequency $$(\omega_{a_1}-\omega_b)/2$$. We therefore have two-frequencies multiplexing for both the ATS and the buffer. 

This multiplexing can be realized in QUA by defining two distinct quantum elements with identical OPX output ports. The definition of the two elements allows the call of two independent pulser resources, in order to synthezise separately the pulses with different frequencies and them mix them altogether before sending it to the hardware using a single pair of outputs.

To summarize, we have derived the implementation of two quantum gates, crucial for quantum control and more particularly for the realization of a repetition code. Those two gates are computed in such a way that it is maintained bias-preserving, ensuring that the exponential suppression of bit-flip errors arising from a high number of photons in the cavity is still relevant.
We shall now focus on describing how the QOP can efficiently perform simultaneous active and autonomous quantum error correction by taking the example of the surface code described in [5].

## The surface code



![Circuit used for an X-basis measurement in the context of an X-type stabilizer measurement. The first step consists of entangling the ancilla qubit with the data qubits. Afterwords, the ancilla qubit is deflated followed by a SWAP with a readout mode. Lastly, the readout mode is repeatedly measured using a transmon qubit. The duration’s for the parts of the measurement procedure are labeled at the bottom of the figure below each circuit element. While these repeated parity measurements are occurring, the CNOT gates of the next error correction cycle can begin. Also included is a diagram of the physical layout of the stabilizer to give context to the measurement circuit [5].](images/Syndrome_measurement.png)
![Cat-qubit stabilization in the surface-code architecture. Each ATS is coupled to two data modes α, γ and two ancilla modes β,δ. In practice, ATSs are also coupled to a fifth readout mode (not shown here because it is not stabilized by any ATS). Each ATS is responsible for performing four CNOT gates (at different time steps) and stabilizing two phononic modes in the cat-code manifold during each time step [5].](images/Stabilization_schedule.png)
