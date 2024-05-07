# Cross-Entropy Benchmarking (XEB)

Cross-Entropy Benchmarking (XEB) [1] is a method to estimate the fidelity of a quantum computer. Originally thought
as an experiment to demonstrate the quantum supremacy regime [2], XEB has become an alternative to Randomized Benchmarking for 
estimating gate and layer fidelities in near-term quantum devices [3]. The protocol relies on the ability to perform random circuits on the quantum computer and compare the experimental results with the expected distributions computed with ideal simulation.

We leverage the power of QUA and the OPX to perform real-time gate random sampling, which allows us to generate random circuits on the fly and straightforwardly execute them on the quantum computer. The script will then calculate the cross-entropy between the ideal and actual probability distributions to estimate the layer fidelity.

## Experimental Setup
<img align="right" width="400" src="setup.png">

The use-case in this example is tailored for a superconducting quantum processor using flux-tunable transmon qubits, where we focus on a subset of two qubits that are capacitively coupled to each other. Single qubit operations are controlled by sending microwave pulses through a xy-line that is capacitively coupled to the individual qubits. The two-qubit gate is implemented by a controlled-Z (CZ) gate utilizing fast-flux pulses to rapidly change the qubit frequencies. One important experiment on the way of tuning up a CZ gate is the flux-pulse calibration that yield qubit state oscillations depending on the pulse parameters. This experiment was performed and presented in the use-case [Two-Qubit Gate Optimization](https://github.com/qua-platform/qua-libs/tree/main/Quantum-Control-Applications/Superconducting/Two-Flux-Tunable-Transmons/Use%20Case%201%20-%20Two%20qubit%20gate%20optimization%20with%20cryoscope).

## Prerequisites
Prior to running the XEB example file `xeb_2q.py`, the user has to run the calibrations that define the gate and measurement parameters:
- Single Qubit Gates: Implement three single qubit gates: $X/2$ (SX), $Y/2$ (SY), and either $T$ or $SW$ (depending on the gate set choice, see below)
- Flux-Pulsed CZ Gate: Implement the two-qubit gate of interest, that will form the entangling layer in the experiment (together with the single qubit gates).
- Calibrated Measurement Protocol for Qubit State Discrimination: Simultaneously measure the two-qubit system in its computational basis states ∣00⟩, ∣01⟩, ∣10⟩, ∣11⟩.
## How does the script work?

The protocol relies on the ability to perform random circuits on the quantum computer and compare the results with the ideal simulation. 
The advantage of using QUA is that the randomization of the circuits can be done in parallel to circuit execution through real-time processing and random sampling.

The user can choose which gate set to use to generate random unitaries. Usually, the experiment is performed with layers of random single-qubit gates followed by one fixed two-qubit gate. 
The script will generate random circuits with the chosen gate set and run them on the quantum computer. The script will then calculate the cross-entropy between the ideal and actual probability distributions to estimate the layer fidelity (from which the two-qubit gate fidelity can be inferred).

As opposed to Randomized Benchmarking, we do not invert the randomly generated circuit by applying an inverse, but we rather perform a fidelity estimation over the statistics of the outcomes when measuring the system in the computational basis.

There are therefore four steps in the script:
- Random circuits generation: Done within QUA in real-time, the script generates random circuits of different depths with the chosen gate set. At the same time, the gates sampled in real-time are streamed back to the classical side for the theoretical simulation.
- Execution: For each random sequence of gates of varying depths, the script runs the circuit on the quantum computer while leveraging real-time pulse modulation of the OPX. This is particularly useful for playing all possible random gates through one single gate baseline (usually the $SX$ gate)
- Theoretical simulation: The script will simulate the random circuits on the classical side to calculate the ideal probability distributions.
- Cross-entropy calculation: The script will calculate the cross-entropy between the ideal and actual probability distributions to estimate the layer fidelity.

This script requires Qiskit [2] (we recommend installing beyond 1.0, see documentation [here](https://qiskit.org/documentation/install.html) or check this [video](https://youtu.be/dZWz4Gs_BuI?si=EOqyeOhZ05YcBlXA)) for the reconstruction of theoretical quantum circuits. This is helpful as it enables the user to leverage all Qiskit visualization tools to debug the experiments.
For the post-processing, we leverage Cirq [3] to calculate the cross-entropy between the ideal and actual probability distributions.

## References

[1] Boixo et al. [Characterizing Quantum Supremacy in Near-Term Devices](https://www.nature.com/articles/s41567-018-0124-x). Nature Physics 14, 595–600 (2018).  

[2] Arute, F., Arya, K., Babbush, R. et al. [Quantum supremacy using a programmable superconducting processor](https://doi.org/10.1038/s41586-019-1666-5). Nature 574, 505–510 (2019).

[3] Foxen, B. et al. [Demonstrating a Continuous Set of Two-Qubit Gates for Near-Term Quantum Algorithms](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.120504). Phys. Rev. Lett. 125, 120504 (2020).

[4] [Qiskit](https://qiskit.org/): An Open-source Framework for Quantum Computing. 

[5] [Cirq](https://quantumai.google/cirq): An Open-source Framework for NISQ Algorithms. 