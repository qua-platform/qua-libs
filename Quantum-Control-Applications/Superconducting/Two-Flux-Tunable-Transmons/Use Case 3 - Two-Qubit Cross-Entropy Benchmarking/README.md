# Cross-Entropy Benchmarking (XEB)

This script implements the Cross-Entropy Benchmarking (XEB) [1] technique for assessing the performance of a quantum computer with a focus on two qubits.

## What is XEB?

XEB is a method to estimate the fidelity of a quantum computer. Fidelity represents how well the actual quantum computer executes a circuit compared to a perfect, noiseless simulation.

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

This script requires Qiskit [2] (we recommend installing beyond 1.0, see documentation here:  https://qiskit.org/documentation/install.html or https://youtu.be/dZWz4Gs_BuI?si=EOqyeOhZ05YcBlXA) for the reconstruction of theoretical quantum circuits. This is helpful as it enables the user to leverage all Qiskit visualization tools to debug the experiments.
For the post-processing, we leverage Cirq [3] to calculate the cross-entropy between the ideal and actual probability distributions.

## References

[1] Boixo et al. (2018). Characterizing Quantum Supremacy in Near-Term Devices. https://www.nature.com/articles/s41567-018-0124-x

[2] Qiskit: An Open-source Framework for Quantum Computing. https://qiskit.org/

[3] Cirq: An Open-source Framework for NISQ Algorithms. https://quantumai.google/cirq