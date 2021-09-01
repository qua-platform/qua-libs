# Randomized Benchmarking for two-qubit gates using baking

Randomized benchmarking (RB) is a procedure to generate an 
average figure-of-merit characterizing the fidelity of 
qubit operations. This procedure is useful as full process tomography 
may be prohibitively costly in terms of the number of required operations.   
RB utilizes operations from the Clifford group, which in the case of two qubits contains 11520 elements
(for $$n$$ qubits $$
\left|\mathcal{C}_{n}\right|=2^{n^{2}+2 n} \prod_{j=1}^{n}\left(4^{j}-1\right)
$$).

We propose to illustrate the use of baking tool to perform this protocol.

## The class *RBTwoQubits*

The pattern behind RB is always the same: 
- Generate a random sequence composed of Cliffords
- Append to the sequence one Clifford reverting the action of the random sequence, such that the identity circuit is played
- Measure the survival probability
- Repeat prior steps for varying number of Cliffords within the sequence

In order to have an easy way to derive all Cliffords of the two qubit Clifford group, 
we use the decomposition illustrated in the following reference: https://arxiv.org/pdf/1210.7011.pdf.
This decomposition allows an easy derivation of the group from the Clifford group for one qubit, and uses three main two qubit gates
(that shall be decomposed later into a native set of gates by the user depending on its own hardware):
- CNOT
- SWAP
- iSWAP

We introduce a specific class that allows an easy generation and execution of multiple random Clifford sequences based on 
few generic inputs the user shall provide:
- a  QuantumMachinesManager instance, which will be used to launch  the series of jobs necessary to play all synthezised sequences 
- a config dictionary, used to open a Quantum Machine and to store the waveforms randomly generated
- - *quantum_elements*: all quantum elements necessary to compute the gates. Note that the first two elements of this list shall be consisting the adressing of *qubit0* and *qubit1* directly, or 
if macros are indicated 
- *max_length*: Maximum number of Cliffords the random sequence shall contain
- *K*: Number of random sequences to be generated for averaging
- *two_qb_gate_macros*: Dictionary containing macros taking a baking object as input and using
 baking play statements on necessary quantum elements to perform the three 2 qubit gates mentioned above ("CNOT", "SWAP" and "iSWAP" which are the three keys of the dictionary).
Those macros shall be Python functions taking at least one baking object in order to store the corresponding set of waveforms in the config to play the gate
- *measure_macro*: QUA macro to be used within a QUA program to perform the readout of the final two qubit state
- *measure_args*: Tuple containing arguments to be passed to *measure_macro* to perform readout
- *single_qb_macros*: Optional dictionary in the case where a single qubit gate is not simply a QUA play statement on one quantum element, one can specify a dictionary of 
a set of macros. Note that the dictionary shall then contain macros for all single qubit generators, i.e "I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2"
- *truncation_positions*: Optional Iterable indicating at which lengths the search and implementation of an inverse operation shall be done (necessary for the fitting).
If no iterable is provided, then all truncations from 1 to *max_length* are performed
- *seed*: Random seed


# Run the experiment
Once the instance of the previously class is created,
one can launch the full experiment by using the method execute(), which will create a series of pending jobs to run all sequences generated with the baking.
The output of this method is a result_handles to be post-processed for graph generation.

