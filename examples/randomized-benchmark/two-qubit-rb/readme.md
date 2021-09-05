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
- *N_Clifford*, the number of desired Clifford per sequence. Here an array of different values can be provided to retrieve different truncations
- *K*: Number of random sequences to be generated for averaging
- *two_qb_gate_macros*: Dictionary containing macros taking a baking object as input and using
 baking play statements on necessary quantum elements to perform the three 2 qubit gates mentioned above ("CNOT", "SWAP" and "iSWAP" which are the three keys of the dictionary).
- *quantum_elements*, shall specify the name in the config of the two quantum elements corresonding to the two qubits to be characterized. This is to be specified if the 1 qubit Clifford set is specified as operations for each of those two quantum elements. 
Those macros shall be Python functions taking at least one baking object in order to store the corresponding set of waveforms in the config to play the gate
- *single_qb_macros*: Optional dictionary in the case where a single qubit gate is not simply a QUA play statement on one quantum element, one can specify a dictionary of 
a set of macros. Note that the dictionary shall then contain macros for all single qubit generators, i.e "I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2". If this argument is provided, there is no need to specify the quantum_elements argument above
- *seed*: Random seed


# Run the experiment
Once the instance of the previously class is created,
one can launch the full experiment by using the method run, which will create a series of pending jobs to run all random sequences and their truncations generated with the baking tool.
The output of this method is a list of result_handles to be post-processed for graph generation. Each result_handles correspond to the result of one sequence in particular.
This method takes as an input a QUA program, which should contain a *b.run()* statement, where *b* is the attribute *baked_reference* of the RBTwoQubits instance. It should also contain the measurement and the stream processing adapted to the user's setup.
An example of such QUA program is provided in the executable script.

With this method, one can hope to run up to 170 Cliffords (assuming single qubit gate are 32 ns long and two qubit gates are 100 ns long). This estimation is highly dependent of each pulse length.

