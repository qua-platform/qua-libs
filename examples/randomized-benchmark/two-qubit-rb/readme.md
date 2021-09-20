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
- a  *QuantumMachinesManager* instance, which will be used to launch  the series of jobs necessary to play all synthezised sequences 
- a *config* dictionary, used to open a Quantum Machine and to store the waveforms randomly generated
- *N_Clifford*, the number of desired Clifford per sequence. Here an array of different values can be provided to retrieve different truncations
- *N_sequences*: Number of random sequences to be generated for averaging
- *two_qb_gate_macros*: Dictionary containing macros taking a baking object as input and using
 baking play statements on necessary quantum elements to perform the three 2 qubit gates mentioned above ("CNOT", "SWAP" and "iSWAP" which are also keys of the dictionary).  The native two-qubit gate macro shall always contain a baking align statement before and after the playing statements
- *single_qb_macros*: Optional dictionary in the case where a single qubit gate is not simply a QUA play statement on one quantum element, one can specify a dictionary of 
a set of macros. Note that the dictionary shall then contain macros for all single qubit generators, i.e "I", "X", "Y", "X/2", "Y/2", "-X/2", "-Y/2".
- *seed*: Random seed


# Run the experiment

The example shows briefly how one can run the experiment using the class. The method *run* uses one single argument,
which is the main QUA program the user should define with its own measurement commands and its own stream_processing, such that the retrieval and fitting of results can be done directly. 
The user shall build its QUA program according to two important guidelines:
1. The user shall involve the call of a *b.run()*, where *b* is the baking object retrievable by the attribute of the RBTwoQubitClass *baked_reference*.
2. The state estimation for both qubits should be done directly in QUA, and shall therefore contain two streams (one for each qubit state) filled with 0s or 1s in order to retrieve the result of each circuit independently in the post-processing.

The example of QUA program provided proposes to embed it into a function, allowing the user to choose the number of repetitions of each sequence he wants to set for getting good statistics.

Once the experiment is run, the user can retrieve the survival probability using the method *retrieve_results*, and can also plot the usual graph and perform a fitting of the results using the method *plot* of the class.

# Description of how the experiment is done and current limitations
2 qubit RB can be a challenge to implement because of the associated number of Cliffords one has to sample in the 2 qubit Clifford group (11520).
In our proposed implementation, we exploit the baking tool in order to perform the sequence generation and inversion before running the actual QUA program.
Moreover, we use the *add_compiled* feature of the QOP to be able to load random sequences successively, without having to compile a new program at every iteration. This scheme involves two main limitations that are described below.

## 1. Maximum number of Clifford limited on lengths of gates 
The first consequence of this method is that the number of maximum Clifford operations playable in the experiment is limited by the waveform memory, and is therefore highly dependent on the length of the gates the user has calibrated.
Assuming all single qubit gates have the same length, and using one single native two qubit gate with another (longer) length, we have performed the following approximative benchmark of how many Clifford operations are playable.
Note that this number may slightly vary as all Clifford operations do not carry the same number of gates and there is therefore some randomness on the actual number that can be played. The benchmark done below indicates the maximum number of playable Clifford operations in one single random sequence with certainty.


|                          | 2 qubit gate length [ns]             | 100 | 200 | 252 |
|--------------------------|----------------------------------------|-----|-----|-----|
| 1 qubit gate length [ns] |                                        |     |     |     |
| 32                       |                                        | 175 | 120 | 105 |
| 40                       |  | 150 | 110 | 95  |
| 52                       |                                        | 128 | 100 | 84  |
| 72                       |                                        | 100 | 81  | 72  |                                             | 100 | 81  | 72  |

## 2. Delay between active reset and actual start of the sequence
As mentioned earlier, the use of *add_compile* is done in order to minimize the overall required time to run the full experiment.
This is possible by using waveform overriding (more details here: https://qm-docs.s3.amazonaws.com/v1.10/python/features.html#precompile-jobs).
In order to exploit this feature, all waveforms that are passed between two jobs must be of exact same length.
This means that for each random sequence and each of its subsequent truncations (for shorter number of Clifford composing the sequence), the baked waveform shall systematically be filled with additional 0s in order to match the same number of samples as the longest baked waveform (i.e the one containing the maximum number of Clifford operations).
As a consequence, there is an added additional delay when playing shorter sequences as the padded 0s are placed before playing the actual truncated sequence. This padding method is used to avoid gaps between the play of the sequence and the start of the readout).
In case the user wants to implement an active reset feature, the user shall be aware that there is therefore a variable delay (dependent on which truncation is being played) between the moment the state is reset and the actual start of the random sequence.

