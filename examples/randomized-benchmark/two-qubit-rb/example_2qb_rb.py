from rb_2qb import *
from configuration import config

qmm = QuantumMachinesManager()

"""
For the required two qubit gates to generate the 4 classes (see Supplementary info of this paper:
https://arxiv.org/pdf/1210.7011.pdf), we require the user to complete the following macros below according
to their own native set of qubit gates, that is perform the appropriate decomposition and convert the pulse sequence
in a sequence of baking play statements (amounts to similar structure as QUA, just add the prefix b. before every statement)
Example :
in QUA you would have for a CZ operation:
    play("CZ", "coupler")
in the macros below you write instead:
    b.play("CZ", "coupler")
"""


# Baking Macros required for two qubit gates

def CNOT(b: Baking, *qe_set: str):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #
    pass


def iSWAP(b: Baking, *qe_set: str):
    pass


def SWAP(b: Baking, *qe_set: str):
    pass


"""
In what follows, q_tgt should be the main target qubit for which should be played the single qubit gate.
qe_set can be a set of additional quantum elements that might be needed to actually compute the gate 
(e.g fluxline, trigger, ...) 
"""


def I(b: Baking, q_tgt, *qe_set: str):
    pass


def X(b: Baking, q_tgt, *qe_set: str):
    pass


def Y(b: Baking, q_tgt, *qe_set: str):
    pass


def X_2(b: Baking, q_tgt, *qe_set: str):
    pass


def Y_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mX_2(b: Baking, q_tgt, *qe_set: str):
    pass


def mY_2(b: Baking, q_tgt, *qe_set: str):
    pass


two_qb_gate_macros = {
    "CNOT": CNOT,
    "iSWAP": iSWAP,
    "SWAP": SWAP
}

single_qb_gate_macros = {
    "I": I,
    "X": X,
    "Y": Y,
    "X/2": X_2,
    "-X/2": mX_2,
    "Y/2": Y_2,
    "-Y/2": mY_2
}


def measure():
    pass


def stream_macro():
    pass


nCliffords = range(1, 10, 2)
s = RBTwoQubits(qmm=qmm, config=config, quantum_elements=["qe1", "qe2"],
                N_Clifford=nCliffords, K=1, N_shots=100,
                two_qb_gate_baking_macros=two_qb_gate_macros,
                measure_macro=measure, stream_macro=stream_macro)
sequences = s.sequences
s1 = sequences[0].full_sequence
for s in s1:
    print(len(s), s)
