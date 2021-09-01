from rb_2qb import *
from qm.QuantumMachinesManager import SimulationConfig
from configuration import config
from typing import Optional

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
def retrieve_my_elements(*qe_set):
    assert len(qe_set) == 3, f"My 2 qubit RB should contain only q0, q1 and coupler {qe_set}"
    return qe_set[0], qe_set[1], qe_set[2]


def CNOT(b: Baking, *qe_set: str):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #
    q0, q1, coupler = retrieve_my_elements(*qe_set)

    # Option 1: simple play statement for single qubit gate

    b.play("-Y/2", q1)
    b.align(q1, coupler)
    b.play("CZ", coupler)
    b.align(q0, q1, coupler)
    b.play("Y/2", q1)
    # Option 2: macro required for single qubit gate

    # mY_2(b, q1)
    # b.align(q1, coupler)
    # b.play("CZ", coupler)
    # b.align(q0, q1, coupler)
    # Y_2(b, q1)


def iSWAP(b: Baking, *qe_set: str):
    q0, q1, coupler = retrieve_my_elements(*qe_set)
    alt_set = (q1, q0, coupler)
    b.play("-X/2", q1)
    CNOT(b, *alt_set)
    b.play("-X/2", q1)
    b.play("Y/2", q0)
    CNOT(b, *alt_set)
    b.play("X/2", q1)


def SWAP(b: Baking, *qe_set: str):
    q0, q1, coupler = retrieve_my_elements(*qe_set)
    alt_set = (q1, q0, coupler)
    CNOT(b, *qe_set)
    CNOT(b, *alt_set)
    CNOT(b, *qe_set)


"""
In what follows, q_tgt should be the main target qubit for which should be played the single qubit gate.
qe_set can be a set of additional quantum elements that might be needed to actually compute the gate
(e.g fluxline, trigger, ...). It is then up to the user to use a name convention for elements allowing him to perform
the correct gate to the right target qubit and its associated elements
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


def measure(*measure_args: Optional[Tuple]):
    th1 = declare(fixed, value=0.)
    th2 = declare(fixed, value=0.)
    stream1 = declare_stream()
    stream2 = declare_stream()
    state1 = declare(bool)
    state2 = declare(bool)
    I1 = declare(fixed)
    I2 = declare(fixed)
    d1 = declare(fixed)
    d2 = declare(fixed)
    d3 = declare(fixed)
    d4 = declare(fixed)
    measure('readout', "rr1", None, demod.full('integW1', d1, 'out1'),
            demod.full('integW2', d2, 'out2'))
    measure('readout', "rr2", None, demod.full('integW1', d3, 'out1'),
            demod.full('integW2', d4, 'out2'))
    assign(I1, d1 + d2)
    assign(I2, d3 + d4)
    assign(state1, I1 > th1)
    assign(state2, I2 > th2)
    save(state1, stream1)
    save(state2, stream2)


def stream_macro(stream1, stream2):
    stream1.boolean_to_int().average().save("state1")
    stream2.boolean_to_int().average().save("state2")


nCliffords = range(1, 170, 2)
s = RBTwoQubits(qmm=qmm, config=config, quantum_elements=["q0", "q1", "coupler"],
                N_Clifford=nCliffords, K=1, N_shots=100,
                two_qb_gate_baking_macros=two_qb_gate_macros,
                measure_macro=measure, stream_macro=stream_macro)
sequences = s.sequences
s1 = sequences[0].full_sequence
for h in s1:
    print(len(h), h)

seq = sequences[0]
b = seq.generate_baked_sequence()
print(b.get_Op_length("q0"))
prog = s.qua_prog(b)
print("starting simulation")
job = qmm.simulate(config=config, program=prog, simulate=SimulationConfig(5000))

samples = job.get_simulated_samples()
samples.con1.plot()