from rb_2qb import *
from configuration import config
from qm.QuantumMachinesManager import SimulationConfig
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
# Define here quantum elements required to compute the macros
q0 = "q0"
q1 = "q1"
coupler = "coupler"
# Baking Macros required for two qubit gates

# Here is my assumed native two qubit gate


def CZ(b_seq: Baking):
    """
    The native two-qubit gate shall always contain align before and after the playing statements
    """
    b_seq.align(q0, q1, coupler)

    b_seq.play("CZ", coupler)

    b_seq.align(q0, q1, coupler)


def CNOT(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #

    # Option 1: simple play statement for single qubit gate

    b_seq.play("-Y/2", tgt)
    CZ(b_seq)
    b_seq.play("Y/2", tgt)
    # Option 2: macro required for single qubit gate

    # mY_2(b_seq, tgt)
    # b_seq.align(tgt, coupler)
    # b_seq.play("CZ", coupler)
    # b_seq.align(ctrl, tgt, coupler)
    # Y_2(b_seq, tgt)


def iSWAP(b_seq: Baking):

    b_seq.play("-X/2", q1)
    CNOT(b_seq, q1, q0)
    b_seq.play("-X/2", q1)
    CZ(b_seq)
    b_seq.play("Y/2", q0)
    b_seq.play("X/2", q1)


def SWAP(b_seq: Baking):
    CNOT(b_seq, q0, q1)
    CNOT(b_seq, q1, q0)
    CNOT(b_seq, q0, q1)


"""
In  what follows, q_tgt should be the maintarget qubit for which should be played the single qubit gate.
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


def qua_prog(b_seq: Baking, N_shots: int):
    with program() as prog:
        n = declare(int)
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
        with for_(n, 0, n < N_shots, n+1):
            wait(4, q0, q1)  # Wait for qubits to decay
            b_seq.run()
            align()
            measure('readout', "rr1", None, demod.full('integW1', d1, 'out1'),
                    demod.full('integW2', d2, 'out2'))
            measure('readout', "rr2", None, demod.full('integW1', d3, 'out1'),
                    demod.full('integW2', d4, 'out2'))

            assign(I1, d1+d2)
            assign(I2, d3+d4)
            assign(state1, I1 > th1)
            assign(state2, I2 > th2)
            save(state1, stream1)
            save(state2, stream2)
        with stream_processing():
            stream1.boolean_to_int().average().save("state1")
            stream2.boolean_to_int().average().save("state2")

    return prog


n_max = 2
step = 10
nCliffords = range(1, n_max, step)
N_sequences = 5
print(nCliffords)
RB_exp = RBTwoQubits(qmm=qmm, config=config,
                N_Clifford=nCliffords, K=N_sequences,
                two_qb_gate_baking_macros=two_qb_gate_macros,
                qubits=("q0", "q1"))
sequences = RB_exp.sequences
s1 = sequences[0].full_sequence
# Uncomment lines below to see random sequence in terms of Cliffords
for h in s1:
    print(len(h), h)

# Retrieve here the longest baked waveform to perform overriding with the run function
baked_reference = RB_exp.baked_reference

job = qmm.simulate(RB_exp.config, qua_prog(b_seq=baked_reference, N_shots=100), simulate=SimulationConfig(12000))
samples = job.get_simulated_samples()
samples.con1.plot()
print("reference", baked_reference.get_Op_length("q0"))
# RB_exp.run(prog=qua_prog(baked_reference, 100))
#
# results_list = RB_exp.results
# jobs = RB_exp.job_list

