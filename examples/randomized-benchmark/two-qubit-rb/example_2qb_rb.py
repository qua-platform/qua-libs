from rb_2qb import *
from configuration import config
from qm.QuantumMachinesManager import SimulationConfig
from entropylab_qpudb import Resolver

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
rr1 = "rr1"
rr2 = "rr2"
"""
Define here your Resolver methods such that 
qubit names "q0" and "q1" are mapped to the correct quantum elements in input config associated 
to the two qubits you want to benchmark.
The Resolver allows you to retrieve easily the names of the quantum elements associated to your target qubits, couplers,
and other elements you might need to carry (fluxlines, etc...).

In this example, we have the simplest case where "q0" and "q1" are defined as the two qubits directly in the config,
and there is one single coupler named "coupler". Readout resonators coupled to each qubit are named "rr1" and "rr2"
"""


class RBTwoQubitResolver(Resolver):
    def __init__(self, aliases: Union[dict, None] = None):
        super().__init__(aliases)

    def q(self, qubit, channel=None):
        if qubit == "q0":
            """
            Put here all commands related to the targeted channel (eg. using two different elements for addressing the 
            qubit with two distinct frequencies, this can be specified through the channel argument, or any additional 
            that might be required)
            """
            return qubit

        elif qubit == "q1":
            return qubit
        else:
            return qubit

    def res(self, resonator):
        if resonator == "rr1":
            return resonator
        elif resonator == "rr2":
            return resonator

    def coupler(self, qubit1, qubit2):
        if qubit1 == 'q0' and qubit1 == 'q1':
            return "coupler" # Return name of coupling element for your set of qubits
        else:
            return coupler

    @property
    def aliases(self):
        return self._aliases


# Baking Macros required for two qubit gates

# Here is my assumed native two qubit gate


resolve = RBTwoQubitResolver()


def CZ(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    """
    The native two-qubit gate shall always contain align before and after the playing statements
    """
    b_seq.align(resolve.q(ctrl), resolve.q(tgt), resolve.coupler(ctrl, tgt))

    b_seq.play("CZ", resolve.coupler(ctrl, tgt))

    b_seq.align(resolve.q(ctrl), resolve.q(tgt), resolve.coupler(ctrl, tgt))


def CNOT(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    # Map your pulse sequence for performing a CNOT using baking play statements
    #

    # Option 1: simple play statement for single qubit gate

    mY_2(b_seq, tgt)
    CZ(b_seq, ctrl, tgt)
    Y_2(b_seq, tgt)


def iSWAP(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    mX_2(b_seq, tgt)
    CNOT(b_seq, tgt, ctrl)
    mX_2(b_seq, tgt)
    CZ(b_seq, ctrl, tgt)
    Y_2(b_seq, ctrl)
    X_2(b_seq, tgt)


def SWAP(b_seq: Baking, ctrl: str = "q0", tgt: str = "q1"):
    CNOT(b_seq, ctrl, tgt)
    CNOT(b_seq, tgt, ctrl)
    CNOT(b_seq, ctrl, tgt)


"""
In  what follows, q_tgt should be the main target qubit for which should be played the single qubit gate.
If other elements shall be used to compute a single qubit gate, it is possible to modify/add methods related to the resolver
to retrieve easily each element required.
"""


def I(b: Baking, q_tgt):
    b.play("I", resolve.q(q_tgt))


def X(b: Baking, q_tgt):
    b.play("X", resolve.q(q_tgt))


def Y(b: Baking, q_tgt):
    b.play("Y", resolve.q(q_tgt))


def X_2(b: Baking, q_tgt):
    b.play("X/2", resolve.q(q_tgt))


def Y_2(b: Baking, q_tgt):
    b.play("Y/2", resolve.q(q_tgt))


def mX_2(b: Baking, q_tgt):
    b.play("-X/2", resolve.q(q_tgt))


def mY_2(b: Baking, q_tgt):
    b.play("-Y/2", resolve.q(q_tgt))


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
        with for_(n, 0, n < N_shots, n + 1):
            wait(4, q0, q1)  # Wait for qubits to decay
            b_seq.run()
            align()
            measure('readout', resolve.res(rr1), None, demod.full('integW1', d1, 'out1'),
                    demod.full('integW2', d2, 'out2'))
            measure('readout', resolve.res(rr2), None, demod.full('integW1', d3, 'out1'),
                    demod.full('integW2', d4, 'out2'))

            assign(I1, d1 + d2)
            assign(I2, d3 + d4)
            assign(state1, I1 > th1)
            assign(state2, I2 > th2)
            save(state1, stream1)
            save(state2, stream2)
        with stream_processing():
            stream1.boolean_to_int().save_all("state0")
            stream2.boolean_to_int().save_all("state1")

    return prog


n_max = 2
step = 10
nCliffords = range(1, n_max, step)
N_sequences = 5
N_shots = 100
print(nCliffords)
RB_exp = RBTwoQubits(qmm=qmm, config=config,
                     N_Clifford=nCliffords, N_sequences=N_sequences,
                     two_qb_gate_baking_macros=two_qb_gate_macros,
                     single_qb_macros=single_qb_gate_macros, qubit_register=("q0", "q1"))
sequences = RB_exp.sequences
s1 = sequences[0].full_sequence
# Uncomment lines below to see random sequence in terms of Cliffords
for Cl in s1:
    print(len(Cl), Cl)

# Retrieve here the longest baked waveform to perform overriding with the run function
baked_reference = RB_exp.baked_reference

# job = qmm.simulate(RB_exp.config, qua_prog(b_seq=baked_reference, N_shots=100), simulate=SimulationConfig(12000))
# samples = job.get_simulated_samples()
# samples.con1.plot()
# print("reference", baked_reference.get_Op_length("q0"))

RB_exp.run(prog=qua_prog(baked_reference, N_shots=N_shots))
P_00, Average_Error_per_Clifford = RB_exp.retrieve_results(stream_name_0="state0", stream_name_1="state1", N_shots=N_shots)
RB_exp.plot()
