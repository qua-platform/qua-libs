from rb_2qb import *
from configuration import config
from qm.QuantumMachinesManager import SimulationConfig
from entropylab_qpudb import Resolver

qmm = QuantumMachinesManager()

"""
To generate the 4 classes allowing Clifford selection (see Supplementary info of this paper:
https://arxiv.org/pdf/1210.7011.pdf), we require the user to complete qubit gates macros according
to their own native set of qubit gates, that is perform the appropriate decomposition and convert the pulse sequence
into a sequence of baking play statements (amounts to similar structure as QUA, just add the prefix b. before every `
statement).
Example :
in QUA you would have for a CZ operation:
    play("CZ", "coupler")
in the macros below you write instead:
    b.play("CZ", "coupler")
"""
# Define here quantum elements required to compute the macros
q0 = "q0"
q1 = "q1"
T1 = 4  # Assumed T1 for qubit decay
"""
Define here a Resolver instance methods to wrap a correspondence between qubits to be characterized and quantum elements
in the configuration.
The Resolver allows you to retrieve easily the names of the quantum elements associated to your target qubits, couplers,
and other elements you might need to carry (fluxlines, etc...).

In this example, we use flux tuneable transmon qubits, with a CPHASE (CZ) interaction carried through a coupler
"""


class RBTwoQubitResolver(Resolver):
    def __init__(self, aliases: Union[dict, None] = None):
        super().__init__(aliases)

    def q(self, qubit, channel=None):

        if channel == 'xy':
            return f'q{self._aliases.get(qubit, qubit)}_xy'
        elif channel == 'z':
            return f'q{self._aliases.get(qubit, qubit)}_z'
        elif channel is None:
            return f'q{self._aliases.get(qubit, qubit)}'
        elif channel:
            raise ValueError(f"channel {channel} is specified but not xy, z")

    def res(self, qubit):
        return f'rr{self.aliases.get(qubit, qubit)}'

    def coupler(self, qubit1, qubit2):
        coupler_index = "".join([str(q) for q in sorted([self._aliases.get(qubit1, qubit1),
                                                         self._aliases.get(qubit2, qubit2)])])
        return f'coupler{coupler_index}'

    @property
    def aliases(self):
        return self._aliases


aliases = {
    q0: 0,
    q1: 1
}
resolve = RBTwoQubitResolver(aliases)


def get_IF(qubit: str):
    return config["elements"][qubit]["intermediate_frequency"]

# Macro for native qubit gate (shall not be part of the two-qubit gate macro dictionary


def echo_mZX_2(b_seq: Baking, ctrl: str, tgt: str):
    """
    The native two-qubit gate shall always contain align before and after the playing statements
    """
    Delta = get_IF(resolve.q(tgt)) - get_IF(resolve.q(ctrl))
    # π pulse
    X(b_seq, ctrl)

    b_seq.align(resolve.q(ctrl), resolve.q(tgt))
    # CR/2 tone
    b_seq.set_detuning(resolve.q(ctrl), Delta)
    b_seq.play("CR_pi_over_4", ctrl)
    b_seq.set_detuning(resolve.q(ctrl), 0)

    b_seq.align(resolve.q(ctrl), resolve.q(tgt))
    # π pulse
    X(b_seq, ctrl)
    b_seq.align(resolve.q(ctrl), resolve.q(tgt))

    # -CR/2 tone
    b_seq.set_detuning(resolve.q(ctrl), Delta)
    b_seq.play("CR_pi_over_4", ctrl, amp=-1.)
    b_seq.set_detuning(resolve.q(ctrl), 0)

    b_seq.align(resolve.q(ctrl), resolve.q(tgt))

# Baking Macros required for the two qubit gates to go in the macro dictionary to initialize the RBTwoQubit instance
# Shall contain macros for CNOT, SWAP, and iSWAP

# Single qubit macros to be added in single qubit macros dictionary to initialize the RBTwoQubit instance
# Should contain I, X, Y, X/2, -X/2, Y/2, -Y/2


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


""" 
The two dictionaries for macros given to the RBTwoQubit class shall be in the exact following format.
Note that we only require gates that are used to generate the Clifford sequence (CZ defined above is only used
to compute the native qubit gate, but it does not serve for Clifford class selection as shown in reference 
https://arxiv.org/pdf/1210.7011.pdf)

"""

two_qb_gate_macros = {
    "-ZX_pi_over_2": echo_mZX_2
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


# Define the QUA program according to the setup measurement scheme. Shall include at least baking object as argument
# to use the method b_seq.run(). Here, we also pass as a parameter the sampling number for each circuit


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
        d3 = declare(fixed)

        with for_(n, 0, n < N_shots, n + 1):
            # Wait for qubits to decay
            wait(T1)

            b_seq.run()
            align()
            measure('readout', resolve.res(q0), None, demod.full('integW1', d1, 'out1'))
            measure('readout', resolve.res(q1), None, demod.full('integW1', d3, 'out1'),)

            assign(I1, d1)
            assign(I2, d3)
            assign(state1, I1 > th1)
            assign(state2, I2 > th2)
            save(state1, stream1)
            save(state2, stream2)
        with stream_processing():
            stream1.boolean_to_int().save_all("state0")
            stream2.boolean_to_int().save_all("state1")

    return prog


# Define here parameters characterizing the experiment (number of operations to be played, number of random sequence to
# be played)

n_max = 10  # Maximum length of random sequence, i.e. maximum number of Clifford composing sequence
step = 1
nCliffords = range(n_max-1, n_max, step)
N_sequences = 1  # Specify all truncations you want to generate to realize the fitting
N_shots = 1

# Define class instance to generate random sequences
RB_exp = RBTwoQubits(qmm=qmm, config=config,
                     N_Clifford=nCliffords, N_sequences=N_sequences,
                     two_qb_gate_baking_macros=two_qb_gate_macros,
                     single_qb_macros=single_qb_gate_macros, qubit_register=aliases)

# Here you can check which sequences have been generated (in terms of Clifford operations)
sequences = RB_exp.sequences
s1 = sequences[0].full_sequence
# Uncomment lines below to see random sequence in terms of Cliffords
# for Cl in s1:
#     print(len(Cl), Cl)
cayley_table = sequences[0].cayley_table
inverse_list = sequences[0].inverse_index_list
inverse_Clifford_list = sequences[0].inverse_Clifford_list


# Necessary step:
# Retrieve the longest baked waveform to perform overriding with the run function, "baked_reference"
baked_reference = RB_exp.baked_reference

print("length of longest random sequence:", baked_reference.get_op_length())

# RB_exp.run(prog=qua_prog(baked_reference, N_shots=N_shots))
# P_00, Average_Error_per_Clifford = RB_exp.retrieve_results(stream_name_0="state0",
#                                                            stream_name_1="state1",
#                                                            N_shots=N_shots)
# RB_exp.plot()


# Additional lines to plot the pulse sequence
#
# job = qmm.simulate(RB_exp.config, qua_prog(b_seq=baked_reference, N_shots=N_shots),
#                    simulate=SimulationConfig(12000))
# samples = job.get_simulated_samples()
# samples.con1.plot()
