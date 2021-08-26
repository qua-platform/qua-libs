from rb_2qb import *
from configuration import config

qmm = QuantumMachinesManager()


def measure():
    pass


s = RBTwoQubits(qmm=qmm, config=config, max_length=20, K=1, two_qb_gate_baking_macros=two_qb_gate_macros,
                measure_macro=measure, quantum_elements=["qe1", "qe2"])
sequences = s.sequences
b = sequences[0].generate_baked_sequence()
print(sequences[0].full_sequence)