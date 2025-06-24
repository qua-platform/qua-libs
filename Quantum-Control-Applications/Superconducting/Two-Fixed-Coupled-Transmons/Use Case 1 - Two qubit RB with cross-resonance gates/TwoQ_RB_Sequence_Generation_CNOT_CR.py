import numpy as np
import cirq as cirq
import pickle
import time


def instruct_to_integer(instruct):
    """
    input:
        instruct: a list of instructions such as [('x90', 0), ('I', 1), ('CNOT', 01)]
    return:
        instruct_integers: a list of integers each corresponding to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
                            the example above gives you [1, 49]
    """

    instruct_integers = []
    d_ = 0
    while d_ < len(instruct):
        if instruct[d_][1] == "01":
            instruct_integers.append(49)
            d_ += 1
        else:
            if instruct[d_][0] == "I":
                X = 0
            if instruct[d_][0] == "x90":
                X = 1
            if instruct[d_][0] == "-x90":
                X = 2
            if instruct[d_][0] == "x180":
                X = 3
            if instruct[d_][0] == "y90":
                X = 4
            if instruct[d_][0] == "-y90":
                X = 5
            if instruct[d_][0] == "y180":
                X = 6
            if instruct[d_ + 1][0] == "I":
                Y = 0
            if instruct[d_ + 1][0] == "x90":
                Y = 1
            if instruct[d_ + 1][0] == "-x90":
                Y = 2
            if instruct[d_ + 1][0] == "x180":
                Y = 3
            if instruct[d_ + 1][0] == "y90":
                Y = 4
            if instruct[d_ + 1][0] == "-y90":
                Y = 5
            if instruct[d_ + 1][0] == "y180":
                Y = 6
            instruct_integers.append((X) + (Y) * 7)
            d_ = d_ + 2
    return instruct_integers


def generate_sequence_list(depth):
    """
    for a given depth, generate ONE random sequence containing 2q clifford gates at a certain depth
    input:
        depth: the depth of the sequence of the clifford gates
    return:
        instruct_integers: a list of integers; each corresponds to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
    """

    with open("2q_Clifford_gen_CNOT_instruct.pkl", "rb") as file:
        instruct_list = pickle.load(file)
    with open("2q_Clifford_gen_CNOT_circuit_cirq14.pkl", "rb") as file:
        circuit_list = pickle.load(file)
        # based on cirq=1.4.1
    # with open('2q_Clifford_gen_CNOT_circuit_cirq12.pkl', 'rb') as file:
    #     circuit_list = pickle.load(file)
    #     # based on cirq=1.2.0
    with open("2q_Clifford_gen_CNOT_unitary.pkl", "rb") as file:
        unitary_list = pickle.load(file)

    if depth == 0:
        circuit = cirq.Circuit()
        instruct = []
        circuit.append(circuit_list[0])
        instruct.extend(instruct_list[0])
        instruct_integers = instruct_to_integer(instruct)
        return instruct_integers
    # np.random.seed(seed = int(time.time()%1*1e8)) # can set the seed here, or in
    sequence_ints = np.random.randint(11520, size=depth).tolist()
    # sequence_ints = np.random.randint(576, size = depth).tolist() # for parallel single-qubit RB

    circuit = cirq.Circuit()
    instruct = []
    for d_ in range(depth):
        circuit.append(circuit_list[sequence_ints[d_]])
        instruct.extend(instruct_list[sequence_ints[d_]])
    unitary = cirq.unitary(circuit)
    for c_ in range(11520):
        if abs((unitary @ unitary_list[c_]).trace()) > 3.95:
            # set a bar for judging whether circuit[c_] is the inverse gate; ideally should be 4
            # 3.95 rather than 4 to account for any floating point error
            break
    circuit.append(circuit_list[c_])
    sequence_ints.append(c_)
    instruct.extend(instruct_list[c_])
    instruct_integers = instruct_to_integer(instruct)

    return instruct_integers


def generate_sequence_list_interleaved(depth):
    """
    for a given depth, generate ONE random sequence containing 2q clifford gates at a certain depth
    interleaved by CNOT
    return:
        instruct_integers: a list of integers each corresponding to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
    """

    with open("2q_Clifford_gen_CNOT_instruct.pkl", "rb") as file:
        instruct_list = pickle.load(file)
    with open("2q_Clifford_gen_CNOT_circuit_cirq14.pkl", "rb") as file:
        circuit_list = pickle.load(file)
    with open("2q_Clifford_gen_CNOT_unitary.pkl", "rb") as file:
        unitary_list = pickle.load(file)
    # np.random.seed(seed = int(time.time()%1*1e8))

    if depth == 0:
        circuit = cirq.Circuit()
        instruct = []
        circuit.append(circuit_list[0])
        instruct.extend(instruct_list[0])
        instruct_integers = instruct_to_integer(instruct)
        return instruct_integers

    sequence_ints = np.random.randint(11520, size=depth).tolist()
    circuit = cirq.Circuit()
    instruct = []
    for d_ in range(depth):
        circuit.append(circuit_list[sequence_ints[d_]])
        instruct.extend(instruct_list[sequence_ints[d_]])
        circuit.append(circuit_list[576])  # the 576th gate is CNOT
        instruct.extend([("CNOT", "01")])
    unitary = cirq.unitary(circuit)
    for c_ in range(11520):
        if abs((unitary @ unitary_list[c_]).trace()) > 3.95:
            # set a bar for judging whether circuit[c_] is the inverse gate
            # 3.95 rather than 4 to account for any floating point error; ideally should be 4
            break
    circuit.append(circuit_list[c_])
    sequence_ints.append(c_)
    instruct.extend(instruct_list[c_])
    instruct_integers = instruct_to_integer(instruct)

    return instruct_integers


def pre_generate_sequence(number_of_sequences, depth_list):
    """
    for a given depth list, generate random sequences containing 2q clifford gates for each depth
    input:
        number_of_sequences: number of random realizations
        depth_list: a list of clifford depths you want to test
    return:
        sequence_list: a list of integers; each integer corresponds to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
                       this list stacks the instruction lists for len(depth_list)*number_of_sequences random sequences
        len_list: a size-len(depth_list)*number_of_sequences integer list
                  recording the length of each random sequence
                  tells qm where to start and stop when reading sequence_list
    """
    sequence_list = []
    len_list = []
    for ns_ in range(number_of_sequences):
        for dl_ in range(len(depth_list)):
            new_list = generate_sequence_list(depth_list[dl_])
            sequence_list.extend(new_list)
            len_list.append(len(new_list))

    return sequence_list, len_list


def pre_generate_sequence_interleaved(number_of_sequences, depth_list):

    sequence_list = []
    len_list = []
    for ns_ in range(number_of_sequences):
        for dl_ in range(len(depth_list)):
            new_list = generate_sequence_list_interleaved(depth_list[dl_])
            sequence_list.extend(new_list)
            len_list.append(len(new_list))

    return sequence_list, len_list
