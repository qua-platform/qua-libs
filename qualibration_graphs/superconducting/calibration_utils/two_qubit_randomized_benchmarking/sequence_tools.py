# %%

import numpy as np
import pickle

from tqdm import tqdm
from importlib.resources import files
from typing import List, Tuple

Op = Tuple[str, str]


instruct_pkl = "2q_Clifford_gen_CNOT_instruct.pkl"
unitary_pkl = "2q_Clifford_gen_CNOT_unitary.pkl"


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


def split_ops(ops: List[Op]) -> List[List[Op]]:
    """
    Expect order: (0,1) pairs, optionally interleaved with '01'.
    Enforce: no consecutive single-qubit ops on the same qubit, and singles
    must appear as '0' followed immediately by '1'.
    Output groups: [op0, op1] or ['01'].
    """
    groups: List[List[Op]] = []
    i = 0
    n = len(ops)

    while i < n:
        gate, tgt = ops[i]
        if tgt == "01":
            # the CNOT is saved as the following form in the list
            groups.append(
                [("I", "0"), ("I", "1"), ("CNOT", "01"), ("I", "0"), ("I", "1")]
            )
            i += 1
        elif tgt == "0":
            if i + 1 >= n:
                raise ValueError(
                    f"Expected ('*','1') after index {i}, got end of list."
                )
            _, nxt = ops[i + 1]
            if nxt != "1":
                raise ValueError(
                    f"Expected ('*','1') after ('{gate}','0') at index {i}, got ('*','{nxt}')."
                )
            groups.append([ops[i], ops[i + 1]])
            i += 2
        elif tgt == "1":
            raise ValueError(
                f"Unexpected single on '1' at index {i}; singles must be '0' then '1'."
            )
        else:
            raise ValueError(f"Unexpected target tag {tgt!r} at index {i}.")

    return groups


# def generate_sequence_list(depth):
#     """
#     for a given depth, generate ONE random sequence containing 2q clifford gates at a certain depth
#     input:
#         depth: the depth of the sequence of the clifford gates
#     return:
#         instruct_integers: a list of integers; each corresponds to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
#     """

#     instruct_list, unitary_list = load_pkls()

#     if depth == 0:
#         instruct = []
#         instruct.extend(instruct_list[0])
#         instruct_integers = instruct_to_integer(instruct)
#         return instruct_integers
#     sequence_ints = np.random.randint(11520, size=depth).tolist()

#     instruct = []
#     unitaries = []
#     for d_ in range(depth):
#         instruct.extend(instruct_list[sequence_ints[d_]])
#         unitaries.append(unitary_list[sequence_ints[d_]])
#     if len(unitaries) == 1:
#         unitary = unitaries[0]
#     else:
#         unitary = np.linalg.multi_dot(unitaries[::-1])
#     for c_ in range(11520):
#         if abs((unitary @ unitary_list[c_]).trace()) > 3.95:
#             # set a bar for judging whether circuit[c_] is the inverse gate; ideally should be 4
#             # 3.95 rather than 4 to account for any floating point error
#             break
#     sequence_ints.append(c_)
#     instruct.extend(instruct_list[c_])
#     instruct_integers = instruct_to_integer(instruct)

#     return instruct_integers


def generate_sequence_list_interleaved(depth, interleaved_instruct=None):
    """
    for a given depth, generate ONE random sequence containing 2q clifford gates at a certain depth
    interleaved by CNOT
    return:
        instruct_integers: a list of integers each corresponding to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
    """

    instruct_list, unitary_list = load_pkls()
    if interleaved_instruct:
        split_instruct_list = split_ops(interleaved_instruct)
        interleaved_idx = [instruct_list.index(s) for s in split_instruct_list]
        interleaved_unitary = [unitary_list[i] for i in interleaved_idx]
        if len(interleaved_unitary) == 1:
            interleaved_unitary = interleaved_unitary[0]
        else:
            interleaved_unitary = np.linalg.multi_dot(interleaved_unitary[::-1])

    if depth == 0:
        instruct = []
        instruct.extend(instruct_list[0])
        instruct_integers = instruct_to_integer(instruct)
        return instruct_integers

    sequence_ints = np.random.randint(11520, size=depth).tolist()
    instruct = []
    unitaries = []
    for d_ in range(depth):
        instruct.extend(instruct_list[sequence_ints[d_]])
        unitaries.append(unitary_list[sequence_ints[d_]])
        if interleaved_instruct:
            instruct.extend(interleaved_instruct)
            unitaries.append(interleaved_unitary)
    if len(unitaries) == 1:
        unitary = unitaries[0]
    else:
        unitary = np.linalg.multi_dot(unitaries[::-1])
    for c_ in range(11520):
        if abs((unitary @ unitary_list[c_]).trace()) > 3.95:
            # set a bar for judging whether circuit[c_] is the inverse gate
            # 3.95 rather than 4 to account for any floating point error; ideally should be 4
            break
    sequence_ints.append(c_)
    instruct.extend(instruct_list[c_])
    instruct_integers = instruct_to_integer(instruct)

    return instruct_integers


# def pre_generate_sequence(number_of_sequences, depth_list):
#     """
#     for a given depth list, generate random sequences containing 2q clifford gates for each depth
#     input:
#         number_of_sequences: number of random realizations
#         depth_list: a list of clifford depths you want to test
#     return:
#         sequence_list: a list of integers; each integer corresponds to a specific pulse (see "run=two-qubit-rb-CR-CNOT.py")
#                        this list stacks the instruction lists for len(depth_list)*number_of_sequences random sequences
#         len_list: a size-len(depth_list)*number_of_sequences integer list
#                   recording the length of each random sequence
#                   tells qm where to start and stop when reading sequence_list
#     """
#     sequence_list = []
#     len_list = []
#     for ns_ in range(number_of_sequences):
#         for dl_ in range(len(depth_list)):
#             new_list = generate_sequence_list(depth_list[dl_])
#             sequence_list.extend(new_list)
#             len_list.append(len(new_list))

#     return sequence_list, len_list


def pre_generate_sequence_interleaved(
    number_of_sequences,
    depth_list,
    interleaved_instruct=None,
):
    sequence_list = []
    len_list = []
    for ns_ in tqdm(range(number_of_sequences), desc="Generating sequences"):
        for dl_ in range(len(depth_list)):
            new_list = generate_sequence_list_interleaved(
                depth_list[dl_], interleaved_instruct
            )
            sequence_list.extend(new_list)
            len_list.append(len(new_list))

    return sequence_list, len_list


def load_pkls():
    data_dir = files(__package__)
    instruct_list = pickle.loads((data_dir / instruct_pkl).read_bytes())
    unitary_list = pickle.loads((data_dir / unitary_pkl).read_bytes())

    return instruct_list, unitary_list
