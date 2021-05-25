import pandas as pd
from bakery import *
from qm.qua import *
c1_ops = [  # Clifford operations
    ('I',),
    ('X',),
    ('Y',),
    ('Y', 'X'),

    ('X/2', 'Y/2'),
    ('X/2', '-Y/2'),
    ('-X/2', 'Y/2'),
    ('-X/2', '-Y/2'),
    ('Y/2', 'X/2'),
    ('Y/2', '-X/2'),
    ('-Y/2', 'X/2'),
    ('-Y/2', '-X/2'),

    ('X/2',),
    ('-X/2',),
    ('Y/2',),
    ('-Y/2',),
    ('-X/2', 'Y/2', 'X/2'),
    ('-X/2', '-Y/2', 'X/2'),

    ('X', 'Y/2'),
    ('X', '-Y/2'),
    ('Y', 'X/2'),
    ('Y', '-X/2'),
    ('X/2', 'Y/2', 'X/2'),
    ('-X/2', 'Y/2', '-X/2'),

]


c1_table = pd.read_csv('c1_cayley_table.csv').to_numpy()[:, 1:]  # Cayley table corresponding to above Clifford group structure

# create inverses lists
inverse_list = [np.nonzero(line == 0)[0][0] for line in c1_table]
# print(inverse_list)


def find_revert_op(input_state_index):
    """Looks in the Cayley table the operation needed to reset the state to ground state from input state_tracker"""
    for i in range(len(c1_ops)):
        if c1_table[input_state_index][i] == 0:
            return i


def play_revert_op(index: int, baked_list: list[Baking]):
    """Plays an operation resetting qubit in its ground state based on the
    transformation provided by the index in Cayley table
    :param index index of the transformed qubit state
    :param baked_list list of baking objects containing the waveforms associated to each Clifford"""

    with switch_(index):
        for i in range (len(baked_list)):
            with case_(i):
                baked_list[i].run()


def measure_state(state, I):
    """
    A measurement function depending on the type of qubit.
    This example implementation is typical of a SC qubit measurement (via a dispersive readout)
    :param state: a QUA var where the state will be saved
    :param I: a QUA var containing the demod result
    :return: none
    """
    th = 0
    measure("readout", "rr", None, integration.full("integW1", I))
    assign(state, I > th)


def active_reset(state):

    with if_(state):
        play("X", "qe1")
