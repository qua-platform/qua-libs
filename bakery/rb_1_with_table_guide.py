import numpy as np
import pandas as pd
c1_ops = [  # these are the cliffords
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


c1_table = pd.read_csv('c1_cayley_table.csv').to_numpy()[:, 1:]  # this is the cayley table that corresponds to these cliffords
print(c1_table)

# for example, Y*Y is:
print(c1_table[2][2])
# also X/2 * -X/2 is I:
print(c1_table[12][13])
# todo: figure out what is the order of composition vs. row/column

# create inverses lists
inverse_list = [np.nonzero(line == 0)[0][0] for line in c1_table]
print(inverse_list)
#import qutip as qt

def find_revert_op(input_state_index):
    """Looks in the Cayley table the operation needed to reset the state to ground state from input state_tracker"""
    for i in range(len(c1_ops)):
        if c1_table[input_state_index][i] == 0:
            return i