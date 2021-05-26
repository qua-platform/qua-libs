import pandas as pd
from bakery.bakery import *
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

# Cayley table corresponding to above Clifford group structure

c1_table = pd.read_csv('c1_cayley_table.csv').to_numpy()[:, 1:]


class RB_one_qubit:
    def __init__(self, config: dict, d_max: int, K: int, qubit: str, cliffords: List = None,
                 cayley_table: np.ndarray = None):
        """
        Class to retrieve easily baked RB sequences and their inverse operations
        :param config Configuration file
        :param d_max Maximum length of desired RB sequence
        :param K Number of RB sequences to be generated
        :param qubit Name of the quantum element designating the qubit
        :param cliffords List of Clifford operations chosen, by default c1_ops defined above
        :param cayley_table Table indicating group structure associated to Cliffords provided
        """

        if cliffords is None:
            cliffords = c1_ops
            cayley_table = c1_table
        self.config = config
        self.cayley_table = cayley_table
        self.cliffords = cliffords
        self.qubit = qubit
        self.K = K
        self.sequences = [Baking] * K
        self.d_max = d_max
        self.state_tracker = [0] * d_max  # Keeps track of all transformations done on qubit state
        self.state_init = 0
        self.revert_ops = [0] * d_max  # Keeps track of inverse op index associated to each sequence
        self.duration_tracker = [0] * d_max  # Keeps track of each Clifford's duration
        self.baked_cliffords = [Baking] * len(self.cliffords)
        for i in range(len(self.cliffords)):
            with baking(self.config) as b2:
                for op in self.cliffords[i]:
                    b2.play(op, self.qubit)
            self.baked_cliffords[i] = b2

        for k in range(self.K):  # Generate K RB sequences of length d_max
            with baking(config) as b:
                for d in range(self.d_max):
                    i = np.random.randint(0, len(self.cliffords))
                    self.duration_tracker[d] = d + 1  # Set the duration to the value of the sequence step

                    # Play the random Clifford
                    random_clifford = self.cliffords[i]
                    for op in random_clifford:
                        b.play(op, self.qubit)
                        self.duration_tracker[d] += 1  # Add additional duration for each pulse played to build Clifford

                    if d == 0:  # Handle the case for qubit set to original/ground state
                        self.state_tracker[d] = self.cayley_table[self.state_init][i]
                    else:  # Get the newly transformed state within th Cayley table based on previous step
                        self.state_tracker[d] = self.cayley_table[self.state_tracker[d - 1]][i]
                    self.revert_ops[d] = self._find_revert_op(self.state_tracker[d])
            self.sequences[k] = b  # Stores all the RB sequences

    def _find_revert_op(self, input_state_index: int):
        """Looks in the Cayley table the operation needed to reset the state to ground state from input state_tracker
        :param input_state_index Index of the current state tracker
        :return index of the next Clifford to apply to invert RB sequence"""
        for i in range(len(self.cliffords)):
            if self.cayley_table[input_state_index][i] == 0:
                return i

    def play_revert_op(self, index: int):
        """Plays an operation resetting qubit in its ground state based on the
        transformation provided by the index in Cayley table
        :param index index of the transformed qubit state"""

        with switch_(index):
            for i in range(len(self.baked_cliffords)):
                with case_(i):
                    self.baked_cliffords[i].run()
