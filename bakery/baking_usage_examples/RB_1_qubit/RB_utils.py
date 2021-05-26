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


class RBOneQubit:
    def __init__(self, config: dict, d_max: int, K: int, qubit: str):
        """
        Class to retrieve easily baked RB sequences and their inverse operations
        :param config Configuration file
        :param d_max Maximum length of desired RB sequence
        :param K Number of RB sequences
        :param qubit Name of the quantum element designating the qubit
        """

        self.sequences = [RBSequence(config, d_max, qubit)]* K
        self.inverse_ops = [seq.revert_ops for seq in self.sequences]
        self.duration_trackers = [seq.duration_tracker for seq in self.sequences]
        self.baked_sequences = [seq.sequence for seq in self.sequences]


def find_revert_op(input_state_index: int):
    """Looks in the Cayley table the operation needed to reset the state to ground state from input state_tracker
    :param input_state_index Index of the current state tracker
    :return index of the next Clifford to apply to invert RB sequence"""
    for i in range(len(c1_ops)):
        if c1_table[input_state_index][i] == 0:
            return i


class RBSequence:
    def __init__(self, config: dict, d_max: int, qubit: str):
        self.d_max = d_max
        self.config = config
        self.qubit = qubit
        self.state_tracker = [0] * d_max  # Keeps track of all transformations done on qubit state
        self.state_init = 0
        self.revert_ops = [0] * d_max  # Keeps track of inverse op index associated to each sequence
        self.duration_tracker = [0] * d_max  # Keeps track of each Clifford's duration
        self.baked_cliffords = self.generate_cliffords()  # List of baking objects for running Cliffords
        self.sequence = self.generate_RB_sequence()  # Store the RB sequence

    def play_revert_op(self, index: int):
        """Plays an operation resetting qubit in its ground state based on the
        transformation provided by the index in Cayley table
        :param index index of the transformed qubit state"""

        with switch_(index):
            for i in range(len(self.baked_cliffords)):
                with case_(i):
                    self.baked_cliffords[i].run()

    def generate_cliffords(self):

        baked_clifford = []
        for i in range(len(c1_ops)):
            with baking(self.config) as b2:
                for op in c1_ops[i]:
                    b2.play(op, self.qubit)
            baked_clifford.append(b2)
        return baked_clifford

    def generate_RB_sequence(self):

        with baking(self.config) as b:
            for d in range(self.d_max):
                i = np.random.randint(0, len(c1_ops))
                self.duration_tracker[d] = d + 1  # Set the duration to the value of the sequence step

                # Play the random Clifford
                random_clifford = c1_ops[i]
                for op in random_clifford:
                    b.play(op, self.qubit)
                    self.duration_tracker[d] += 1  # Add additional duration for each pulse played to build Clifford

                if d == 0:  # Handle the case for qubit set to original/ground state
                    self.state_tracker[d] = c1_table[self.state_init][i]
                else:  # Get the newly transformed state within th Cayley table based on previous step
                    self.state_tracker[d] = c1_table[self.state_tracker[d - 1]][i]
                self.revert_ops[d] = find_revert_op(self.state_tracker[d])
        return b
