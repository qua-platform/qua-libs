import os
from qualang_tools.bakery.bakery import Baking
from configuration import *
from .. import TwoQubitRb

from ..verification import CommandRegistry
from ..verification import SequenceTracker


def test_all_verification():
    """
    Tests that a variety of random sequences are tracked, successfully verified
    by unitary-based simulation, and output to file. Tests that mapping from
    command-id to gate is also properly saved to file.
    """

    for i in range(3):
        cr = CommandRegistry()
        st = SequenceTracker(command_registry=cr)

        def bake_phased_xz(baker: Baking, q, x, z, a):
            cr.register_phase_xz(q, x, z, a)

        def bake_cz(baker: Baking, q1, q2):
            cr.register_cz()

        def bake_cnot(baker: Baking, q1, q2):
            cr.register_cnot(q=q1)

        def prep():
            pass

        def meas():
            pass

        cz_generator = {"CZ": bake_cz}
        cnot_generator = {"CNOT": bake_cnot}
        cz_cnot_generator = {"CZ": bake_cz, "CNOT": bake_cnot}

        bake_2q_gate_generator = [cz_generator, cnot_generator, cz_cnot_generator][i]

        rb = TwoQubitRb(config, bake_phased_xz, bake_2q_gate_generator, prep, meas,
                        verify_generation=False, interleaving_gate=None,
                        command_registry=cr, sequence_tracker=st)
        repeats = 10
        depth = 10
        for _ in range(repeats):
            sequence = rb._gen_rb_sequence(depth)
            st.make_sequence(sequence)

        st.verify_sequences()

        parent_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        cr.save_to_file(parent_dir / 'commands.txt')
        st.save_to_file(parent_dir / 'sequences.txt')


if __name__ == '__main__':
    test_all_verification()