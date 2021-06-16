from bakery.bakery import *


rnd_gate_map = {0: "sx", 1: "sy", 2: "sw"}


class XEB:
    def __init__(self, config: dict, m_max: int, qe_list: List[str]):
        """
        Class instance for cross-entropy benchmarking sequence generation.
        :param config Configuration file
        :param m_max Maximum length of XEB sequence
        :param qe_list List of quantum elements used to do the sequence ([qubit1, qubit2, coupler])

        """
        self.config = config
        self.qe_list = qe_list
        self.m_max = m_max
        self.duration_tracker = [0] * m_max
        self.operations_list = {qe: [] for qe in qe_list}
        self.baked_sequence = self.generate_xeb_sequence()

    def generate_xeb_sequence(self):
        rand_seq1 = np.random.randint(3, size=self.m_max)
        rand_seq2 = np.random.randint(3, size=self.m_max)
        q1 = self.qe_list[0]
        q2 = self.qe_list[1]
        coupler = self.qe_list[2]
        with baking(self.config) as b:
            i = 0
            for rnd1, rnd2 in zip(rand_seq1, rand_seq2):
                b.align(q1, q2, coupler)
                b.play(rnd_gate_map[rnd1], q1)
                b.play(rnd_gate_map[rnd2], q2)
                b.align(q1, q2, coupler)
                b.play("coupler_op", coupler)
                self.operations_list[q1].append(rnd_gate_map[rnd1])
                self.operations_list[q2].append(rnd_gate_map[rnd2])
                self.operations_list[q2].append("coupler_op")

                self.duration_tracker[i] = b.get_current_length(coupler)
                i += 1
        return b
