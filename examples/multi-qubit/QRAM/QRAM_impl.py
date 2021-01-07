from qm.qua import *
import numpy as np
from configuration import config
from qm import QuantumMachinesManager, SimulationConfig, LoopbackInterface


class CavityMode:
    def __init__(self, label, frequency, transmon):
        self.id = id(self)
        self.label = label
        self.frequency = frequency
        self.transmon = transmon

    def initialize(self, number):
        self.create_fock_state(number)

    def create_fock_state(self, number):
        if number == 0:
            return
        Ts = 10  # duration of the swap operation pi/2g_0
        for i in range(1, number + 1):
            self.transmon.x_pi()
            wait(int(Ts / np.sqrt(i)), "transmon")


class Transmon:
    def __init__(self, label, frequency):
        self.label = label
        self.frequency = frequency

    def initialize(self):
        self.x_pi()

    def x_pi(self, freq=None):
        if freq:
            update_frequency("transmon", freq)
        else:
            update_frequency("transmon", self.frequency)
        play("X", "transmon")

    def swap(self, freq1, freq2):
        update_frequency("transmon", freq1)
        play("SWAP", "transmon")
        update_frequency("transmon_duplicate", freq2)
        play("SWAP", "transmon_duplicate")

    def cswap(self, freq):
        update_frequency("transmon", freq)
        play("CSWAP", "transmon")


class QuantumRouter:
    def __init__(self, top, routing, left, right):
        self.id = id(self)
        self.top = top
        self.routing = routing
        self.right = right
        self.left = left

    def __repr__(self):
        return f"top={self.top},routing={self.routing},right=[{self.right}],left=[{self.left}]"

    def initialize(self):
        self.swap(self.top, self.routing)

    def downstream(self):
        self.cswap(self.routing, self.top, self.right)
        self.swap(self.top, self.left)

    def upstream(self):
        self.swap(self.top, self.left)
        self.cswap(self.routing, self.top, self.right)

    def extract(self):
        self.swap(self.top, self.routing)

    @staticmethod
    def cswap(c, a, b):
        freq = a.frequency + c.frequency - b.frequency
        a.transmon.cswap(freq)

    @staticmethod
    def swap(a, b):
        freq1 = 2 * a.frequency
        freq2 = a.frequency + b.frequency
        a.transmon.swap(freq1, freq2)


class QRAM:
    def __init__(self, graph, pointer, memory):
        """

        :param graph: list of lists, where each inner list contains all the QuantumRouter of a specific level
        :param pointer: a CavityMode representing the pointer
        :param memory: a list of CavityMode representing the memory modes
        """
        self.root = graph[0][0]
        self.pointer = pointer
        self.memory = memory
        self.graph = graph
        self.height = len(graph)

    def downstream(self, level):
        for i in range(level + 1):
            for router in self.graph[i]:
                router.downstream()

    def downstream_address(self, address):
        for i, b in enumerate(address):
            self.root.top.initialize(b)  # load address bit
            self.downstream(i)
            for router in graph[i]:
                router.initialize()

    def downstream_pointer(self):
        self.root.top = self.pointer
        self.downstream(self.height - 1)

    def memory_load(self):
        # load the memory modes into the top modes of the bottom routers
        for i, router in enumerate(self.graph[-1]):
            router.cswap(router.left, self.memory[i], router.top)
            router.cswap(router.right, self.memory[i + 1], router.top)

    def upstream(self, level):
        for i in range(level, -1, -1):
            for r in self.graph[i]:
                r.upstream()

    def upstream_data(self):
        self.memory_load()
        self.upstream(self.height - 2)

    def upstream_pointer(self):
        self.upstream(self.height - 1)

    def upstream_address(self):
        for i in range(self.height - 1, -1, -1):
            for r in self.graph[i]:
                r.extract()
            self.upstream(i - 1)

    def get_data(self, address):
        self.downstream_address(address)
        self.downstream_pointer()
        self.upstream_data()
        self.upstream_pointer()
        self.upstream_address()


transmon = Transmon("transmon", 3e6)
pointer = CavityMode("pointer", 15.3e6, transmon)
memory = [CavityMode("reg" + str(i), freq, transmon) for i, freq in enumerate([1.3e6, 2.3e6, 3.3e6, 4.3e6])]
a = QuantumRouter(*[CavityMode("a" + str(i), freq, transmon) for i, freq in enumerate([5.3e6, 6.3e6, 7.3e6, 8.3e6])])
b = QuantumRouter(*[CavityMode("b" + str(i), freq, transmon) for i, freq in enumerate([7.3e6, 9.3e6, 10.3e6, 11.3e6])])
b.top = a.left
c = QuantumRouter(*[CavityMode("c" + str(i), freq, transmon) for i, freq in enumerate([8.3e6, 12.3e6, 13.3e6, 14.3e6])])
c.top = a.right
graph = [[a], [b, c]]  # routers of the graph grouped by levels
q = QRAM(graph, pointer, memory)
with program() as test:
    q.get_data([0, 0])

qmm = QuantumMachinesManager.QuantumMachinesManager()
simulation_config = SimulationConfig(
    duration=int(1.5e3),
)
job = qmm.simulate(config, test, simulation_config)
job.get_simulated_samples().con1.plot()
