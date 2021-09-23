from configuration import *
from qm import SimulationConfig, LoopbackInterface
from qm.QuantumMachinesManager import QuantumMachinesManager

from qm.qua import *

t_parity = 128
bits = 3
threshold = 0.01

simulation_config = SimulationConfig(
    duration=int(400),  # need to run the simulation for long enough to get all points
)

with program() as prog:
    bit_k = [declare(int)] * bits
    zero = declare(int, value=0)
    k = declare(int)
    n = declare(int, value=0)
    I = declare(fixed)

    for k in range(1, bits+1):
        play("X90", "transmon")
        wait(int(t_parity / 2 ** k), "transmon")
        frame_rotation_2pi(-n / 2 ** k)
        play("X90", "transmon")
        align()
        measure(
            "Readout_Op",
            "RR",
            None,
            dual_demod.full("optimal_integW_1", "out1", "optimal_integW_2", "out2", I),
        )

        assign(bit_k[k-1], Cast.to_int(I > threshold))
        assign(n, n + bit_k[k-1] * 2 ** k)

        align()

    with switch_(n):
        for i in range(2 ** bits):
            with case_(i):
                play(f"reset_fock_{i}", "storage")
    save(n, "n")

qmm = QuantumMachinesManager()
job = qmm.simulate(config, prog, simulation_config)

job.get_simulated_samples().con1.plot()
