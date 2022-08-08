from state_and_config import build_config, state
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from qm.simulate import SimulationConfig
from qm.qua import *

# build config
config = build_config(state)
qmm = QuantumMachinesManager(
    host="nord-quantique-d14d58b1.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

with program() as hello_qua:

    a = declare(fixed)

    play("x180", "q0", duration=10000)
    # with for_(a, 0.1, a<1.0, a+0.1):
    #     play("x180"*amp(a), "q0")
    #     play("x180"*amp(1.0-a), "q0")

    I = declare(fixed)

    for q in [0, 1]:

        update_frequency(f"q{q}", int(100e6))

        reset_phase(f"q{q}")
        play("x90", f"q{q}")
        reset_phase(f"q{q}")
        play("x180", f"q{q}")
        reset_phase(f"q{q}")
        play("x-90", f"q{q}")
        reset_phase(f"q{q}")
        play("x-180", f"q{q}")
        reset_phase(f"q{q}")
        play("y90", f"q{q}")
        reset_phase(f"q{q}")
        play("y180", f"q{q}")
        reset_phase(f"q{q}")
        play("y-90", f"q{q}")
        reset_phase(f"q{q}")
        play("y-180", f"q{q}")

    for c in [[0, 1], [1, 0]]:
        align()
        wait(10, f"cr_c{c[0]}t{c[1]}")
        play("cw", f"cr_c{c[0]}t{c[1]}")

    for r in [0, 1]:

        align()
        wait(10)
        play("cw", f"rr{r}")
        wait(10)
        measure(
            "readout",
            f"rr{r}",
            None,
            dual_demod.full("cos", "out1", "sin", "out2", I),
        )

job = qmm.simulate(build_config(state), hello_qua, SimulationConfig(11000))
job.get_simulated_samples().con1.plot()
