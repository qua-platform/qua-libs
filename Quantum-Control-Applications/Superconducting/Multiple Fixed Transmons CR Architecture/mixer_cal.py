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

with program() as mixer_cal:

    with infinite_loop_():

        for q in [0, 1]:
            play("cw", f"q{q}")

        # for c in [[1, 0], [0, 1]]:
        #     play("cw", f"cr_c{c[0]}t{c[1]}")
        #
        # for r in [0, 1]:
        #     play("readout", f"rr{r}")

job = qmm.simulate(build_config(state), mixer_cal, SimulationConfig(1500))
job.get_simulated_samples().con1.plot()
