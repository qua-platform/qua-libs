from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from quam_libs.components import QuAM

machine = QuAM.load()
config = machine.generate_config()
qmm = machine.connect()

qubits = machine.active_qubits

simulate = True

with program() as prog:
    qubits[0].xy.update_frequency(-100e6)

    a = declare(fixed)

    machine.twpa_run()

    with infinite_loop_():
        qubits[0].xy.play("x180")
        wait(100, qubits[0].xy.name)

del machine

import json
with open('/tmp/out', "w+") as f:
    json.dump(config, f, indent=2)

if simulate:
    job = qmm.simulate(config, prog, SimulationConfig(duration=1000))
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path="./")

else:
    qm = qmm.open_qm(config)
    job = qm.execute(prog)
