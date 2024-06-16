import numpy as np
from components import QuAM, Transmon
from xeb_config import XEBConfig
from xeb import XEB
from simulated_backend import backend
from qua_gate import QUAGate
from qm.qua import *


def cz_gate(qubit1: Transmon, qubit2: Transmon):
    """
    CZ gate macro
    Here, we take inspiration from the macro implemented in QUA-Libs and adapt it to our needs using QuAM components.
    """
    qubit1.z.set_dc_offset(-0.10557)
    qubit1.z.wait(189 // 4)
    align()
    qubit1.z.set_dc_offset(0)
    qubit1.z.wait(10)


cz_qua = QUAGate("cz", cz_gate)

xeb_config = XEBConfig(
    seqs=10,
    depths=np.arange(1, 50),
    n_shots=100,
    qubits_ids=["q0", "q1"],
    baseline_gate_name="x90",
    gate_set_choice="sw",
    two_qb_gate=cz_qua,
    impose_0_cycle=False,
    save_dir="",
    should_save_data=True,
    generate_new_data=True,
    disjoint_processing=False,
)

simulate = False
machine = QuAM.load("state.json")
xeb = XEB(xeb_config, quam=machine)
if simulate:
    job = xeb.simulate(backend=backend)
else:
    job = xeb.run(simulate=False)

job.circuits[3][5].draw("mpl")

result = job.result()

result.plot_fidelities()
result.plot_records()
result.plot_state_heatmap()
