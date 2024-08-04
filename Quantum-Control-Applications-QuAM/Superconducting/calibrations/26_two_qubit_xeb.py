import numpy as np
from quam_libs.components import QuAM, TransmonPair
from quam_libs.experiments.two_qubit_xeb import (
    XEBConfig,
    XEB,
    backend as fake_backend,
    QUAGate,
)

machine = QuAM.load()

# Get the relevant QuAM components
qc = machine.qubits["q4"]
print(machine.qubits)
qt = machine.qubits["q5"]
qubit_pair = machine.qubit_pairs["q45"]


def cz_gate(qubit_pair: TransmonPair):
    """
    CZ gate macro
    Here, we take inspiration from the macro implemented in QUA-Libs and adapt it to our needs using QuAM components.
    """
    pass


cz_qua = QUAGate("cz", cz_gate)

xeb_config = XEBConfig(
    seqs=10,
    depths=np.arange(1, 8),
    n_shots=1000,
    qubits_ids=[qc.name, qt.name],
    qubit_pairs_ids=[qubit_pair.name],
    baseline_gate_name="x90",
    gate_set_choice="sw",
    two_qb_gate=cz_qua,
    save_dir="",
    should_save_data=True,
    generate_new_data=True,
    disjoint_processing=False,
    reset_method="active",
    reset_kwargs={"max_tries": 3, "pi_pulse": "x180"},
)

simulate = False  # Set to True to simulate the experiment with Qiskit Aer instead of running it on the QPU
xeb = XEB(xeb_config, quam=machine)
if simulate:
    job = xeb.simulate(backend=fake_backend)
else:
    job = xeb.run(simulate=False)  # If simulate is False, job is run on the QPU, else pulse output is simulated

job.circuits[3][5].draw("mpl")

result = job.result()

result.plot_fidelities()
result.plot_records()
result.plot_state_heatmap()
