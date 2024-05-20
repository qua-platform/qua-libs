import numpy as np
from qm import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from quam.components import *
from quam.examples.superconducting_qubits import Transmon, QuAM
from xeb_config import XEBConfig
from xeb import XEB
from quam.components.pulses import GaussianPulse
from qiskit_aer.noise import depolarizing_error, NoiseModel
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
num_qubits = 2  # Number of qubits
apply_CZ = True  # Apply CZ gate
error1q = 0.07
error2q = 0.03
effective_error = error2q + num_qubits*error1q if num_qubits == 2 and apply_CZ else num_qubits*error1q
depol_error1q = depolarizing_error(error1q, 1)
depol_error2q = depolarizing_error(error2q, 2)
sq_gate_set = ["h", "t", "rx", "ry", "sw"]
noise_model = NoiseModel(basis_gates = sq_gate_set)
if num_qubits == 2:
    noise_model.add_all_qubit_quantum_error(depol_error2q, ["cz"])
noise_model.add_all_qubit_quantum_error(depol_error1q, sq_gate_set)
# noise_model.add_all_qubit_quantum_error(depol_error1q, [ 'rx', 'sw', 'ry', 't'])
backend = AerSimulator(noise_model=noise_model, method="density_matrix", basis_gates=noise_model.basis_gates)
# backend.target.add_instruction(SW, properties={(qubit,): None for qubit in range(num_qubits)}, name="sw")
print(noise_model.noise_qubits)
print(noise_model)
print(backend.operation_names)
xeb_config = XEBConfig(
    seqs=10,
    depths=np.arange(50),
    n_shots=100,
    qubits_ids=["q0", 'q1'],
    baseline_gate_name="X90",
    gate_set_choice="sw",
    two_qb_gate=None,
    impose_0_cycle=False,
    save_dir="",
    should_save_data=True,
    generate_new_data=True,
    disjoint_processing=False,
)

machine = QuAM()  #

num_qubits = 2
for idx in range(num_qubits):
    # Create transmon qubit component
    transmon = Transmon(id=idx)

    # Add xy drive line channel
    transmon.xy = IQChannel(
        opx_output_I=("con1", 3 * idx + 3),
        opx_output_Q=("con1", 3 * idx + 4),
        frequency_converter_up=FrequencyConverter(
            mixer=Mixer(),
            local_oscillator=LocalOscillator(power=10, frequency=6e9),
        ),
        intermediate_frequency=100e6,
    )

    # Add transmon flux line channel
    transmon.z = SingleChannel(opx_output=("con1", 3 * idx + 5))

    # Add resonator channel
    transmon.resonator = InOutIQChannel(
        id=idx,
        opx_output_I=("con1", 3 * idx + 1),
        opx_output_Q=("con1", 3 * idx + 2),
        opx_input_I=("con1", 1),
        opx_input_Q=(
            "con1",
            2,
        ),
        frequency_converter_up=FrequencyConverter(
            mixer=Mixer(), local_oscillator=LocalOscillator(power=10, frequency=6e9)
        ),
    )
    machine.qubits[transmon.name] = transmon

# Create a Gaussian pulse
gaussian_pulses = [GaussianPulse(length=20, amplitude=0.2, sigma=3) for _ in range(4)]

# Attach the pulse to the XY channel of the first qubit
machine.qubits["q0"].xy.operations["X90"] = gaussian_pulses[0]
machine.qubits["q0"].resonator.operations["readout"] = gaussian_pulses[1]
machine.qubits["q1"].xy.operations["X90"] = gaussian_pulses[2]
machine.qubits["q1"].resonator.operations["readout"] = gaussian_pulses[3]
qmm = QuantumMachinesManager(
    host="tyler-263ed49e.dev.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

xeb = XEB(xeb_config, quam=machine, qmm=qmm)
job = xeb.simulate(backend)

job.circuits[3][2].draw('mpl')
results = job.result()
plt.figure()
results.plot_fidelities()
results.plot_state_heatmap()
results.plot_records()