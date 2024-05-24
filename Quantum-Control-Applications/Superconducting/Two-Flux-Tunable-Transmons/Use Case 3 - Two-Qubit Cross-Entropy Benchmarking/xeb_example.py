import numpy as np
from qm import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from quam.components import *
from quam.examples.superconducting_qubits import Transmon, QuAM
from xeb_config import XEBConfig
from xeb import XEB
from quam.components.pulses import GaussianPulse, ReadoutPulse
from simulated_backend import backend
from qua_gate import QUAGate

def cz_gate(qubit1, qubit2):
    play('cz_pulse', qubit1_el)
    frame_rotation
xeb_config = XEBConfig(
    seqs=10,
    depths=np.arange(50),
    n_shots=100,
    qubits_ids=["q0", "q1"],
    baseline_gate_name="X90",
    gate_set_choice="sw",
    two_qb_gate=QUAGate("cz", cz_gate),
    impose_0_cycle=False,
    save_dir="",
    should_save_data=True,
    generate_new_data=True,
    disjoint_processing=False,
)

machine = QuAM()

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
qmm = QuantumMachinesManager(
    host="tyler-263ed49e.dev.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

xeb = XEB(xeb_config, quam=machine, qmm=qmm)
job = xeb.run(simulate=False)

job.circuits[3][5].draw("mpl")

result = job.result()

result.plot_fidelities()
result.plot_records()
result.plot_state_heatmap()
