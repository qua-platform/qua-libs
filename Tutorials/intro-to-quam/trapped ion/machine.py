# %%
import numpy as np

from typing import (
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)
from dataclasses import field

from qm.qua import *
from quam.core import quam_dataclass, QuamRoot
from quam.components import Qubit
from quam.components.ports import (
    LFFEMAnalogInputPort,
    LFFEMAnalogOutputPort,
    MWFEMAnalogOutputPort,
    FEMDigitalOutputPort,
)
from quam.components.channels import (
    Channel,
    SingleChannel,
    DigitalOutputChannel,
    InOutSingleChannel,
    MWChannel,
)
from quam.components.pulses import (
    Pulse,
    SquareReadoutPulse,
    SquarePulse,
    BaseReadoutPulse,
)
from quam.components.macro import QubitMacro
from quam.utils.qua_types import (
    ScalarFloat,
    QuaVariableFloat,
)
from quam.utils.pulse import add_amplitude_scale_to_pulse_name

#############################################################
## Adding integration to the measurement
#############################################################


def measure_integrated(
    self,
    pulse_name: str,
    amplitude_scale: Optional[Union[ScalarFloat, Sequence[ScalarFloat]]] = None,
    qua_var: QuaVariableFloat = None,
    stream=None,
) -> QuaVariableFloat:
    pulse: BaseReadoutPulse = self.operations[pulse_name]

    if qua_var is None:
        qua_var = declare(fixed)

    pulse_name_with_amp_scale = add_amplitude_scale_to_pulse_name(
        pulse_name, amplitude_scale
    )

    integration_weight_labels = list(pulse.integration_weights_mapping)
    measure(
        pulse_name_with_amp_scale,
        self.name,
        integration.full(integration_weight_labels, qua_var),
        adc_stream=stream,
    )
    return qua_var


InOutSingleChannel.measure_integrated = measure_integrated

#############################################################
## Qubit abstraction
#############################################################


@quam_dataclass
class HyperfineQubit(Qubit):
    shelving: SingleChannel = None
    readout: InOutSingleChannel = None


@quam_dataclass
class GlobalOperations(Qubit):
    normalized_rabi_freqs: List[float] = field(default_factory=list)
    global_mw: MWChannel = None
    ion_displacement: Channel = None


@quam_dataclass
class Quam(QuamRoot):
    qubits: Dict[str, HyperfineQubit] = field(default_factory=dict)
    global_op: GlobalOperations = None


#############################################################
## Operation macros
#############################################################


@quam_dataclass
class MeasureMacro(QubitMacro):
    threshold: float

    def apply(self):
        # perform shelving operation
        self.qubit.shelving.play("const")
        self.qubit.align()

        # integrating the PMT signal
        I = self.qubit.readout.measure_integrated("const")

        # We declare a QUA variable to store the boolean result of thresholding the I value.
        qubit_state = declare(int)
        # Since |1> is shelved, high fluorescence corresponds to |0>
        # i.e. I < self.threshold implies |1> and vice versa
        assign(qubit_state, Cast.to_int(I < self.threshold))
        return qubit_state


@quam_dataclass
class SingleXMacro(QubitMacro):
    # n_qubits: int
    # normalized_rabi_freqs: List[float]

    def apply(self, qubit_idx: int):
        # assert n_qubits == 2, "implemented for only 2 qubits"
        # assert np.allclose(normalized_rabi_freqs, [1, 2])
        self.qubit.ion_displacement.play("ttl")
        self.qubit.align()
        with switch_(qubit_idx):
            with case_(1):
                self.qubit.global_mw.play("x180")
            with case_(2):
                self.qubit.global_mw.play("y180")
                self.qubit.global_mw.play("x180")
                self.qubit.global_mw.play("y180")
        self.qubit.align()
        self.qubit.ion_displacement.play("ttl")
        wait(100)
        align()


@quam_dataclass
class DoubleXMacro(QubitMacro):
    # n_qubits: int
    # normalized_rabi_freqs: List[float]

    def apply(self, qubit_idx: int, amp_scale=1, N_double_X=1):
        i = declare(int)
        self.qubit.ion_displacement.play("ttl")
        self.qubit.align()
        with switch_(qubit_idx):
            with case_(1):
                with for_(i, 0, i < N_double_X * 2, i + 1):
                    self.qubit.global_mw.play("x180", amplitude_scale=amp_scale)
            with case_(2):
                with for_(i, 0, i < N_double_X * 2, i + 1):
                    self.qubit.global_mw.play("y180", amplitude_scale=amp_scale)
                    self.qubit.global_mw.play("x180", amplitude_scale=amp_scale)
                    self.qubit.global_mw.play("y180", amplitude_scale=amp_scale)
        self.qubit.align()
        self.qubit.ion_displacement.play("ttl")
        wait(100)
        align()


#############################################################
## Generate QUAM object
#############################################################

machine = Quam()

n_qubits = 2
aom_position = np.linspace(200e6, 250e6, n_qubits)
mw_IF = 100e6
mw_LO = 3e9
mw_band = 1
assert n_qubits <= 4, "This setup supports up to 4 qubits."

# for each qubit
for i in range(n_qubits):
    qubit_id = f"q{i + 1}"
    qubit = HyperfineQubit(
        id=f"{qubit_id}",
        readout=InOutSingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 2),
            opx_input=LFFEMAnalogInputPort("con1", 1, 2),
            intermediate_frequency=aom_position[i],
        ),
        shelving=SingleChannel(
            opx_output=LFFEMAnalogOutputPort("con1", 1, 3),
            intermediate_frequency=aom_position[i],
        ),
    )

    # define pulse
    qubit.shelving.operations["const"] = SquarePulse(length=1_000, amplitude=0.1)
    qubit.readout.operations["const"] = SquareReadoutPulse(length=2_000, amplitude=0.1)

    # define macro
    qubit.macros["measure"] = MeasureMacro(threshold=10)

    # add to quam
    machine.qubits[qubit_id] = qubit

# set global properties
machine.global_op = GlobalOperations(
    normalized_rabi_freqs=np.linspace(1, 2, n_qubits).tolist(),
    global_mw=MWChannel(
        id="global_mw",
        opx_output=MWFEMAnalogOutputPort(
            "con1", 8, 1, band=mw_band, upconverter_frequency=mw_LO
        ),
        intermediate_frequency=mw_IF,
    ),
    ion_displacement=Channel(
        digital_outputs={
            "ttl": DigitalOutputChannel(
                opx_output=FEMDigitalOutputPort("con1", 8, 1), delay=136, buffer=0
            )
        },
    ),
)

# define pulse
machine.global_op.global_mw.operations["x180"] = SquarePulse(amplitude=0.2, length=1000)
machine.global_op.global_mw.operations["y180"] = SquarePulse(
    amplitude=0.2, length=1000, axis_angle=90
)
machine.global_op.ion_displacement.operations["ttl"] = Pulse(
    length=1000, digital_marker=[(1, 500), (0, 0)]
)

# operation macro
machine.global_op.macros["X"] = SingleXMacro()
machine.global_op.macros["X"] = DoubleXMacro()

machine.print_summary()

# Save the updated QUAM configuration
machine.save("state.json")
# Load the QUAM configuration
# machine = BasicQuam.load("state.json")

# QUA configuration
# qua_config = machine.generate_config()
# qm = qmm.open_qm(qua_config)

# %%
from qm import generate_qua_script

n_avg = 2
optimize_qubit_idx = 2

with program() as prog:
    n = declare(int)
    es_st = declare_stream()
    qubit_idx = declare(int, optimize_qubit_idx)

    with for_(n, 0, n < n_avg, n + 1):
        machine.global_op.apply("X", qubit_idx)
        for i, qubit in enumerate(machine.qubits.values()):
            es = qubit.apply("measure")
            save(es, es_st)
            align()
            wait(1_000)

    with stream_processing():
        es_st.buffer(n_avg).average().save_all("es")


print(generate_qua_script(prog))

# %%
from qm import QuantumMachinesManager
from qm import SimulationConfig

qop_ip = "172.16.33.115"  # Write the QM router IP address
cluster_name = "CS_4"  # Write your cluster_name if version >= QOP220
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
qua_config = machine.generate_config()

# Simulates the QUA program for the specified duration
simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
# Simulate blocks python until the simulation is done
job = qmm.simulate(qua_config, prog, simulation_config)
# Get the simulated samples
samples = job.get_simulated_samples()
# Plot the simulated samples
samples.con1.plot()

# %%
# Get the waveform report object
waveform_report = job.get_simulated_waveform_report()
# Cast the waveform report to a python dictionary
waveform_dict = waveform_report.to_dict()
# Visualize and save the waveform report
waveform_report.create_plot(samples, plot=True, save_path=None)
