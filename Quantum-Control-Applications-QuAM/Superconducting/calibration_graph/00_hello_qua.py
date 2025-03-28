# %% {Imports}
import matplotlib.pyplot as plt

from qm.qua import *
from qm import SimulationConfig

from qualang_tools.units import unit

from qualibrate import QualibrationNode, NodeParameters
from quam_config import QuAM


description = """
        RUN BASIC QUA PROGRAM TO TEST QOP CONNECTION
"""


node = QualibrationNode[NodeParameters, QuAM](name="00_hello_qua", description=description, parameters=NodeParameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
node.machine = QuAM.load()
# Generate the OPX and Octave configurations
config = node.machine.generate_config()
# Open Communication with the QOP
qmm = node.machine.connect()

qubits = node.machine.active_qubits

simulate = False

with program() as prog:

    qubits[2].xy.update_frequency(-100e6)
    qubits[1].xy.update_frequency(-100e6)
    qubits[0].xy.update_frequency(-100e6)

    a = declare(fixed)

    with infinite_loop_():

        # with for_(a, 0, a < 1.0, a +0.1):

        qubits[2].xy.play("x180")
        # qubits[1].xy.play('x180')
        # wait(4)
        # align()
        qubits[0].resonator.play("const")
        # align()
        wait(1_000)
        # for qubit in node.machine.active_qubits:
        #     qubit.z.play('const')
        #     qubit.z.wait(4)

        # with for_(a, 0, a < 2.0, a+0.2):
        # # for qubit in node.machine.active_qubits:
        #     qubits[2].xy.play('x180', amplitude_scale=a)
        #     qubits[2].xy.wait(4)


if simulate:
    job = qmm.simulate(config, prog, SimulationConfig(duration=1000))
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(prog)

plt.show()
