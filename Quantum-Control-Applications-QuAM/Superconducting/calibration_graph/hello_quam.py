#%%
from qm.qua import *
from qualang_tools.units import unit
from quam_libs.components import QuAM
import matplotlib.pyplot as plt


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
# qmm = machine.connect()

q1 = machine.qubits["q1"]
q2 = machine.qubits["q2"]
qp12 = (q1 @ q2)

intermediate_frequency = qp12.cross_resonance.intermediate_frequency
# qp21 = (q2 @ q1)

with program() as prog:

    with infinite_loop_():
        # cross resonance
        qp12.cross_resonance.play("const")  # drive pulse on q1 at f=ft with spec. amp & phase
        qp12.qubit_target.xy.play("cr_target_square")  # drive pulse on q2 at f=ft with spec. amp & phase

        # # zz-inducing drive
        # qp12.zz_drive.play("square")  # drive pulse on q1 at f=ft-d with spec. amp & phase
        # qp12.qubit_target.xy.play("zz_target_square")  # drive pulse on q2 at f=ft-d with spec. amp & phase


qm = qmm.open_qm(config)
job = qm.execute(prog)
plt.show()


# %%
