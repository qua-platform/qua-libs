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
qmm = machine.connect()

qc = machine.qubits["q1"]
qt = machine.qubits["q2"]
qp = (qc @ qt)
cr = qp.cross_resonance
zz = qp.zz_drive
intermediate_frequency = cr.intermediate_frequency

with program() as prog:

    with infinite_loop_():
        # cross resonance
        cr.play("square")  # drive pulse on q1 at f=ft with spec. amp & phase
        qt.xy.play(f"{cr.name}_Square")  # drive pulse on q2 at f=ft with spec. amp & phase

        # zz-inducing drive
        zz.play("square")  # drive pulse on q1 at f=ft-d with spec. amp & phase
        qt.xy_detuned.play(f"{zz.name}_Square")   # drive pulse on q2 at f=ft-d with spec. amp & phase


qm = qmm.open_qm(config)
job = qm.execute(prog)
plt.show()


# %%
