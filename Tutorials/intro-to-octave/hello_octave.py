"""
hello_octave.py: template for basic usage of octave
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from set_octave import *
from configuration import *
from qm import SimulationConfig
import time


############################
# Set octave configuration #
############################
octave_1 = OctavesSettings()
octave_1.name = "octave1"
octave_1.con = "con1"
octave_1.ip = octave_ip
octave_1.port = octave_port
octave_1.clock = "Internal"
octaves_settings = [octave_1]
port_mapping = [{
    ('con1',  1) : ('octave1', 'I1'),
    ('con1',  2) : ('octave1', 'Q1'),
    ('con1',  3) : ('octave1', 'I2'),
    ('con1',  4) : ('octave1', 'Q2'),
    ('con1',  5) : ('octave1', 'I3'),
    ('con1',  6) : ('octave1', 'Q3'),
    ('con1',  7) : ('octave1', 'I4'),
    ('con1',  8) : ('octave1', 'Q4'),
    ('con1',  9) : ('octave1', 'I5'),
    ('con1', 10) : ('octave1', 'Q5'),
}] # If using the default port mapping you can delete this, otherwise change it to your OPX-Octave port mapping. If having more than one octave, please define it as a list of dictionaries
octave_config = octave_configuration(octaves_settings, port_mapping=port_mapping)

###################################
# Open Communication with the QOP #
###################################
# qmm = QuantumMachinesManager(host=opx_ip, port=opx_port, octave=octave_config)
qmm = QuantumMachinesManager(host=opx_ip, cluster_name="Cluster_81", octave=octave_config)
qm = qmm.open_qm(config)
###################
# The QUA program #
###################
with program() as hello_octave:
    with infinite_loop_():
        play("cw", "qe1")

###################
# Octave settings #
###################

element_1 = ElementsSettings()
element_1.name = "qe1"
element_1.LO_source = "Internal"
element_1.gain = -5
element_1.switch_mode = "on"
element_1.RF_in_port = ["octave1", 1]
element_1.Down_convert_LO_source = "Internal"
element_1.IF_mode = "direct"
elements_settings = [element_1] # If you use the default parameters as defined in the README file, you can delete this
qmm, qm = octave_settings(
    qmm=qmm,
    qm=qm,
    prog=hello_octave,
    config=config,
    octaves_settings=octaves_settings,
    elements_settings=elements_settings,
    calibration=True,
)

simulate = False
if simulate:
    simulation_config = SimulationConfig(duration=400)  # in clock cycles
    job_sim = qmm.simulate(config, hello_octave, simulation_config)
    # Simulate blocks python until the simulation is done
    job_sim.get_simulated_samples().con1.plot()
else:
    job = qm.execute(hello_octave)
    # Execute does not block python! As this is an infinite loop, the job would run forever. In this case, we've put a 10
    # seconds sleep and then halted the job.
    time.sleep(10)
    job.halt()
