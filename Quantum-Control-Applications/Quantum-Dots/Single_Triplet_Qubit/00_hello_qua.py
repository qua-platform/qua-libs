"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *


###################
# The QUA program #
###################
level_empty = (-0.2, +0.2)
level_init = (0.1, 0.05)
level_manip = (-0.05, 0.1)
level_readout = (0.2, -0.1)

duration_empty = 1000 // 4
duration_init_ramp = 1000 // 4
duration_init = 2000 // 4
duration_manip = 8000 // 4
duration_readout = 500 // 4

with program() as hello_qua:
    i = declare(int)
    j = declare(int)
    k = declare(int)
    set_dc_offset("P1_sticky", "single", level_empty[0])
    set_dc_offset("P2_sticky", "single", level_empty[1])
    align()
    # Empty-Initialization-Idle-Readout
    with for_(i, 0, i < 3, i + 1):
        with strict_timing_():
            for i, el in enumerate(["P1_sticky", "P2_sticky"]):
                # Empty
                wait(duration_empty, el)
                # Ramp to init
                play(ramp((level_init[i] - level_empty[i]) / (duration_init_ramp * 4)), el, duration=duration_init_ramp)
                # Init
                wait(duration_init, el)
                # # Manipulation
                # if i == 0:
                #     align(el, "qubit_left")
                #     wait(duration_manip-bias_length//4-2*500-500, "qubit_left")
                #     play("gauss", "qubit_left", duration=500)
                #     wait(500, "qubit_left")
                #     play("gauss", "qubit_left", duration=500)
                play("bias" * amp((level_manip[i] - level_init[i]) / P1_amp), el)
                wait(duration_manip - bias_length // 4, el)
                # Readout
                play("bias" * amp((level_readout[i] - level_manip[i]) / P1_amp), el)
                wait(duration_readout - bias_length // 4, el)
        # Go to empty
        ramp_to_zero("P1_sticky")
        ramp_to_zero("P2_sticky")

    # Manipulation EDSR
    with strict_timing_():
        with for_(j, 0, j < 3, j + 1):
            wait(
                duration_empty + duration_init_ramp + duration_init + duration_manip - bias_length // 4 - 2 * 100 - 500,
                "qubit_left",
            )
            play("cw", "qubit_left", duration=100)
            wait(500, "qubit_left")
            play("cw", "qubit_left", duration=100)
            wait(duration_readout + hold_offset_duration - 121, "qubit_left")

        # Manipulation Singlet-Triplet
        with for_(k, 0, k < 3, k + 1):
            wait(
                duration_empty + duration_init_ramp + duration_init + duration_manip - bias_length // 4 - 2 * 100 - 500,
                "P1",
            )
            play("bias", "P1", duration=100)
            wait(500, "P1")
            play("bias", "P1", duration=100)
            wait(duration_readout + hold_offset_duration - 121, "P1")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
