# %%
"""
A simple program that plays CW tones to all relevant MW channels and takse the DC offsets for the LF FEM channels from the config.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration_mw_fem import *

##################
#   Parameters   #
##################

###################
#   QUA Program   #
###################

with program() as PROGRAM:

    with infinite_loop_():
        play("cw", "q1_rr")

    with infinite_loop_():
        play("cw", "q2_xy")

    with infinite_loop_():
        play("cw", "q3_xy")

    # # readout lines
    # for rr in resonators:
    #     with infinite_loop_():
    #         play("cw", rr)

    # # twpa
    # for twpa in twpas:
    #     with infinite_loop_():
    #         play("cw", twpa)

    # xy drives
    # for qubit in QUBIT_CONSTANTS.keys():
    # for qubit in ["q2_xy", "q3_xy"]:
    #     with infinite_loop_():
    #         play("cw", qubit)


if __name__ == "__main__":
    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    ###########################
    # Run or Simulate Program #
    ###########################

    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        # Simulate blocks python until the simulation is done
        job = qmm.simulate(config, PROGRAM, simulation_config)
        # Plot the simulated samples
        job.get_simulated_samples().con1.plot()
    else:
        from qm import generate_qua_script
        sourceFile = open('debug_01_continuous.py', 'w')
        print(generate_qua_script(continous, config), file=sourceFile) 
        sourceFile.close()

        # Open a quantum machine to execute the QUA program
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
        job = qm.execute(PROGRAM)

# %%
