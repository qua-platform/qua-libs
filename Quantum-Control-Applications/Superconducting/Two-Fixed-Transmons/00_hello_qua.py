# %%
"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration_mw_fem import *

##################
#   Parameters   #
##################

###################
#   QUA Program   #
###################

with program() as PROGRAM:

    # # readout lines
    # for rr in list(RR_CONSTANTS.keys()):
    #     play("cw", rr)

    # align()
    # # xy drives
    # for qubit in list(QUBIT_CONSTANTS.keys()):
    #     play("cw", qubit)
    
    play("cw", "q1_rr")
    play("cw", "q2_xy")
    play("cw", "q3_xy")
    
        
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
        # Open a quantum machine to execute the QUA program
        qm = qmm.open_qm(config)

        # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
        job = qm.execute(PROGRAM)

# %%
