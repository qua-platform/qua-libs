"""
        READOUT SEARCH
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element

###################
# The QUA program #
###################

n_avg = 100  # Number of averaging loops
drive_frequency = 10 * u.GHz
N = 100
bit_shit_cte = 7
n_avg_ro = 2 ** bit_shit_cte
Coulomb_pk_to_pk = 0.2

with program() as readout_search:
    n = declare(int)  # QUA integer used as an index for the outer averaging loop
    n_ro = declare(int)  # QUA integer used as an index for the inner averaging loop
    counter = declare(int)  # QUA integer used as an index for the Coulomb drive
    I_on = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout when a qubit pulse is played
    I_off_avg = declare(fixed)  # QUA fixed used to store the outcome of the readout without qubit pulse
    I_on_st = declare_stream()  # Stream for I_on
    I_off_st = declare_stream()  # Stream for I_off
    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("TIA", I_on, I_off)
    # Set the qubit drive frequency
    update_frequency("qubit", int(drive_frequency - qubit_LO_left))

    with for_(n, 0, n < n_avg, n + 1):  # The outer averaging loop
        assign(I_on_avg, 0)
        assign(I_off_avg, 0)
        # Play the Coulomb and qubit pulse continuously for a time given by coulomb_drive_length
        #      ____      ____      ____      ____
        #     |    |    |    |    |    |    |    |  --> I_on
        # _@__|    |_@__|    |_@__|    |_@__|    |...

        with for_(counter, 0, counter < N, counter + 1):
            # The Coulomb pulse
            play("bias" * amp(Coulomb_pk_to_pk), "P1")
            play("bias" * amp(-Coulomb_pk_to_pk), "P1")
            wait(16 * u.ns, "qubit")  # Wait before driving the qubit
            play("cw", "qubit")  # Qubit drive

        # # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
        # wait(IV_buffer_len * u.ns, "TIA")
        with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_on
            measure('readout', 'TIA', None, integration.full('cos', I_on, 'out1'))
            assign(I_on_avg, (I_on >> bit_shit_cte) + I_on_avg)
        save(I_on_avg, I_on_st)

        align()
        # Play the Coulomb pulse only continuously for a time given by coulomb_drive_length
        #      ____      ____      ____      ____
        #     |    |    |    |    |    |    |    |  --> I_off
        # ____|    |____|    |____|    |____|    |...

        with for_(counter, 0, counter < N, counter + 1):
            # The Coulomb pulse
            play("bias" * amp(Coulomb_pk_to_pk), "P1")
            play("bias" * amp(-Coulomb_pk_to_pk), "P1")

        # # Wait for the IV converter to reach its steady state and measure for a duration given by total_integration_time
        # wait(IV_buffer_len * u.ns, "TIA")
        with for_(n_ro, 0, n_ro < n_avg_ro, n_ro + 1):  # The inner averaging loop for I_off
            measure('readout', 'TIA', None, integration.full('cos', I_off, 'out1'))
            assign(I_off_avg, (I_off >> bit_shit_cte) + I_off_avg)
        save(I_off_avg, I_off_st)

    with stream_processing():
        # Average and stream I_on, I_off, the difference and their sum - similar to a lock-in
        I_on_st.average().save("I_on")
        I_off_st.average().save("I_off")
        (I_on_st - I_off_st).average().save("I_diff")
        (I_on_st + I_off_st).average().save("I_sum")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, readout_search, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(readout_search)