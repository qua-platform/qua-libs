"""
        CHARGE STABILITY DIAGRAM
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close

###################
# The QUA program #
###################

n_avg = 100
n_points_slow = 10
n_points_fast = 10

with program() as charge_stability:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    j = declare(int)  # QUA integer used as an index to loop over the voltage points
    I = declare(fixed)  # QUA fixed () used to store the outcome of the readout
    I_st = declare_stream()  # Data stream used to process and transfer the saved results
    n_st = declare_stream()  # Data stream used to transfer the iteration number

    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        with for_(i, 0, i < n_points_slow, i + 1):  # The loop over the QDAC2 voltage list
            # Trigger the next QDAC2 voltage  from the pre-loaded list
            play("trigger", "qdac_trigger2")
            with for_(j, 0, j < n_points_fast, j + 1):  # The loop over the QDAC2 voltage list
                # Trigger the next QDAC2 voltage  from the pre-loaded list
                play("trigger", "qdac_trigger1")
                # Wait before measuring according to the QDAC2 response time
                wait(1000 * u.ns, "charge_sensor_DC")
                # Readout: the voltage measured by the analog input 1 is recorded and the integrated result is stored in "I"
                measure('readout', 'charge_sensor_DC', None, integration.full('cos', I, 'out1'))
                # Transfer the results from the OPX FPGA to its processor
                save(I, I_st)
        # Save the iteration number (progress counter)
        save(n, n_st)

    # Stream processing section used to process the data before saving it.
    with stream_processing():
        # For live plotting, the global averaging is used in order to fecth data while the program is running
        I_st.buffer(n_points_fast).buffer(n_points_slow).average().save("I")
        n_st.save("iteration")


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
    job = qmm.simulate(config, charge_stability, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(charge_stability)
    # Get results from QUA program
    my_results = fetching_tool(job, data_list=['counter', 'I'], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while my_results.is_processing():
        counter, I = my_results.fetch_all()
        I_volts = u.demod2volts(I, readout_len)
        progress_counter(counter, n_avg, start_time=my_results.get_start_time())
        plt.cla()
        plt.pcolor(np.arange(n_points_fast), np.arange(n_points_slow), I_volts)
        plt.xlabel('Sensor gate [V]')
        plt.ylabel('Voltage')
        plt.pause(0.1)
    plt.show()