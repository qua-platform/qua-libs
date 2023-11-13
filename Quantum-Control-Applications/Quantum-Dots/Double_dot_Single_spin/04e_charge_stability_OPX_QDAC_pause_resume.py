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
from time import sleep

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
    I = declare(fixed)  # QUA fixed used to store the outcome of the readout
    I_st = declare_stream()  # Data stream used to process and transfer the saved results
    n_st = declare_stream()

    with for_(i, 0, i < n_points_slow + 1, i + 1):

        pause()

        with for_(j, 0, j < n_points_fast, j + 1):

            pause()

            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                # Readout: the voltage measured by the analog input 1 is recorded and the integrated result is stored in "I"
                measure('readout', 'charge_sensor_DC', None, integration.full('cos', I, 'out1'))
                # Transfer the results from the OPX FPGA to its processor
                save(I, I_st)
                # Wait at each iteration in order to ensure that the data will not be transfered faster than 1 sample per µs to the stream processing.
                # Otherwise the processor will recieve the samples fatser than it can process them which can cause the OPX to crash.
                wait(1_000 * u.ns)  # in ns

        # Save the LO iteration to get the progress bar
        save(i, n_st)
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        # The points sent to "I_st" are grouped in a 1D buffer and then averaged together
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("I")
        n_st.save_all("iteration")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

def wait_until_job_is_paused(current_job):
    """
    Waits until the OPX FPGA reaches the pause statement.
    Used when the OPX sequence needs to be synchronized with an external parameter sweep.

    :param current_job: the job object.
    """
    while not current_job.is_paused():
        sleep(0.1)
        pass
    return True

#######################
# Simulate or execute #
#######################

# Open the quantum machine
qm = qmm.open_qm(config)
# Send the QUA program to the OPX, which compiles and executes it
job = qm.execute(charge_stability)
# Creates results handles to fetch the data
res_handles = job.result_handles
I_handle = res_handles.get("I")
n_handle = res_handles.get("iteration")
# Initialize empty vectors to store the global 'I' & 'Q' results
I_tot = []
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
for i in range(n_points_slow):  # Loop over y-voltages
    # Set voltage
    # Resume the QUA program (escape the 'pause' statement)
    job.resume()
    # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
    wait_until_job_is_paused(job)
    for j in range(n_points_fast):  # Loop over x-voltages
        # Set voltage
        # Resume the QUA program (escape the 'pause' statement)
        job.resume()
        # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
        wait_until_job_is_paused(job)
    # Wait until the data of this run is processed by the stream processing
    I_handle.wait_for_values(i + 1)
    n_handle.wait_for_values(i + 1)
    # Fetch the data from the last OPX run corresponding to the current LO frequency
    I = np.concatenate(I_handle.fetch(i)["value"])
    iteration = n_handle.fetch(i)["value"][0]
    # Update the list of global results
    I_tot.append(I)
    # Progress bar
    progress_counter(iteration, n_points_slow)
    plt.cla()
    plt.pcolor(n_points_fast, n_points_slow, I_tot)
    plt.pause(0.1)

# Interrupt the FPGA program
job.halt()
