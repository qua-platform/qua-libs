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
from qdac2_driver import *

###################
# The QUA program #
###################

n_avg = 100
n_points_slow = 10
n_points_fast = 10

voltage_values_slow = list(np.linspace(-1.5, 1.5, n_points_slow))
voltage_values_fast = list(np.linspace(-1.5, 1.5, n_points_fast))


with program() as charge_stability:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    j = declare(int)  # QUA integer used as an index to loop over the voltage points
    I = declare(fixed)  # QUA fixed used to store the outcome of the readout
    I_st = declare_stream()  # Data stream used to process and transfer the saved results
    Q = declare(fixed)  # QUA fixed used to store the outcome of the readout
    Q_st = declare_stream()  # Data stream used to process and transfer the saved results
    n_st = declare_stream()

    with for_(i, 0, i < n_points_slow + 1, i + 1):

        with for_(j, 0, j < n_points_fast, j + 1):

            pause()

            wait(10 * u.ms, 'charge_sensor_RF')

            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                # Readout: the voltage measured by the analog input 1 is recorded and the integrated result is stored in "I"
                measure('readout', 'charge_sensor_RF', None, demod.full('cos', I, 'out2'), demod.full('sin', Q, 'out2'))
                # Transfer the results from the OPX FPGA to its processor
                save(I, I_st)
                save(Q, Q_st)
                # Wait at each iteration in order to ensure that the data will not be transfered faster than 1 sample per µs to the stream processing.
                # Otherwise the processor will recieve the samples fatser than it can process them which can cause the OPX to crash.
                wait(1_000 * u.ns)  # in ns

        # Save the LO iteration to get the progress bar
        save(i, n_st)
    # Stream processing section used to process the data before saving it.
    with stream_processing():
        # The points sent to "I_st" are grouped in a 1D buffer and then averaged together
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("I")
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("Q")
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
# Create the qdac instrument
qdac = QDACII("Ethernet", IP_address="127.0.0.1", port=5025)  # Using Ethernet protocol
### QDAC2 section
# TODO: ask what channels to use
qdac_channel_fast = 0
qdac_channel_slow = 0
# Set the channel output range
qdac.write(f"sour{qdac_channel_fast}:rang low")
# Set the channel output filter
qdac.write(f"sour{qdac_channel_fast}:filt med")
# Set the slew rate in V/s to avoid transients when abruptly stepping the voltage -Must be within [0.01; 2e7] V/s
qdac.write(f"sour{qdac_channel_fast}:dc:volt:slew {1000}")
qdac.write(f"sour{qdac_channel_fast}:volt:mode fix")
# Set the channel output range
qdac.write(f"sour{qdac_channel_slow}:rang low")
# Set the channel output filter
qdac.write(f"sour{qdac_channel_slow}:filt med")
# Set the slew rate in V/s to avoid transients when abruptly stepping the voltage -Must be within [0.01; 2e7] V/s
qdac.write(f"sour{qdac_channel_slow}:dc:volt:slew {1000}")
qdac.write(f"sour{qdac_channel_slow}:volt:mode fix")


# Send the QUA program to the OPX, which compiles and executes it
job = qm.execute(charge_stability)
# Creates results handles to fetch the data
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
n_handle = res_handles.get("iteration")
# Initialize empty vectors to store the global 'I' & 'Q' results
I_tot = []
Q_tot = []
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
for i in range(n_points_slow):  # Loop over y-voltages
    # Set voltage
    # Update the QDAC level
    qdac.write(f"sour{qdac_channel_slow}:volt {voltage_values_slow[i]}")

    for j in range(n_points_fast):  # Loop over x-voltages
        # Set voltage
        # Resume the QUA program (escape the 'pause' statement)
        qdac.write(f"sour{qdac_channel_fast}:volt {voltage_values_fast[i]}")

        job.resume()
        # Wait until the program reaches the 'pause' statement again, indicating that the QUA program is done
        wait_until_job_is_paused(job)
        
    # Wait until the data of this run is processed by the stream processing
    I_handle.wait_for_values(i + 1)
    Q_handle.wait_for_values(i + 1)
    n_handle.wait_for_values(i + 1)
    # Fetch the data from the last OPX run corresponding to the current LO frequency
    I = np.concatenate(I_handle.fetch(i)["value"])
    Q = np.concatenate(Q_handle.fetch(i)["value"])
    iteration = n_handle.fetch(i)["value"][0]
    # Update the list of global results
    I_tot.append(I)
    Q_tot.append(Q)
    # Progress bar
    progress_counter(iteration, n_points_slow)
    plt.subplot(121)
    plt.cla()
    plt.pcolor(n_points_fast, n_points_slow[:len(I_tot)], I_tot)
    plt.subplot(122)
    plt.cla()
    plt.pcolor(n_points_fast, n_points_slow[:len(Q_tot)], Q_tot)
    plt.pause(0.1)

# Interrupt the FPGA program
job.halt()
