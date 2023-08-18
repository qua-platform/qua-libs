"""
state_tomography.py:
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

n_avg = 1_000_000

with program() as state_tomography:
    n = declare(int)
    n_st = declare_stream()
    c = declare(int)
    counts = declare(int)
    counts_st = declare_stream()
    times = declare(int, size=100)
    times_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(c, 0, c <= 2, c + 1):
            # Add here whatever state you want to characterize
            with switch_(c):
                with case_(0):  # projection along X
                    play("-y90", "NV")
                with case_(1):  # projection along Y
                    play("x90", "NV")
                with case_(2):  # projection along Z
                    wait(pi_len_NV * u.ns, "NV")
            align()
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts
            wait(100 * u.ns, "AOM1")
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        counts_st.buffer(3).average().save("counts")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=28000)
    job = qmm.simulate(config, state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)
    # execute QUA program
    job = qm.execute(state_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    counts, iteration = results.fetch_all()

    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Zero order approximation
    rho = 0.5 * (I + counts[0] * sigma_x + counts[1] * sigma_y + counts[2] * sigma_z)
    print(f"The density matrix is:\n{rho}")
