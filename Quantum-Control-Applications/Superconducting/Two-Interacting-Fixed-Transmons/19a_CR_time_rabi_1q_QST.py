from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout, one_qb_QST, plot_1qb_tomography_results
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################
times = np.arange(4, 200, 2)  # In clock cycles = 4ns
n_avg = 1000

with program() as CR_time_rabi_one_qst:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the qubit pulse duration
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, times)):
            with for_(c, 0, c < 3, c + 1):
                # |0> control - CR
                play("square_positive", "cr_c1t2", duration=t)
                align()
                one_qb_QST("q2_xy", pi_len, c)
                align()
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)

                align()  # global align

                # |1> control - CR
                play("x180", "q1_xy")
                align()
                play("square_positive", "cr_c1t2", duration=t)
                align()
                play("x180", "q1_xy")
                align()
                one_qb_QST("q2_xy", pi_len, c)
                align()
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(3).buffer(len(times)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(2).buffer(3).buffer(len(times)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(times)).average().save("Q2")

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
    job = qmm.simulate(config, CR_time_rabi_one_qst, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(CR_time_rabi_one_qst)
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot tomography
        plot_1qb_tomography_results(I2, times * 4)
    # Close the quantum machines at the end
    qm.close()
