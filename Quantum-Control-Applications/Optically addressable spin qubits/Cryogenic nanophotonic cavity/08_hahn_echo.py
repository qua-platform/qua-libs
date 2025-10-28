"""
08_hahn_echo.py: Measures T2.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm import LoopbackInterface
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
t_min = 4  # in clock cycles units (must be >= 4)
t_max = 500  # in clock cycles units
dt = 20  # in clock cycles units
t_vec = np.arange(t_min, t_max + 0.1, dt)  # +0.1 to include t_max in array
n_avg = 1e6

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_vec": t_vec,
    "config": config,
}
###################
# The QUA program #
###################
with program() as hahn_echo:
    counts1 = declare(int)  # saves number of photon counts
    counts2 = declare(int)  # saves number of photon counts
    times1 = declare(int, size=100)
    times2 = declare(int, size=100)
    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    counts_1_st = declare_stream()  # stream for counts
    counts_2_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, t_vec)):
            # initialization
            play("laser_ON", "F_transition")
            align()
            play("laser_ON", "A_transition")
            play("switch_ON", "excited_state_mw")
            align()

            play("pi_half", "Yb")  # Pi/2 pulse to qubit
            wait(t, "Yb")  # variable delay in spin Echo
            play("pi", "Yb")  # Pi pulse to qubit
            wait(t, "Yb")  # variable delay in spin Echo
            play("pi_half", "Yb")  # Pi/2 pulse to qubit

            align()

            # readout laser
            play("laser_ON", "A_transition", duration=int(meas_len // 4))
            # does it needs buffer to prevent damage?

            align()

            # decay readout
            measure("readout", "SNSPD", time_tagging.analog(times1, meas_len, counts1))
            save(counts1, counts_1_st)  # save counts
            wait(100)

            align()

            # initialization
            play("laser_ON", "F_transition")
            align()
            play("laser_ON", "A_transition")
            play("switch_ON", "excited_state_mw")
            align()

            # pulse sequence
            play("pi_half", "Yb")  # Pi/2 pulse to qubit
            wait(t, "Yb")  # variable delay in spin Echo
            play("pi", "Yb")  # Pi pulse to qubit
            wait(t, "Yb")  # variable delay in spin Echo
            frame_rotation_2pi(0.5, "Yb")  # Turns next pulse to -x
            play("pi_half", "Yb")  # Pi/2 pulse to qubit
            reset_frame("Yb")

            align()

            # readout laser
            play("laser_ON", "A_transition", duration=int(meas_len // 4))
            # does it needs buffer to prevent damage?

            align()  # global align

            # decay readout
            measure("readout", "SNSPD", time_tagging.analog(times2, meas_len, counts2))
            save(counts2, counts_2_st)  # save counts
            wait(100)

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_1_st.buffer(len(t_vec)).average().save("counts1")
        counts_2_st.buffer(len(t_vec)).average().save("counts2")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = False
if simulate:
    simulation_config = SimulationConfig(duration=28_000, simulation_interface=LoopbackInterface(["1"]))
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hahn_echo, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    qm = qmm.open_qm(config)
    # execute QUA program
    job = qm.execute(hahn_echo)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts1", "counts2", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        counts1, counts2, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot data
        plt.cla()
        plt.plot(4 * t_vec, counts1 / 1000 / (meas_len / u.s), counts2 / 1000 / (meas_len / u.s))
        plt.xlabel("Dephasing time [ns]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Hahn Echo")
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts1_data": counts1})
    save_data_dict.update({"counts2_data": counts2})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
