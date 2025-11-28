"""
06b_power_rabi.py: A Rabi experiment sweeping the amplitude of the MW pulse.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
a_min = 0.1  # proportional factor to the pulse amplitude
a_max = 1  # proportional factor to the pulse amplitude
da = 0.02
a_vec = np.arange(a_min, a_max + da / 2, da)  # +da/2 to include a_max
n_avg = 1e6  # number of iterations

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "a_vec": a_vec,
    "config": config,
}

###################
# The QUA program #
###################
with program() as power_rabi:
    counts = declare(int)  # variable for number of counts
    times = declare(int, size=100)
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    counts_st = declare_stream()  # stream for counts
    n_st = declare_stream()  # stream to save iterations

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, a_vec)):
            # initialization
            play("laser_ON", "F_transition")
            align()
            play("laser_ON", "A_transition")
            play("switch_ON", "excited_state_mw")
            align()

            # pulse sequence
            play("pi" * amp(a), "Yb")  # pulse of varied amplitude

            align()

            # readout laser
            play("laser_ON", "A_transition", duration=int(meas_len // 4))
            # does it needs buffer to prevent damage?

            align()

            # decay readout
            measure("readout", "SNSPD", None, time_tagging.analog(times, meas_len, counts))
            save(counts, counts_st)  # save counts
            wait(100)

        save(n, n_st)  # save number of iteration inside for_loop

    with stream_processing():
        counts_st.buffer(len(a_vec)).average().save("counts")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

simulate = False
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=28_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, power_rabi, simulation_config)
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
    job = qm.execute(power_rabi)
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
        # Plot data
        plt.cla()
        plt.plot(a_vec * pi_amp_NV, counts / 1000 / (meas_len * 1e-9))
        plt.xlabel("Amplitude [volts]")
        plt.ylabel("Intensity [kcps]")
        plt.title("Power Rabi")
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"counts_data": counts})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
