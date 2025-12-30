"""
03_counter.py: Starts a counter which reports the current counts from the SPCM.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

##################
#   Parameters   #
##################
# Parameters Definition
total_integration_time = int(100 * u.ms)  # 100ms
single_integration_time_ns = int(500 * u.us)  # 500us
single_integration_time_cycles = single_integration_time_ns // 4
n_count = int(total_integration_time / single_integration_time_ns)

###################
# The QUA program #
###################
with program() as counter:
    times = declare(int, size=1000)
    counts = declare(int)
    total_counts = declare(int)
    n = declare(int)
    counts_st = declare_stream()
    with infinite_loop_():
        with for_(n, 0, n < n_count, n + 1):
            play("laser_ON", "AOM", duration=single_integration_time_cycles)
            measure("readout", "SNSPD", None, time_tagging.analog(times, single_integration_time_ns, counts))
            assign(total_counts, total_counts + counts)
        save(total_counts, counts_st)
        assign(total_counts, 0)

    with stream_processing():
        counts_st.with_timestamps().save("counts")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=5_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, counter, simulation_config)
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

    job = qm.execute(counter)
    # Get results from QUA program
    res_handles = job.result_handles
    counts_handle = res_handles.get("counts")
    counts_handle.wait_for_values(1)
    time = []
    counts = []
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while res_handles.is_processing():
        new_counts = counts_handle.fetch_all()
        counts.append(new_counts["value"] / total_integration_time / 1000)
        time.append(new_counts["timestamp"] / u.s)  # Convert timestamps to seconds
        plt.cla()
        if len(time) > 50:
            plt.plot(time[-50:], counts[-50:])
        else:
            plt.plot(time, counts)

        plt.xlabel("time [s]")
        plt.ylabel("counts [kcps]")
        plt.title("Counter")
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
