"""
counter.py: Starts a counter which reports the current counts from the SPCM.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

total_integration_time = int(100 * u.ms)  # 100ms
single_integration_time_ns = int(500 * u.us)  # 500us
single_integration_time_cycles = single_integration_time_ns // 4
n_count = int(total_integration_time / single_integration_time_ns)

with program() as counter:
    times = declare(int, size=1000)
    counts = declare(int)
    total_counts = declare(int)
    n = declare(int)
    counts_st = declare_stream()
    with infinite_loop_():
        with for_(n, 0, n < n_count, n + 1):
            play("laser_ON", "AOM", duration=single_integration_time_cycles)
            measure("readout", "SPCM", None, time_tagging.analog(times, single_integration_time_ns, counts))
            assign(total_counts, total_counts + counts)
        save(total_counts, counts_st)
        assign(total_counts, 0)

    with stream_processing():
        counts_st.with_timestamps().save("counts")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=4000)  # in clock cycles
    job_sim = qmm.simulate(config, counter, simulation_config)
    # Simulate blocks python until the simulation is done
    job_sim.get_simulated_samples().con1.plot()
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
        time.append(new_counts["timestamp"] / u.s)  # Convert timestams to seconds
        plt.cla()
        if len(time) > 50:
            plt.plot(time[-50:], counts[-50:])
        else:
            plt.plot(time, counts)

        plt.xlabel("time [s]")
        plt.ylabel("counts [kcps]")
        plt.title("Counter")
        plt.pause(0.1)
