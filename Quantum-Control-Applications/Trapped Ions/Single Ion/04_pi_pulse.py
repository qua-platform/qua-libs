"""Pi pulse calibration via time Rabi."""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

from configuration import config, pi_len, qop_ip, cluster_name
from macros import doppler_cool, state_preparation, measure_state, plot_state_and_histogram

u = unit(coerce_to_integer=True)

num_shots = 100
# Pulse durations in clock cycles (4 ns), scanned up to twice the nominal pi pulse
durations = np.arange(4, 2 * (pi_len // 4) + 1, 24)
durations_ns = 4 * durations

with program() as pi_pulse:
    t = declare(int)
    n = declare(int)
    counts = declare(int)
    state = declare(int)
    times = declare(int, size=1000)
    counts_st = declare_stream()
    state_st = declare_stream()
    n_st = declare_stream()

    with for_(*from_array(t, durations)):
        with for_(n, 0, n < num_shots, n + 1):
            doppler_cool()
            state_preparation()
            play("x180", "qubit", duration=t)
            align()
            measure_state(counts, times, state)
            save(counts, counts_st)
            save(state, state_st)
            wait(1000 * u.ns)
        save(n, n_st)

    with stream_processing():
        counts_st.buffer(num_shots).save_all("counts")
        state_st.buffer(num_shots).save_all("state")
        n_st.save("iteration")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
simulate = False

if simulate:
    job = qmm.simulate(config, pi_pulse, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = None
    job = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(pi_pulse)
        results = fetching_tool(job, data_list=["counts", "state", "iteration"], mode="live")

        fig, (ax_rabi, ax_hist) = plt.subplots(1, 3, figsize=(12, 4))
        interrupt_on_close(fig, job)

        p1_per_t = []
        n_pts = 0
        while results.is_processing():
            counts, states, iteration = results.fetch_all()
            progress_counter(iteration, len(durations) * num_shots, start_time=results.get_start_time())
            counts = np.asarray(counts).ravel()
            states = np.asarray(states).ravel()
            n_blocks = len(states) // num_shots
            p1_per_t = [np.mean(states[i * num_shots : (i + 1) * num_shots]) for i in range(n_blocks)]
            point_idx = (len(counts) - 1) // num_shots
            point_counts = counts[point_idx * num_shots :]
            n_pts = min(len(p1_per_t), len(durations))
            
            plot_state_and_histogram(fig, ax_rabi, ax_hist, 
                                     f"Pi pulse calibration",
                                     "Pi pulse duration [ns]", "P(|1⟩)", durations_ns[:n_pts], p1_per_t[:n_pts],
                                     "Photon counts", "Shots", point_counts)

            plt.pause(0.1)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        if qm is not None:
            qm.close()
        plt.show()
