"""Carrier transition spectroscopy."""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

from configuration import config, qop_ip, cluster_name
from macros import doppler_cool, measure_state, state_preparation, plot_state_and_histogram

u = unit(coerce_to_integer=True)

probe_len = 20 * u.us
num_shots = 100
freqs = np.arange(190, 211, 1, dtype=int) * u.MHz

with program() as carrier_spectroscopy:
    f = declare(int)
    n = declare(int)
    counts = declare(int)
    state = declare(int)
    times = declare(int, size=1000)
    counts_st = declare_stream()
    state_st = declare_stream()
    n_st = declare_stream()

    with for_(*from_array(f, freqs)):
        update_frequency("qubit", f)
        with for_(n, 0, n < num_shots, n + 1):
            doppler_cool()
            state_preparation()
            play("constant", "qubit", duration=probe_len // 4)
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
    job = qmm.simulate(config, carrier_spectroscopy, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = None
    job = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(carrier_spectroscopy)
        results = fetching_tool(job, data_list=["counts", "state", "iteration"], mode="live")

        fig, (ax_state, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))
        interrupt_on_close(fig, job)

        while results.is_processing():
            counts, states, iteration = results.fetch_all()
            progress_counter(iteration, len(freqs) * num_shots, start_time=results.get_start_time())
            counts = np.asarray(counts).ravel()
            states = np.asarray(states).ravel()
            n_blocks = len(counts) // num_shots
            mean_per_freq = [np.mean(counts[i * num_shots : (i + 1) * num_shots]) for i in range(n_blocks)]
            p1_per_freq = [np.mean(states[i * num_shots : (i + 1) * num_shots]) for i in range(n_blocks)]
            point_idx = (len(counts) - 1) // num_shots # which data point after start
            point_counts = counts[point_idx * num_shots :] # counts for the last data point
            n_pts = min(len(mean_per_freq), len(freqs))

            plot_state_and_histogram(fig, ax_state, ax_hist, 
                                     f"Carrier spectroscopy",
                                     "Qubit IF [MHz]", "P(|1⟩)", freqs[:n_pts], p1_per_freq[:n_pts],
                                     "Photon counts", "Shots", point_counts)
            
            plt.pause(0.1)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        if qm is not None:
            qm.close()
        plt.show()
