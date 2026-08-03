"""Doppler cooling followed by fluorescence detection."""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit

from configuration import config, qop_ip, cluster_name
from macros import doppler_cool, state_preparation, measure_state, plot_state_and_histogram

u = unit(coerce_to_integer=True)

num_shots = 200

with program() as cooling_and_detection:
    n = declare(int)
    counts = declare(int)
    state = declare(int)
    times = declare(int, size=1000)
    counts_st = declare_stream()
    state_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < num_shots, n + 1):
        doppler_cool()
        state_preparation()
        measure_state(counts, times, state)
        save(counts, counts_st)
        save(state, state_st)
        save(n, n_st)
        wait(1000 * u.ns)

    with stream_processing():
        counts_st.save_all("counts")
        state_st.save_all("state")
        n_st.save("iteration")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
simulate = False

if simulate:
    job = qmm.simulate(config, cooling_and_detection, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = None
    job = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(cooling_and_detection)
        results = fetching_tool(job, data_list=["counts", "state", "iteration"], mode="live")

        fig, (ax_state, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))
        interrupt_on_close(fig, job)

        while results.is_processing():
            counts, states, iteration = results.fetch_all()
            progress_counter(iteration, num_shots, start_time=results.get_start_time())
            states = np.asarray(states).ravel()
            shots = np.arange(1, len(states) + 1)
            p1_per_shot = np.cumsum(states) / shots
                        
            plot_state_and_histogram(fig, ax_state, ax_hist, 
                                     f"Detection", 
                                     "Shot", "P(|1⟩)", shots, p1_per_shot, 
                                     "Photon counts", "Shots", counts)

            plt.pause(0.1)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        if qm is not None:
            qm.close()
        plt.show()
