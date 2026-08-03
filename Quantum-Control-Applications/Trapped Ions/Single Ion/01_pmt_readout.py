"""Live PMT photon counter."""

import matplotlib.pyplot as plt
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

from configuration import config, qop_ip, cluster_name
from macros import doppler_cool

u = unit(coerce_to_integer=True)

window_len = 500 * u.us
n_windows = 200

with program() as pmt_readout:
    times = declare(int, size=1000)
    counts = declare(int)
    total_counts = declare(int)
    i = declare(int)
    counts_st = declare_stream()

    with infinite_loop_():
        with for_(i, 0, i < n_windows, i + 1):
            doppler_cool()
            measure("readout", "pmt", time_tagging.analog(times, window_len, counts))
            assign(total_counts, total_counts + counts)
        save(total_counts, counts_st)
        assign(total_counts, 0)

    with stream_processing():
        counts_st.with_timestamps().save_all("counts")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)
simulate = False

if simulate:
    job = qmm.simulate(config, pmt_readout, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = None
    job = None
    try:
        qm = qmm.open_qm(config, close_other_machines=True)
        job = qm.execute(pmt_readout)
        counts_handle = job.result_handles.get("counts")
        counts_handle.wait_for_values(1)

        fig = plt.figure()
        interrupt_on_close(fig, job)
        time_s, rate_kcps = [], []
        last_idx = 0

        while job.result_handles.is_processing():
            new_idx = counts_handle.count_so_far()
            batch = counts_handle.fetch(slice(last_idx, new_idx))
            last_idx = new_idx
            rate_kcps.extend(batch["value"] / (window_len * n_windows) / 1000)
            time_s.extend(batch["timestamp"] / u.s)
            plt.cla()
            plt.plot(time_s[-50:], rate_kcps[-50:])
            plt.xlabel("Time [s]")
            plt.ylabel("Counts [kcps]")
            plt.title("PMT readout")
            plt.pause(0.1)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        if qm is not None:
            qm.close()
