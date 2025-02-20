"""
Example program for conditional feedback based on time tagger results.
"""

import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *

# Measurement parameters
meas_len = 2000
resolution = 1000  # ps
t_vec = np.arange(0, meas_len * 1e3, 1)
delay = 16 * u.ns

###################
# The QUA program #
###################

with program() as time_tagger:
    i = declare(int)
    n = declare(int)
    times = declare(int, size=100)
    times2 = declare(int, size=100)
    times_count_st = declare_stream()
    times_no_count_st = declare_stream()
    counts = declare(int)
    counts2 = declare(int)
    counts_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < 10_000_000, n + 1):
        wait(delay, "control_eom")
        wait(100 * u.ns + delay + 4 * u.ns, "control_eom2")
        play("control", "control_eom", duration=16 * u.ns)

        measure("readout", "time_tagger", None, time_tagging.high_res(times, 48, counts))

        with if_(counts > 0):
            play("control", "control_eom2", duration=320 * u.ns)
            measure("readout", "time_tagger2", None, time_tagging.high_res(times2, meas_len, counts2))
            with for_(i, 0, i < counts2, i + 1):
                save(times2[i], times_count_st)
        with else_():
            play("control", "control_eom2", duration=160 * u.ns)
            measure("readout", "time_tagger2", None, time_tagging.high_res(times2, meas_len, counts2))
            with for_(i, 0, i < counts2, i + 1):
                save(times2[i], times_no_count_st)

        save(counts, counts_st)

    with stream_processing():
        counts_st.save_all("counts")
        times_count_st.histogram([
            [i, i + (resolution - 1)] for i in range(0, meas_len * int(1e3), resolution)
        ]).save("times_count_hist")
        times_no_count_st.histogram([
            [i, i + (resolution - 1)] for i in range(0, meas_len * int(1e3), resolution)
        ]).save("times_no_count_hist")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=opx_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1_000)
    job = qmm.simulate(config, time_tagger, simulation_config)
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path='./')
    plt.show()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(time_tagger)
    res = job.result_handles
    job.result_handles.wait_for_all_values()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot((t_vec[::resolution] + resolution / 2) / 1000 * u.ns, res.get("times_count_hist").fetch_all())
    ax[1].plot((t_vec[::resolution] + resolution / 2) / 1000 * u.ns, res.get("times_no_count_hist").fetch_all())
    ax[0].set_xlabel("t [ns]")
    ax[0].set_ylabel("counts")
    ax[0].set_title("TimeTag Photon Count")
    ax[1].set_xlabel("t [ns]")
    ax[1].set_ylabel("counts")
    ax[1].set_title("TimeTag No Photon Count")
    fig.tight_layout()
    plt.show()