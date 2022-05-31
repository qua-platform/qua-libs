"""
Starts a counter which reports the current counts from the SPCM.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

total_integration_time = int(0.1e9)  # 100ms
single_integration_time_ns = int(500e3)  # 500us
single_integration_time_cycles = int(single_integration_time_ns // 4)
n_count = int(total_integration_time / single_integration_time_ns)

with program() as counter:
    times = declare(int, size=1000)
    counts = declare(int)
    total_counts = declare(int)
    counts_st = declare_stream()
    n = declare(int)
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

qm = qmm.open_qm(config)

job = qm.execute(counter)
res_handles = job.result_handles
counts_handle = res_handles.get("counts")
counts_handle.wait_for_values(1)
time = []
counts = []


def on_close(event):
    event.canvas.stop_event_loop()
    job.halt()


f = plt.figure()
f.canvas.mpl_connect("close_event", on_close)

while res_handles.is_processing():
    plt.cla()
    new_counts = counts_handle.fetch_all()
    counts.append(new_counts["value"] / total_integration_time / 1000)
    time.append(new_counts["timestamp"] * 1e-9)
    if len(time) > 50:
        plt.plot(time[-50:], counts[-50:])
    else:
        plt.plot(time, counts)

    plt.xlabel("time [s]")
    plt.ylabel("counts [kcps]")
    plt.title("Counter")
    plt.pause(0.1)
