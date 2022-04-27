import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from configuration import *

################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

###################
# The QUA program #
###################

# Declare Python variables
###########################
total_time = 100e6  # in ns
n_count = int(total_time / meas_len)

with program() as counter:

    # Declare QUA variables
    ###################
    times = declare(int, size=1000)  # variable to save time tag values
    counts = declare(int)  # saves number of photon counts
    total_counts = declare(int)
    counts_st = declare_stream()
    n = declare(int)

    # Pulse sequence
    ################
    with infinite_loop_():

        assign(total_counts, 0)  # set total_counts to zero

        with for_(n, 0, n < n_count, n + 1):
            play("PL", "laser_EX705nm", duration=meas_len // 4)  # Photoluminescence
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )  # photon count on SNSPD
            assign(total_counts, total_counts + counts)

        save(total_counts, counts_st)  # save counts to stream variable

    # Stream processing
    ###################
    with stream_processing():
        counts_st.with_timestamps().save("counts")

#######################
# Simulate or execute #
#######################

job = qm.execute(counter)
res_handle = job.result_handles
vec_handle = res_handle.get("counts")
vec_handle.wait_for_values(1)
time = []
counts = []

while vec_handle.is_processing():
    try:
        new_counts = vec_handle.fetch_all()

    except Exception as e:
        print(e)
    else:
        counts.append(new_counts["value"] / total_time / 1000)
        time.append(new_counts["timestamp"] * 1e-9)
        if len(time) > 50:
            plt.plot(time[-50:], counts[-50:])
        else:
            plt.plot(time, counts)

        plt.xlabel("time [s]")
        plt.ylabel("counts [kcps]")
        plt.title("Counter")
        plt.pause(0.1)
        plt.clf()
