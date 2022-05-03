from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig

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

f_min = -70e6  # begin freq sweep
f_max = 30e6  # end of freq sweep
df = 2e6  # freq step
f_vec = np.arange(f_min, f_max + df / 2, df)  # f_max + df/2 so that f_max is included
n_avg = 1e6  # number of averages

with program() as cw_odmr:

    # Declare QUA variables
    ###################
    times = declare(int, size=1000)  # string were time tags are saved
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations

    # Pulse sequence
    ################
    # with infinite_loop_():  # continuous live plot
    with for_(n, 0, n < n_avg, n + 1):

        with for_(f, f_min, f <= f_max, f + df):

            update_frequency("qubit", f)  # updated frequency
            align("qubit", "laser_EX705nm", "SNSPD")  # align all elements

            play("mw", "qubit", duration=int(meas_len // 4))  # play microwave pulse
            play("PL", "laser_EX705nm", duration=int(meas_len // 4))  # Photoluminescence
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )  # photon count on SNSPD

            save(counts, counts_st)  # save counts on stream
            save(n, n_st)  # save number of iteration inside for_loop

    # Stream processing
    ###################
    with stream_processing():
        counts_st.buffer(len(f_vec)).average().save("counts")
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    qmm.simulate(config, cw_odmr, SimulationConfig(int(1.1 * meas_len))).get_simulated_samples().con1.plot()
    plt.show()
else:
    job = qm.execute(cw_odmr)  # execute QUA program

    res_handle = job.result_handles  # get access to handles
    vec_handle = res_handle.get("counts")
    vec_handle.wait_for_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)

    while vec_handle.is_processing():
        try:
            counts = vec_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1

        except Exception as e:
            pass

        else:
            plt.plot(LO_freq + f_vec, counts / 1000 / (meas_len * 1e-9) / iteration)  # kcps
            plt.xlabel("f_vec [Hz]")
            plt.ylabel("counts [kcps]")
            plt.title("ODMR")
            plt.pause(0.1)
            plt.clf()
            print(iteration)

    counts = vec_handle.fetch_all()
    plt.plot(LO_freq + f_vec, counts / 1000 / (meas_len * 1e-9) / n_avg)  # kcps
    plt.xlabel("f_vec [Hz]")
    plt.ylabel("counts [kcps]")
    plt.title("ODMR")
