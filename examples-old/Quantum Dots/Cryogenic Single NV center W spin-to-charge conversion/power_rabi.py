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

a_min = 0.1  # proportional factor to the pulse amplitude
a_max = 1  # proportional factor to the pulse amplitude
da = 0.2
a_vec = np.arange(a_min, a_max + da / 2, da)  # +da/2 to include a_max
n_avg = 1e6  # number of iterations

with program() as power_rabi:

    # Declare QUA variables
    ###################
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    counts = declare(int)  # variable to save the total number of photons
    counts_st = declare_stream()  # stream for 'counts'
    a = declare(fixed)  # variable to sweep over the amplitude
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        with for_(a, a_min, a < a_max + da / 2, a + da):
            play("charge_init", "laser_705nm")  # charge initialization
            play("spin_init", "laser_E12")  # spin initialization
            align("qubit", "laser_E12", "laser_705nm")
            play("gauss" * amp(a), "qubit")  # gaussian pulse of varied amplitude
            align("qubit", "laser_EX", "SNSPD")
            play("EX", "laser_EX")  # Spin readout to m = 0 transition
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )  # photon count
            save(counts, counts_st)  # save counts

        save(n, n_st)  # save number of iteration inside for_loop

    # Stream processing
    ###################
    with stream_processing():
        counts_st.buffer(len(a_vec)).average().save("counts")
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    qmm.simulate(config, power_rabi, SimulationConfig(10000)).get_simulated_samples().con1.plot()
else:
    job = qm.execute(power_rabi)  # execute QUA program

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
            plt.plot(a_vec, counts / 1000 / (meas_len * 1e-9) / iteration)  # kcps
            plt.xlabel("Amplitude")
            plt.ylabel("counts [kcps]")
            plt.title("Power Rabi")
            plt.pause(0.1)
            plt.clf()

    counts = vec_handle.fetch_all()
    plt.plot(a_vec, counts / 1000 / (meas_len * 1e-9) / n_avg)  # kcps
    plt.xlabel("amplitude")
    plt.ylabel("counts [kcps]")
    plt.title("Power Rabi")
