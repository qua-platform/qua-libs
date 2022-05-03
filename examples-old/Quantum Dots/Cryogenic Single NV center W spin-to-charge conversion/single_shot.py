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

n_avg = 1e6

with program() as time_rabi:

    # Declare QUA variables
    ###################
    times_dark = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    counts_dark = declare(int)  # variable to save the total number of photons
    counts_dark_st = declare_stream()  # stream for 'counts'

    times_bright = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    counts_bright = declare(int)  # variable to save the total number of photons
    counts_bright_st = declare_stream()  # stream for 'counts'

    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        # m = -1 state readout (neutral state - bright)
        assign(counts_bright, 0)
        play("charge_init", "laser_705nm")  # charge initialization
        play("spin_init", "laser_E12")  # spin initialization
        align("qubit", "laser_E12", "laser_705nm")

        play("pi", "qubit")  # Pi pulse to qubit

        align("qubit", "laser_EX1151nm")
        play("SCC", "laser_EX1151nm")  # Spin to Charge Conversion
        align("laser_EX1151nm", "laser_EX12", "SNSPD")
        play("charge_read", "laser_EX12")  # Charge readout
        measure(
            "photon_count",
            "SNSPD",
            None,
            time_tagging.analog(times_bright, meas_len, counts_bright),
        )  # photon count
        save(counts_bright, counts_bright_st)  # save counts

        # m = 0 readout (charged state - dark)
        align()  # global align
        assign(counts_dark, 0)
        play("charge_init", "laser_705nm")  # charge initialization
        play("spin_init", "laser_E12")  # spin initialization
        play("SCC", "laser_EX1151nm")  # Spin to Charge Conversion
        align("laser_EX1151nm", "laser_EX12", "SNSPD")
        play("charge_read", "laser_EX12")  # Charge readout
        measure(
            "photon_count",
            "SNSPD",
            None,
            time_tagging.analog(times_dark, meas_len, counts_dark),
        )  # photon count
        save(counts_dark, counts_dark_st)  # save counts

        save(n, n_st)  # save number of iteration inside for_loop

    # Stream processing
    ###################
    with stream_processing():
        counts_bright_st.histogram([[i, i] for i in range(100)]).save("counts_bright")
        counts_dark_st.histogram([[i, i] for i in range(100)]).save("counts_dark")
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    qmm.simulate(config, time_rabi, SimulationConfig(10000)).get_simulated_samples().con1.plot()
else:
    job = qm.execute(time_rabi)  # execute QUA program

    res_handle = job.result_handles  # get access to handles
    vec_bright_handle = res_handle.get("counts_bright")
    vec_bright_handle.wait_for_values(1)
    vec_dark_handle = res_handle.get("counts_dark")
    vec_dark_handle.wait_for_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)

    while vec_bright_handle.is_processing():
        try:
            counts_bright = vec_bright_handle.fetch_all()
            counts_dark = vec_dark_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1

        except Exception as e:
            pass

        else:
            plt.plot(range(100), counts_bright / iteration)  # kcps
            plt.plot(range(100), counts_dark / iteration)  # kcps
            plt.xlabel("Photon number [N]")
            plt.ylabel("Fraction of occurrences")
            plt.title("Single shot readout")
            plt.pause(0.1)
            plt.clf()

    counts_bright = vec_bright_handle.fetch_all()
    counts_dark = vec_dark_handle.fetch_all()
    plt.plot(range(100), counts_bright / n_avg)  # kcps
    plt.plot(range(100), counts_dark / n_avg)  # kcps
    plt.xlabel("Photon number [N]")
    plt.ylabel("Fraction of occurrences")
    plt.title("Single shot readout")
    plt.savefig("time_rabi.png")
