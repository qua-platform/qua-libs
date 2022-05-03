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

t_min = 4  # in clock cycles units (4ns)
t_max = 1000000  # in clock cycles units (4ns)
dt = 20000  # in clock cycles units (4ns)
t_vec = np.arange(t_min, t_max + dt / 2, dt)
n_avg = 1e6

with program() as t1_measurement:

    # Declare QUA variables
    ###################
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted

    counts = declare(int)  # variable to save the total number of photons
    counts_ref = declare(int)
    counts_diff = declare(int)
    counts_st = declare_stream()  # stream for 'counts'
    counts_ref_st = declare_stream()
    counts_diff_st = declare_stream()

    t = declare(int)  # variable to sweep over in time
    n = declare(int)  # variable to for_loop
    n_st = declare_stream()  # stream to save iterations

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        with for_(t, t_min, t <= t_max, t + dt):

            # m = 0 readout (charged state - dark)
            play("charge_init", "laser_705nm")  # charge initialization
            play("spin_init", "laser_E12")  # spin initialization
            play("SCC", "laser_EX1151nm")  # Spin to Charge Conversion
            align("laser_EX1151nm", "laser_EX12", "SNSPD")
            play("charge_read", "laser_EX12")  # Charge readout
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts_ref),
            )
            save(counts_ref, counts_ref_st)

            align()  # global align

            # m = -1 state readout (neutral state - bright)
            play("charge_init", "laser_705nm")  # charge initialization
            play("spin_init", "laser_E12")  # spin initialization
            align("qubit", "laser_E12", "laser_705nm")

            play("pi", "qubit")  # Pi pulse to qubit

            align("qubit", "laser_EX1151nm")
            wait(t, "laser_EX1151nm")  # variable delay before SCC
            play("SCC", "laser_EX1151nm")  # Spin to Charge Conversion
            align("laser_EX1151nm", "laser_EX12", "SNSPD")
            play("charge_read", "laser_EX12")  # Charge readout
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )

            assign(counts_diff, counts - counts_ref)
            save(counts_ref, counts_ref_st)
            save(counts_diff, counts_diff_st)

        save(n, n_st)  # save number of iteration inside for_loop

    # Stream processing
    ###################
    with stream_processing():
        counts_st.buffer(len(t_vec)).average().save("counts")
        counts_ref_st.buffer(len(t_vec)).average().save("counts_ref")
        counts_diff_st.buffer(len(t_vec)).map(FUNCTIONS.average()).average().save("counts_diff")
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    qmm.simulate(config, t1_measurement, SimulationConfig(10000)).get_simulated_samples().con1.plot()
else:
    job = qm.execute(t1_measurement)  # execute QUA program

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
            plt.plot(4 * t_vec, counts / 1000 / (meas_len * 1e-9) / iteration)  # kcps
            plt.xlabel("t [ns]")
            plt.ylabel("counts [kcps]")
            plt.title("T1 measurement")
            plt.pause(0.1)
            plt.clf()

    counts = vec_handle.fetch_all()
    plt.plot(4 * t_vec, counts / 1000 / (meas_len * 1e-9) / n_avg)  # kcps
    plt.xlabel("t [ns]")
    plt.ylabel("counts [kcps]")
    plt.title("T1 measurement")
