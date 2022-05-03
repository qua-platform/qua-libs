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
a_max = 1
da = 0.05
a_vec = np.arange(a_min, a_max + da / 2, da)
t_min = 4  # in clock cycles units (4ns)
t_max = 100  # in clock cycles units (4ns)
dt = 1  # in clock cycles units (4ns)
t_vec = np.arange(t_min, t_max + dt / 2, dt)
n_avg = 1e6  # number of iterations

with program() as power_rabi_2d:

    # Declare QUA variables
    ###################
    times = declare(int, size=100)
    counts = declare(int)
    counts_st = declare_stream()
    a = declare(fixed)
    t = declare(int)
    n = declare(int)
    n_st = declare_stream()  # stream to save iterations

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        with for_(t, t_min, t <= t_max, t + dt):
            with for_(a, a_min, a < a_max + da / 2, a + da):
                play("charge_init", "laser_705nm")  # charge initialization
                play("spin_init", "laser_E12")  # spin initialization
                align("qubit", "laser_E12", "laser_705nm")
                play("gauss" * amp(a), "qubit", duration=t)  # gaussian pulse of duration t cycles and varied amplitude
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
        counts_st.buffer(len(t_vec)).buffer(len(a_vec)).average().save("counts")
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    qmm.simulate(config, power_rabi_2d, SimulationConfig(10000)).get_simulated_samples().con1.plot()
else:
    job = qm.execute(power_rabi_2d)  # execute QUA program

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
            plt.pcolormesh(4 * t_vec, a_vec, counts / 1000 / (meas_len * 1e-9) / iteration)  # kcps
            plt.xlabel("t_int [ns]")
            plt.ylabel("amplitude")
            plt.title("2D Rabi")
            plt.pause(0.1)
            plt.clf()

    counts = vec_handle.fetch_all()
    plt.pcolormesh(4 * t_vec, a_vec, counts / 1000 / (meas_len * 1e-9) / n_avg)  # kcps
    plt.xlabel("t_int [ns]")
    plt.ylabel("amplitude")
    plt.title("2D Rabi")
