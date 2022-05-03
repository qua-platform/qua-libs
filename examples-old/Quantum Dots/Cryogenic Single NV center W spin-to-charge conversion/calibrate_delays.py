from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig
from qm import LoopbackInterface

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

initial_delay_cycles = 500 // 4  # delay before laser (units of clock cycles = 4 ns)
laser_len_cycles = 3000 // 4  # laser duration length (units of clock cycles = 4 ns)
mw_len_cycles = 1000 // 4  # MW duration length (units of clock cycles = 4 ns)
t_vec = np.arange(0, meas_len, 1)
n_avg = 5e5

with program() as calib_delays:

    # Declare QUA variables
    ###################
    times = declare(int, size=100)  # 'size' defines the max number of photons to be counted
    times_st = declare_stream()  # stream for 'times'
    counts = declare(int)  # variable to save the total number of photons
    i = declare(int)  # variable used to save data
    n = declare(int)  # variable used in for_loop
    n_st = declare_stream()  # stream for 'iteration'

    # Pulse sequence
    ################
    with for_(n, 0, n < n_avg, n + 1):

        wait(initial_delay_cycles, "laser_EX705nm")  # wait before starting PL
        play("PL", "laser_EX705nm", duration=laser_len_cycles)  # Photoluminescence

        wait(initial_delay_cycles + (laser_len_cycles - mw_len_cycles) // 2, "qubit")  # delay the microwave pulse
        play("mw", "qubit", duration=mw_len_cycles)  # play microwave pulse

        measure("photon_count", "SNSPD", None, time_tagging.analog(times, meas_len, counts))  # photon count on SNSPD

        with for_(i, 0, i < counts, i + 1):
            save(times[i], times_st)  # save time tags to stream

        save(n, n_st)  # save number of iteration inside for_loop

    # Stream processing
    ###################
    with stream_processing():
        times_st.histogram([[i, i] for i in range(meas_len)]).save("times_hist")  # histogram
        n_st.save("iteration")

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=int(1e4),
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, calib_delays, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:
    job = qm.execute(calib_delays)  # execute QUA program

    res_handle = job.result_handles  # get access to handles
    times_hist_handle = res_handle.get("times_hist")
    times_hist_handle.wait_for_values(1)
    iteration_handle = res_handle.get("iteration")
    iteration_handle.wait_for_values(1)

    while times_hist_handle.is_processing():
        try:
            times_hist = times_hist_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            plt.plot(t_vec, times_hist / 1000 / (meas_len * 1e-9) / iteration)  # kcps
            plt.xlabel("t [ns]")
            plt.ylabel(f"counts [kcps]")
            # plt.ylabel(f'counts [kcps / {meas_len / 1000}us]')
            plt.title("Delays")
            plt.pause(0.1)
            plt.clf()

        except Exception as e:
            pass

    times_hist = times_hist_handle.fetch_all()
    plt.plot(t_vec, times_hist / 1000 / (meas_len * 1e-9) / n_avg)  # kcps vs. meas_len
    plt.xlabel("t [ns]")
    # plt.ylabel(f'counts [kcps / {meas_len / 1000}us]')
    plt.ylabel(f"counts [kcps]")
    plt.title("Delays")
