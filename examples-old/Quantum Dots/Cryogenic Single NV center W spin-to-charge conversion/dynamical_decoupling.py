from time import sleep

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
import matplotlib.pyplot as plt
from configuration import *
from qm import SimulationConfig

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)

t_min = 50  # in clock cycles units (4ns)
t_max = 500  # in clock cycles units (4ns)
dt = 10  # in clock cycles units (4ns)
t_vec = np.arange(t_min, t_max + dt / 2, dt)
n_avg = 1e6

# Options:
# CPMG
# XY8
# Ramsey (same as CPMG/XY8 with 0)
# Echo (same as CPMG with 1)
dd_sequence = "CPMG"
N = 4


def dd(t_int, N):
    # Assumes it starts frame at x, if not, reset_frame before
    wait(t_int, "nv")

    if dd_sequence == "Ramsey" or N == 0:
        return
    elif dd_sequence == "Echo":
        N = 1

    i = declare(int)
    tt = declare(int)
    assign(tt, 2 * t_int)

    if dd_sequence == "Echo" or dd_sequence == "CPMG":
        frame_rotation_2pi(0.25, "qubit")

    dd_block(tt)

    with for_(i, 0, i < N - 1, i + 1):
        wait(tt, "nv")
        dd_block(tt)

    wait(t_int, "nv")
    if dd_sequence == "Echo" or dd_sequence == "CPMG":
        reset_frame("qubit")


def dd_block(tt):
    if dd_sequence == "CPMG" or dd_sequence == "Echo":
        # Assumes already in the y frame
        play("pi", "nv")  # 1 X

    elif dd_sequence == "XY8":
        # Assumes we're in the x frame
        play("pi", "nv")  # 1 X
        wait(tt, "nv")

        frame_rotation_2pi(0.25, "nv")
        play("pi", "nv")  # 2 Y
        wait(tt, "nv")

        reset_frame("nv")
        play("pi", "nv")  # 3 X
        wait(tt, "nv")

        frame_rotation(np.pi / 2, "nv")
        play("pi", "nv")  # 4 Y
        wait(tt, "nv")

        play("pi", "nv")  # 5 Y
        wait(tt, "nv")

        reset_frame("nv")
        play("pi", "nv")  # 6 X
        wait(tt, "nv")

        frame_rotation(np.pi / 2, "nv")
        play("pi", "nv")  # 7 Y
        wait(tt, "nv")

        reset_frame("nv")
        play("pi", "nv")  # 8 X


with program() as dd_prog:
    times = declare(int, size=100)
    counts = declare(int)
    counts_st = declare_stream()
    counts_ref_st = declare_stream()
    t = declare(int)
    n = declare(int)
    with for_(n, 0, n < n_avg, n + 1):
        play("init", "laser")
        with for_(t, t_min, t <= t_max, t + dt):
            # Meas
            play("pi" * amp(0.5), "qubit")
            dd(t, N)
            play("pi" * amp(0.5), "qubit")
            align()
            play("init", "laser")
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )
            save(counts, counts_st)

            # Ref
            align()
            play("pi" * amp(0.5), "qubit")
            dd(t, N)
            frame_rotation_2pi(0.5, "qubit")
            play("pi" * amp(0.5), "qubit")
            align()
            play("init", "laser")
            measure(
                "photon_count",
                "SNSPD",
                None,
                time_tagging.analog(times, meas_len, counts),
            )
            save(counts, counts_ref_st)
            reset_frame("qubit")

    with stream_processing():
        counts_st.buffer(len(t_vec)).average().save("counts")
        counts_ref_st.buffer(len(t_vec)).average().save("counts_ref")

if True:
    config["elements"]["qubit"]["intermediate_frequency"] = 0
    qmm.simulate(config, dd_prog, SimulationConfig(5000)).get_simulated_samples().con1.plot()
else:
    job = qm.execute(dd_prog)
    res_handle = job.result_handles
    vec_handle = res_handle.get("counts")
    vec_ref_handle = res_handle.get("counts_ref")
    vec_handle.wait_for_values(1)
    while vec_handle.is_processing():
        try:
            counts = vec_handle.fetch_all()
            counts_ref = vec_ref_handle.fetch_all()

        except Exception as e:
            print(e)
        else:
            plt.plot(4 * t_vec, (counts - counts_ref) / (counts + counts_ref))
            plt.xlabel("t [ns]")
            plt.ylabel("Contrast")
            plt.title(dd_sequence)
            plt.pause(0.5)
            plt.clf()

    while vec_ref_handle.is_processing():
        sleep(1)

    counts = vec_handle.fetch_all()
    counts_ref = vec_ref_handle.fetch_all()
    plt.plot(4 * t_vec, (counts - counts_ref) / (counts + counts_ref))
    plt.xlabel("t [ns]")
    plt.ylabel("Contrast")
    plt.title(dd_sequence)
