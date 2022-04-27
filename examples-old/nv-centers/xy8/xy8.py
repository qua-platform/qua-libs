from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np

qmm = QuantumMachinesManager()

qm = qmm.open_qm(config)

t_min = 4
t_max = 100
dt = 1
t_vec = np.arange(t_min, t_max, dt)

repsN = 3
simulate = True


def xy8_n(n):
    # Performs the full xy8_n sequence. First block is outside loop, to avoid delays caused from either the loop or from
    # two consecutive wait commands.
    # Assumes it starts frame at x, if not, need to reset_frame before
    wait(t, "qubit")

    xy8_block()

    with for_(i, 0, i < n - 1, i + 1):
        wait(2 * t, "qubit")
        xy8_block()

    wait(t, "qubit")


def xy8_block():
    # A single XY8 block, ends at x frame.
    play("pi", "qubit")  # 1 X
    wait(2 * t, "qubit")

    frame_rotation(np.pi / 2, "qubit")
    play("pi", "qubit")  # 2 Y
    wait(2 * t, "qubit")

    reset_frame("qubit")
    play("pi", "qubit")  # 3 X
    wait(2 * t, "qubit")

    frame_rotation(np.pi / 2, "qubit")
    play("pi", "qubit")  # 4 Y
    wait(2 * t, "qubit")

    play("pi", "qubit")  # 5 Y
    wait(2 * t, "qubit")

    reset_frame("qubit")
    play("pi", "qubit")  # 6 X
    wait(2 * t, "qubit")

    frame_rotation(np.pi / 2, "qubit")
    play("pi", "qubit")  # 7 Y
    wait(2 * t, "qubit")

    reset_frame("qubit")
    play("pi", "qubit")  # 8 X


with program() as xy8:
    # Realtime FPGA variables
    a = declare(int)  # For averages
    i = declare(int)  # For XY8-N
    t = declare(int)  # For tau
    times = declare(int, size=100)  # Time-Tagging
    counts = declare(int)  # Counts
    counts_ref = declare(int)
    diff = declare(int)  # Diff in counts between counts & counts_ref
    counts_st = declare_stream()  # Streams for server processing
    counts_ref_st = declare_stream()
    diff_st = declare_stream()

    with for_(a, 0, a < 1e6, a + 1):
        play("laser", "qubit")

        with for_(t, t_min, t <= t_max, t + dt):  # Implicit Align
            # Play meas (pi/2 pulse at x)
            play("pi_half", "qubit")
            xy8_n(repsN)
            play("pi_half", "qubit")
            measure("readout", "qubit", None, time_tagging.raw(times, 300, counts))
            # Time tagging done here, in real time

            # Plays ref (pi/2 pulse at -x)
            play("pi_half", "qubit")
            xy8_n(repsN)
            frame_rotation(np.pi, "qubit")
            play("pi_half", "qubit")
            reset_frame("qubit")  # Such that next tau would start in x.
            measure("readout", "qubit", None, time_tagging.raw(times, 300, counts_ref))
            # Time tagging done here, in real time

            # save counts:
            assign(diff, counts - counts_ref)
            save(counts, counts_st)
            save(counts_ref, counts_ref_st)
            save(diff, diff_st)

    with stream_processing():
        counts_st.buffer(len(t_vec)).average().save("dd")
        counts_ref_st.buffer(len(t_vec)).average().save("ddref")
        diff_st.buffer(len(t_vec)).average().save("diff")

if simulate:
    job = qm.simulate(xy8, SimulationConfig(20000))
    # job = qmm.simulate(config, xy8, SimulationConfig(20000, include_analog_waveforms=True))
    samps = job.get_simulated_samples()
    an1 = samps.con1.analog["1"].tolist()
    an2 = samps.con1.analog["2"].tolist()
    dig1 = samps.con1.digital["1"]
    plt.plot(an1)
    plt.plot(an2)
    plt.plot(dig1)

else:
    job = qm.execute(xy8, duration_limit=0, time_limit=0)
    dd_handle = job.result_handles.get("dd")
    dd_ref_handle = job.result_handles.get("ddref")
    diff_handle = job.result_handles.get("diff")
