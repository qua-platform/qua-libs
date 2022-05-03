from scipy.signal import find_peaks

from qualang_tools.bakery import baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
from qm import SimulationConfig
from matplotlib import pyplot as plt

N = 25
M = 5  # size of XY8 block

t_min = 100  # ns
t_max = 100.5  # ns
dt = 0.1  # ns
t_vec = np.arange(t_min, t_max + dt / 2, dt)
t_len = len(t_vec)
sample_rate = round(1 / dt * 1e9)

## These assertions need to pass in order for the sequence to be valid
# Checks that the size of the block is a multiple of the clock cycle (4ns)
assert 2 * 8 * M * dt % 4 == 0
# Checks that the total number of iterations can be created from the basic block
assert N % M == 0


def bake_X(bb):
    bb.reset_frame("qubit")
    bb.play("pi", "qubit")


def bake_Y(bb):
    bb.frame_rotation_2pi(0.25, "qubit")
    bb.play("pi", "qubit")
    bb.reset_frame("qubit")


def bake_XY8(bb, tt):
    bake_X(bb)
    bb.wait(tt, "qubit")
    bake_Y(bb)
    bb.wait(tt, "qubit")
    bake_X(bb)
    bb.wait(tt, "qubit")
    bake_Y(bb)
    bb.wait(tt, "qubit")
    bake_Y(bb)
    bb.wait(tt, "qubit")
    bake_X(bb)
    bb.wait(tt, "qubit")
    bake_Y(bb)
    bb.wait(tt, "qubit")
    bake_X(bb)


xy8_block_list = []  # Stores the baking objects which contains the XY8 blocks of size M
for i in range(t_len):  # Create the different baked sequences
    start_end_t = round((t_vec[i] - pi_len) * sample_rate / 1e9)
    middle_2t = round((2 * t_vec[i] - pi_len) * sample_rate / 1e9)
    time = 0
    with baking(config, sampling_rate=sample_rate, padding_method="none") as b:
        b.wait(start_end_t, "qubit")
        for j in range(M):
            bake_XY8(b, middle_2t)
            b.wait(middle_2t, "qubit")
        b.delete_samples(-middle_2t)
        b.wait(start_end_t, "qubit")
    # Append the baking object in the list to call it from the QUA program
    xy8_block_list.append(b)

with program() as XY8:
    k = declare(int)
    j = declare(int)
    times = declare(int, size=100)
    counts = declare(int)
    counts_st = declare_stream()

    play("init", "laser_AOM")
    align()
    with for_(j, 0, j < len(xy8_block_list), j + 1):
        with switch_(j, unsafe=True):
            for i in range(len(xy8_block_list)):
                with case_(i):
                    play("pi_half", "qubit")
                    with for_(k, 0, k < N / M - 1, k + 1):
                        xy8_block_list[i].run()
                        # There's a missing pi_len between blocks
                        wait(pi_len // 4, "qubit")
                    xy8_block_list[i].run()
                    play("pi_half", "qubit")
        align()
        play("init", "laser_AOM")
        measure("readout", "spcm1", None, time_tagging.analog(times, meas_len // 4, counts))
        save(counts, counts_st)

    with stream_processing():
        counts_st.buffer(len(xy8_block_list)).save("counts")

qmm = QuantumMachinesManager()
job = qmm.simulate(
    config,
    XY8,
    SimulationConfig(int((9 * t_max * N * 8) / 4)),
)
plt.figure()
samps = job.get_simulated_samples()
samps.con1.plot()

plt.show()

# The following line of code adds a vertical line at each theoretical peak of the sequence, this can be used to verify
# that the peaks are at the correct condition.
# Finding the peaks according to channel 1, which is the X pulses (pi half pulses)
channel_1 = samps.con1.analog["1"]
peaks = find_peaks(channel_1)[0]
for i in range(4):
    first_peak = peaks[(int(4 * N + 2)) * i]
    t = t_vec[i]
    # Asserts that the sequence has the correct duration
    assert first_peak + (2 * 8 * N) * t == peaks[(int(4 * N + 1)) * (i + 1) + i]

    # Plots the peak
    first_peak += 0.5  # Gaussian is symmetric, peak is between the two plateau points
    plt.vlines(first_peak, 0, 0.45)
    for j in range(8 * N):
        plt.vlines(first_peak + (2 * j + 1) * t, 0, 0.45)
    plt.vlines(first_peak + (2 * 8 * N) * t, 0, 0.45)
