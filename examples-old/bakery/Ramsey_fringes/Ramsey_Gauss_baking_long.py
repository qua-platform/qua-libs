from qualang_tools.bakery import baking
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from RamseyGauss_configuration import *
from qm import SimulationConfig
from matplotlib import pyplot as plt
from qualang_tools.bakery.bakery import deterministic_run

dephasing = 0  # phase for the 2nd Tpihalf gauss pulse
wait_time_cc = 100

t_min = 0
t_max = 1000
dt = 1
t_vec = np.arange(0, t_max + dt / 2, dt)

short_ramsey_baking_list = []  # Stores the baking objects
# Create the different baked sequences, corresponding to the different taus up to 16 ns
for i in range(2 * Tpihalf):
    with baking(config, padding_method="left") as b:
        init_delay = 2 * Tpihalf  # Put initial delay to ensure that all of the pulses will have the same length
        b.wait(init_delay, "drive")  # We first wait the entire duration.

        # We add the 2nd pi_half pulse with the phase 'dephasing' (Confusingly, the first pulse will be added later)
        # Play uploads the sample in the original config file (here we use an existing pulse in the config)
        b.frame_rotation_2pi(dephasing, "drive")
        b.play("pi_half", "drive")

        # We reset frame such that the first pulse will be at zero phase
        # and such that we will not accumulate phase between iterations.
        b.reset_frame("drive")
        # We add the 1st pi_half pulse. It will be added with the frame at time init_delay - i, which will be 0.
        b.play_at("pi_half", "drive", t=init_delay - i)

    # Append the baking object in the list to call it from the QUA program
    short_ramsey_baking_list.append(b)

long_ramsey_1st_pulse_baking_list = []  # Stores the baking objects
for i in range(4):
    with baking(config, padding_method="left") as b:
        # We play the first pulse then add a wait of i (which goes from 0 to 3).
        # The padding is on the left, so the extra samples are added there
        b.play("pi_half", "drive")
        b.wait(i, "drive")

    # Append the baking object in the list to call it from the QUA program
    long_ramsey_1st_pulse_baking_list.append(b)


# You can retrieve and see the pulse you built for each baking object by modifying
# index of the waveform
# plt.figure()
# for i in range(n_samples):
#     baked_pulse_I = config["waveforms"][f"drive_baked_wf_I_{i}"]["samples"]
#     baked_pulse_Q = config["waveforms"][f"drive_baked_wf_Q_{i}"]["samples"]
#     plt.plot(baked_pulse_I, label=f"pulse{i}_I")
#     plt.plot(baked_pulse_Q, label=f"pulse{i}_Q")
# plt.title("Baked Ramsey sequences (envelope)")
# plt.legend()


with program() as RamseyGauss:  # to measure Rabi flops every 1ns starting from 0ns
    I = declare(fixed, value=0.0)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    t = declare(int)  # The duration in ns
    t_cycles = declare(int)  # The rounded down duration in cycles
    t_left_ns = declare(int)  # The leftover duration in ns
    i_avg = declare(int)

    I_stream = declare_stream()
    Q_stream = declare_stream()

    with for_(i_avg, 0, i_avg < 1000, i_avg + 1):
        with for_(t, 0, t <= t_max, t + dt):
            # Wait for cavity cooldown, very short just for the example.
            wait(10)
            with if_(t < 2 * Tpihalf):
                deterministic_run(short_ramsey_baking_list, t, unsafe=True)
                align()
            with else_():
                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                assign(t_cycles, t_cycles - Tpihalf // 4)
                with switch_(t_left_ns, unsafe=True):
                    for j in range(4):
                        with case_(j):
                            long_ramsey_1st_pulse_baking_list[j].run()
                            wait(t_cycles, "drive")
                            frame_rotation_2pi(dephasing, "drive")
                            play("pi_half", "drive")
                            reset_frame("drive")
                            align()

            align()
            play("chargecav", "resonator")  # to charge the cavity
            measure(
                "readout",
                "resonator",
                None,
                demod.full("cos", I1, "out1"),
                demod.full("sin", Q1, "out1"),
                demod.full("cos", I2, "out2"),
                demod.full("sin", Q2, "out2"),
            )
            assign(I, I1 + Q2)
            assign(Q, I2 - Q1)
            save(I, I_stream)
            save(Q, Q_stream)

    with stream_processing():
        I_stream.buffer(len(t_vec)).average().save("Iall")
        Q_stream.buffer(len(t_vec)).average().save("Qall")

qmm = QuantumMachinesManager()
job = qmm.simulate(config, RamseyGauss, SimulationConfig(30000))
samps = job.get_simulated_samples()
plt.figure()
an1 = samps.con1.analog["1"].tolist()
an3 = samps.con1.analog["3"].tolist()
dig1 = samps.con1.digital["1"]
dig3 = samps.con1.digital["3"]

plt.plot(an1)
plt.plot(an3)

plt.show()
