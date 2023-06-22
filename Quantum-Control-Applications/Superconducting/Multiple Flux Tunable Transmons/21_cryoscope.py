# %%
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery import baking

##########
# baking #
##########

# FLux pulse waveform generation
zeros_before_pulse = 20  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 20  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
flux_waveform = np.array(
    [0.0] * zeros_before_pulse + [const_flux_amp] * const_flux_len + [0.0] * zeros_after_pulse
)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", "q2_z", wf)
            b.play("flux_pulse", "q2_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, len(flux_waveform))
step_response = [1.0] * const_flux_len
xplot = np.arange(0, len(flux_waveform) + 0.1, 1)

###################
# The QUA program #
###################
dc0_q2 = config["controllers"]["con1"]["analog_outputs"][8]["offset"]
dc0_q1 = config["controllers"]["con1"]["analog_outputs"][7]["offset"]
cooldown_time = 5 * 50 * u.us
n_avg = 1000

with program() as cz:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    segment = declare(int)  # Flux pulse segment
    flag = declare(bool)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(segment, 0, segment <= const_flux_len + total_zeros, segment + 1):
            with for_each_(flag, [True, False]):
                play("x90", "q2_xy")

                align()

                wait(
                    20 // 4
                )  # TODO: this wait() creates time between pi and flux pulses, it is possible to calibrate the delay

                with switch_(segment):
                    for j in range(0, len(flux_waveform) + 1):
                        with case_(j):
                            square_pulse_segments[j].run()

                wait((const_flux_len + 100) * u.ns, "q2_xy")

                with if_(flag):
                    play("x90", "q2_xy")
                with else_():
                    play("y90", "q2_xy")

                align()
                multiplexed_readout(I, I_st, Q, Q_st, weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("I1")
        Q_st[0].buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("Q1")
        # resonator 2
        I_st[1].buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("I2")
        Q_st[1].buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip)

simulate = False
if simulate:
    job = qmm.simulate(config, cz, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cz)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, I1, ".-")
        plt.title("q1 - I")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, Q1, ".-")
        plt.title("q1 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, I2, ".-")
        plt.title("q2 - I")
        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, Q2, ".-")
        plt.title("q2 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)

plt.show()
# np.savez(save_dir/'cz', I1=I1, Q1=Q1, I2=I2, Q2=Q2, ts=ts, amps=amps)

# %%