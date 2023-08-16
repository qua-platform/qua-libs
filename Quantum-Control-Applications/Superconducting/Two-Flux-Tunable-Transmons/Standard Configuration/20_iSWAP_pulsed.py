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
# The variable const_flux_len is defined in the configuration
flux_waveform = np.array([const_flux_amp] * const_flux_len)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", "q1_z", wf)
            b.play("flux_pulse", "q1_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, const_flux_len)


###################
# The QUA program #
###################

dc0_q2 = config["controllers"]["con1"]["analog_outputs"][8]["offset"]
dc0_q1 = config["controllers"]["con1"]["analog_outputs"][7]["offset"]
amps = np.arange(1.2, 1.8, 0.005)
cooldown_time = 5 * u.us
n_avg = 60

with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    segment = declare(int)  # Flux pulse segment

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):
            with for_(segment, 0, segment <= const_flux_len, segment + 1):
                play("x180", "q2_xy")
                align()
                wait(4)

                with switch_(segment):
                    for j in range(0, const_flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run(amp_array=[("q1_z", a)])

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(const_flux_len + 1).buffer(len(amps)).average().save("I1")
        Q_st[0].buffer(const_flux_len + 1).buffer(len(amps)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(const_flux_len + 1).buffer(len(amps)).average().save("I2")
        Q_st[1].buffer(const_flux_len + 1).buffer(len(amps)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config)

simulate = True
if simulate:
    job = qmm.simulate(config, iswap, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iswap)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    xplot = np.arange(0, const_flux_len + 0.1, 1)
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + dc0_q2, xplot, I1.T)
        plt.title("q1 - I")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + dc0_q2, xplot, Q1.T)
        plt.title("q1 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + dc0_q2, xplot, I2.T)
        plt.title("q2 - I")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * const_flux_amp + dc0_q2, xplot, Q2.T)
        plt.title("q2 - Q")
        plt.xlabel("FLux amplitude (V)")
        plt.tight_layout()
        plt.pause(0.1)
    # np.savez(save_dir / 'iswap', I1=I1, Q1=Q1, I2=I2, ts=ts, amps=amps)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()