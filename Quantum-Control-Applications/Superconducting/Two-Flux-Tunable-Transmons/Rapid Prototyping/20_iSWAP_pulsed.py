from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery import baking
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)

machine.qubits[active_qubits[1]].z.flux_pulse_amp = -0.104
machine.qubits[active_qubits[1]].z.flux_pulse_length = 52

config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].qubit_name + "_z"
q2_z = machine.qubits[active_qubits[1]].qubit_name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

qb = qb2
##########
# baking #
##########
# Flux pulse waveform generation
# The variable machine.qubits[qubit_index].z.flux_pulse_length is defined in the configuration
flux_waveform = np.array([qb.z.flux_pulse_amp] * qb.z.flux_pulse_length)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", qb.qubit_name + "_z", wf)
            b.play("flux_pulse", qb.qubit_name + "_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, qb.z.flux_pulse_length)


###################
# The QUA program #
###################
amps = np.arange(0.85, 1, 0.001)
cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 40

with program() as iswap:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    segment = declare(int)  # Flux pulse segment

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):
            with for_(segment, 0, segment <= qb.z.flux_pulse_length, segment + 1):
                play("x180", qb2.qubit_name + "_xy")
                align()
                with switch_(segment):
                    for j in range(0, qb.z.flux_pulse_length + 1):
                        with case_(j):
                            square_pulse_segments[j].run(amp_array=[(qb.qubit_name + "_z", a)])

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("I1")
        Q_st[0].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("I2")
        Q_st[1].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    job = qmm.simulate(config, iswap, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iswap)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    xplot = np.arange(0, qb.z.flux_pulse_length + 0.1, 1)
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.suptitle("iSWAP chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, I1.T)
        plt.plot(qb.z.iswap.level, qb.z.iswap.length, "r*")
        plt.title(f"{qb1.qubit_name} - I, f_01={int(qb1.xy.f_01 / u.MHz)} MHz")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, Q1.T)
        plt.plot(qb.z.iswap.level, qb.z.iswap.length, "r*")
        plt.title(f"{qb1.qubit_name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, I2.T)
        plt.plot(qb.z.iswap.level, qb.z.iswap.length, "r*")
        plt.title(f"{qb2.qubit_name} - I, f_01={int(qb2.xy.f_01 / u.MHz)} MHz")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, Q2.T)
        plt.plot(qb.z.iswap.level, qb.z.iswap.length, "r*")
        plt.title(f"{qb2.qubit_name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.tight_layout()
        plt.pause(5)

    # np.savez(save_dir / 'iswap', I1=I1, Q1=Q1, I2=I2, ts=ts, amps=amps)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
# qb.z.iswap.length =
# qb.z.iswap.level =
# machine._save("current_state.json")
