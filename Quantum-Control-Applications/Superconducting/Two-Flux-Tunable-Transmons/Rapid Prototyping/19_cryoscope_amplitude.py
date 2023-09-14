"""
WARNING: the digital filters will add a global delay --> need to recalibrate readout !!
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from scipy import signal
import scipy.optimize
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from macros import expdecay, filter_calc
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
qb = qb1

qb.z.flux_pulse_amp = 0.1





q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

config = build_config(machine)



cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 500
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
flux_amp_array = np.linspace(0, 0.45, 1001)

###################
# The QUA program #
###################


with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    flux_amp = declare(fixed)  # Flux amplitude pre-factor
    flag = declare(bool)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(flux_amp, flux_amp_array)):
            with for_each_(flag, [True, False]):
                play("x90", qb.name + "_xy")

                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                play("const" * amp(flux_amp), qb.name+"_z")
                align(qb.name+"_xy", qb.name+"_z")
                # Wait some time to ensure that the 2nd x90 pulse will arrive after the flux pulse
                wait(20 * u.ns)
                align()
                with if_(flag):
                    play("x90", qb.name + "_xy")
                with else_():
                    play("y90", qb.name + "_xy")

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                assign(state[0], I[0] > qb1.ge_threshold)
                assign(state[1], I[1] > qb2.ge_threshold)
                save(state[0], state_st[0])
                save(state[1], state_st[1])
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # Qubit state
        state_st[0].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state1")
        state_st[1].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state2")
        # I_st[0].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("I1")
        # I_st[1].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("I2")
        # Q_st[0].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("Q1")
        # Q_st[1].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
if simulate:
    job = qmm.simulate(config, cryoscope, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "state1", "state2"], mode="live")
    xplot = flux_amp_array * qb.z.flux_pulse_amp
    while results.is_processing():
        n, state1, state2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)
        # Accumulated phase: angle between Sx and Sy
        if qb == qb1:
            state = state1
        else:
            state = state2
        Sx = state[:, 0] * 2 - 1
        Sy = state[:, 1] * 2 - 1
        qubit_state = Sx + 1j * Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[0]
        # Filtering and derivative of the phase to get the averaged frequency
        coarse_detuning = qubit_phase / (2 * np.pi * qb.z.flux_pulse_length /u.s)
        # Quadratic fit of detuning versus flux pulse amplitude
        pol = np.polyfit(xplot, coarse_detuning, deg=2)
        plt.suptitle("Cryoscope")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, state1, ".-")
        plt.title(f"{qb1.name}")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("State")
        plt.legend(("Sx", "Sy"))
        plt.subplot(222)
        plt.cla()
        plt.title(f"{qb2.name}")
        plt.plot(xplot, state2, ".-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.legend(("Sx", "Sy"))
        plt.subplot(212)
        plt.cla()
        plt.plot(xplot, coarse_detuning / u.MHz, ".")
        plt.plot(xplot, np.polyval(pol, xplot) / u.MHz, "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [MHz]")
        plt.title(f"{qb.name}")
        plt.legend(("data", "Fit"), loc="upper right")
        plt.tight_layout()
        plt.pause(5)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# machine._save("current_state.json")
