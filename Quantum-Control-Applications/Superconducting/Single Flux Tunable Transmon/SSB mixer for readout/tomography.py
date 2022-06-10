"""
tomography.py: template for acquiring the qubit tomography by scanning the phase of the 2nd pi/2 pulse
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import ge_averaged_measurement
import matplotlib.pyplot as plt
import numpy as np

##############################
# Program-specific variables #
##############################
n_avg = 1000  # Number of averaging loops
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)

# Phase scan parameters in units of 2pi
n_phases = 101
phase_array = np.linspace(-0.5, 0.5, n_phases)


###################
# The QUA program #
###################
with program() as rabi_amp_freq:
    n = declare(int)  # Average index
    phase = declare(fixed)  # Sweeping phase
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # Calibrate the ground and excited states readout for deriving the Bloch vector
    Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(cooldown_time, n_avg)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(phase, phase_array.tolist()):
            # First pi/2 pulse
            play("pi_half", "qubit")
            # Rotate the phase of the second pulse
            frame_rotation_2pi(phase, "qubit")
            # Second pi/2 pulse
            play("pi_half", "qubit")
            align("qubit", "resonator")
            # Measure the resonator
            measure(
                "short_readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            # Wait for the resonator to cooldown
            wait(cooldown_time, "resonator", "qubit")
            # Save data to the stream processing
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(n_phases).average().save("I")
        Q_st.buffer(n_phases).average().save("Q")
        Ig_st.average().save("Ig")
        Qg_st.average().save("Qg")
        Ie_st.average().save("Ie")
        Qe_st.average().save("Qe")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi_amp_freq)
    res_handles = job.result_handles
    I_handles = res_handles.get("I")
    Q_handles = res_handles.get("Q")
    I_handles.wait_for_values(1)
    Q_handles.wait_for_values(1)
    Ie_handles = res_handles.get("Ie")
    Qe_handles = res_handles.get("Qe")
    Ie_handles.wait_for_values(1)
    Qe_handles.wait_for_values(1)
    Ig_handles = res_handles.get("Ig")
    Qg_handles = res_handles.get("Qg")
    Ig_handles.wait_for_values(1)
    Qg_handles.wait_for_values(1)

    # Live plotting
    fig = plt.figure(figsize=(15, 15))
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    xplot = phase_array * 2 * np.pi
    while res_handles.is_processing():

        I = I_handles.fetch_all()
        Q = Q_handles.fetch_all()
        Ie = Ie_handles.fetch_all()
        Qe = Qe_handles.fetch_all()
        Ig = Ig_handles.fetch_all()
        Qg = Qg_handles.fetch_all()

        # Phase of ground and excited states
        phase_g = np.angle(Ig + 1j * Qg)
        phase_e = np.angle(Ie + 1j * Qe)
        # Qubit phase
        phase = np.unwrap(np.angle(I + 1j * Q))
        # Population in excited state
        pop = (phase - phase_g) / (phase_e - phase_g)

        # Plots
        plt.subplot(311)
        plt.cla()
        plt.plot(xplot, np.sqrt(I**2 + Q**2))
        plt.xlabel("2nd pi/2 hhase-shift [rad]")
        plt.ylabel("Readout amplitude")
        plt.subplot(312)
        plt.cla()
        plt.plot(xplot, phase)
        plt.xlabel("2nd pi/2 hhase-shift [rad]")
        plt.ylabel("Readout phase [rad]")
        plt.subplot(313)
        plt.cla()
        plt.plot(xplot, pop)
        plt.xlabel("2nd pi/2 hhase-shift [rad]")
        plt.ylabel("Population")
        plt.pause(0.1)
