"""
IQ_blobs.py: template for performing a single shot discrimination and active reset
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import ge_singleshot_measurement
import matplotlib.pyplot as plt
from qualang_tools.analysis import two_state_discriminator

##############################
# Program-specific variables #
##############################
threshold = -9.4e-4  # Threshold for active feedback
n_shot = 10000  # Number of acquired shots
max_count = 100  # Maximum number of tries for active reset (no feedback if set to 0)
cooldown_time = u.to_clock_cycles(5 * qubit_T1)  # Cooldown time in clock cycles (4ns)

###################
# The QUA program #
###################

with program() as singleshot:
    n = declare(int)  # Averaging index
    counter = declare(int)  # Number of tries for active reset

    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Reset the number of tries
        assign(counter, 0)
        # Measure the g and e states
        I_g, Q_g, I_e, Q_e = ge_singleshot_measurement(cooldown_time)

        # Perform active feedback
        align()  # This align is not needed and  can be removed. It is just here to check the timmings with the simulator.
        # Use the single conditional play statement for integrating active reset in other protocols
        play("pi", "qubit", condition=(I_g < threshold))

        # Use a while loop and counter for other protocols and tests
        with while_((I_g < threshold) & (counter < max_count)):
            # Measure the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )
            # Play a pi pulse to get back to the ground state
            play("pi", "qubit", condition=(I_g < threshold))
            # Wait for the resonator to cooldown
            wait(cooldown_time, "resonator", "qubit")
            # Increment the number of tries
            assign(counter, counter + 1)

        # Save data to the stream processing
        save(I_g, Ig_st)
        save(Q_g, Qg_st)
        save(I_e, Ie_st)
        save(Q_e, Qe_st)

    with stream_processing():
        Ig_st.save_all("Ig")
        Qg_st.save_all("Qg")
        Ie_st.save_all("Ie")
        Qe_st.save_all("Qe")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, singleshot, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(singleshot)
    res_handles = job.result_handles

    Ie_handles = res_handles.get("Ie")
    Qe_handles = res_handles.get("Qe")
    Ie_handles.wait_for_values(1)
    Qe_handles.wait_for_values(1)
    Ig_handles = res_handles.get("Ig")
    Qg_handles = res_handles.get("Qg")
    Ig_handles.wait_for_values(1)
    Qg_handles.wait_for_values(1)

    # Live plotting
    fig = plt.figure(figsize=(7, 5))
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while res_handles.is_processing():
        I_e = Ie_handles.fetch_all()["value"]
        Q_e = Qe_handles.fetch_all()["value"]
        I_g = Ig_handles.fetch_all()["value"]
        Q_g = Qg_handles.fetch_all()["value"]
        plt.cla()
        plt.scatter(I_g[: min(len(I_g), len(Q_g))], Q_g[: min(len(I_g), len(Q_g))], color="b", alpha=0.1)
        plt.scatter(I_e[: min(len(I_e), len(Q_e))], Q_e[: min(len(I_e), len(Q_e))], color="r", alpha=0.1)
        plt.axis("equal")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.legend(["Ground", "Excited"])
        plt.pause(0.1)
angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, b_print=True, b_plot=True)
