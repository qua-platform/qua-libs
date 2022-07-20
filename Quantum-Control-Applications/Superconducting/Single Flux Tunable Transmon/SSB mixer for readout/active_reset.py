"""
IQ_blobs.py: template for performing a single shot discrimination and active reset
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import active_reset_single, active_reset_until_success
import matplotlib.pyplot as plt
from qualang_tools.analysis import two_state_discriminator

##############################
# Program-specific variables #
##############################
threshold = ge_threshold  # Threshold used for ge state discrimination
n_shot = 10000  # Number of acquired shots
max_tries = 10  # Maximum number of tries for active reset (no feedback if set to 0)
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)

###################
# The QUA program #
###################

with program() as active_reset_until_success_prog:
    n = declare(int)  # Averaging index
    I_g = declare(fixed)
    Q_g = declare(fixed)
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    tries_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Measure and perform active reset
        I_g, total_tries = active_reset_until_success(threshold=threshold, max_tries=max_tries, Ig=I_g)
        # Check active feedback by measuring again
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
        )
        # Save data to the stream processing
        save(I_g, Ig_st)
        save(Q_g, Qg_st)
        save(total_tries, tries_st)

    with stream_processing():
        Ig_st.save_all("Ig")
        Qg_st.save_all("Qg")
        tries_st.average().save("average_tries")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, active_reset_until_success_prog, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(active_reset_until_success_prog)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["Ig", "Qg", "average_tries"], mode="wait_for_all")
    # Fetch results
    Ig, Qg, average_tries = results.fetch_all()
    # Plot data
    fig = plt.figure(figsize=(7, 5))
    plt.cla()
    plt.scatter(Ig, Qg, color="b", alpha=0.1)
    plt.axvline(threshold, color="k")
    plt.axis("equal")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.legend(["Ground"])
    plt.title(f"Active reset after {average_tries:.0f}/{max_tries} tries in average.")
