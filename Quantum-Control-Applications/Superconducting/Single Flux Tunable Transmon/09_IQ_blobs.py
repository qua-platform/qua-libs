"""
IQ_blobs.py: template for performing a single shot discrimination and active reset
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import single_measurement, reset_qubit
from qualang_tools.analysis import two_state_discriminator

##############################
# Program-specific variables #
##############################
n_shot = 10000  # Number of acquired shots
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)

###################
# The QUA program #
###################

with program() as IQ_blob:
    n = declare(int)  # Averaging index
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    with for_(n, 0, n < n_shot, n + 1):
        # Reset qubit state
        reset_qubit("cooldown_time", cooldown_time=cooldown_time)
        # Measure the ground state
        align("qubit", "resonator")
        I_g, Q_g = single_measurement()
        # Reset qubit state
        reset_qubit("cooldown_time", cooldown_time=cooldown_time)
        # Excited state measurement
        align("qubit", "resonator")
        play("pi", "qubit")
        # Measure the excited state
        I_e, Q_e = single_measurement()
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
    job = qmm.simulate(config, IQ_blob, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(IQ_blob)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["Ie", "Qe", "Ig", "Qg"], mode="wait_for_all")
    # Fetch results
    I_e, Q_e, I_g, Q_g = results.fetch_all()
    # Plot data
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(I_g, Q_g, I_e, Q_e, b_print=True, b_plot=True)
    # If the readout fidelity is satisfactory enough, then the angle and threshold can be updated in the config file.
