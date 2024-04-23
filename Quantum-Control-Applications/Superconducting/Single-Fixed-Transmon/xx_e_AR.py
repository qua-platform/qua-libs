# %%
"""
        IQ BLOBS conditional reset
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.analysis import two_state_discriminator

###################
# The QUA program #
###################

n_runs = 10_000  # Number of runs

with program() as conditional_reset:

    n = declare(int)
    n_st  = declare_stream()
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    
    with for_(n, 0, n < n_runs, n + 1):

        save(n, n_st)

        wait(thermalization_time * u.ns)

        align()

        measure(
            "midcircuit_readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
        )
        # CONDITIONAL RESET TO |g>
        play("x180", 'qubit', condition=I_g > midciruit_ge_threshold)

        align()

        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
        )
        save(I_g, I_g_st)
        save(Q_g, Q_g_st)

        wait(thermalization_time * u.ns)

        align()

        measure(
            "midcircuit_readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
        )

        # CONDITIONAL RESET TO |e>
        play("x180", 'qubit', condition=I_e < midciruit_ge_threshold)

        align()

        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
        )
        save(I_e, I_e_st)
        save(Q_e, Q_e_st)

    with stream_processing():
        n_st.save('iteration')
        # Save all streamed points for plotting the IQ blobs
        I_g_st.save_all("I_g")
        Q_g_st.save_all("Q_g")
        I_e_st.save_all("I_e")
        Q_e_st.save_all("Q_e")


# %%
#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, conditional_reset, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config, close_other_machines=False)
        print("Open QMs: ", qmm.list_open_quantum_machines())
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(conditional_reset)

        fetch_names = ["iteration"]
        results = fetching_tool(job, fetch_names, mode="live")

        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_runs, start_time=results.start_time)

        fetch_names.append("I_g")
        fetch_names.append("Q_g")
        fetch_names.append("I_e")
        fetch_names.append("Q_e")

        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, fetch_names)

        res = results.fetch_all()
        
        angle_val, threshold_val, fidelity_val, gg_val, ge_val, eg_val, ee_val = two_state_discriminator(res[1], res[2], res[3], res[4], True, False)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=False)
    
    plt.figure()
    plt.plot(res[1], res[2], ".", alpha=0.1, markersize=2)
    plt.plot(res[3], res[4], ".", alpha=0.1, markersize=2)
    plt.axis('equal')
    plt.axvline(x=0, linestyle='--', color='k')
    plt.axhline(y=0, linestyle='--', color='k')

    plt.tight_layout()
    plt.show(block=False)

    plt.figure()

    plt.hist(res[1], bins=30, color='blue', alpha=0.5, label=f"Fidelity: {fidelity_val}")
    plt.hist(res[3], bins=30, color='orange', alpha=0.5)
    plt.axvline(x=threshold_val)

    
# %%