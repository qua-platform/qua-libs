# %%
"""
conditional_reset_vs_delay
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from macros import iq_blobs_analysis
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_runs = 1_000  # Number of runs
delay_min = 4
delay_max = 10_000
delay_step = 500
delays = np.arange(delay_min, delay_max, delay_step)

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

    t = declare(int)

    with for_(n, 0, n < n_runs, n + 1):
        
        save(n, n_st)

        with for_(*from_array(t, delays)):

            wait(thermalization_time * u.ns)

            align()

            measure(
                "midcircuit_readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )

            align()

            # CONDITIONAL RESET TO |g>
            wait(t, 'qubit')
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

            align()

            # CONDITIONAL RESET TO |e>
            wait(t, 'qubit')
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
        # Save all streamed points for plotting the IQ blobs
        I_g_st.buffer(len(delays)).save_all(f"I_g")
        Q_g_st.buffer(len(delays)).save_all(f"Q_g")
        I_e_st.buffer(len(delays)).save_all(f"I_e")
        Q_e_st.buffer(len(delays)).save_all(f"Q_e")


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
        fetch_names.append(f"I_g")
        fetch_names.append(f"Q_g")
        fetch_names.append(f"I_e")
        fetch_names.append(f"Q_e")

        results = fetching_tool(job, fetch_names, mode="live")

        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_runs, start_time=results.start_time)

            max_len = len(res[1])
            _, _, fidelities = iq_blobs_analysis(res[1][:max_len], res[2][:max_len], res[3][:max_len], res[4][:max_len], method="fidelity")
            plt.cla()
            plt.plot(delays * 4, fidelities)
            plt.xlabel('Delays [ns]')
            plt.ylabel('Fidelity')

            plt.tight_layout()
            plt.pause(1)

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=False)

# %%
