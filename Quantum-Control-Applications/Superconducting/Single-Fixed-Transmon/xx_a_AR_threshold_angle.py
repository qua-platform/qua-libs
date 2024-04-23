# %%
"""
        IQ BLOBS
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.loops import from_array
from macros import iq_blobs_analysis, fit_polynomial

###################
# The QUA program #
###################

n_runs = 10_000  # Number of runs

# NOTE: for testing proper rotation change to "rotated" 

a_max = 1.5
a_min = 0.5
a_step = 0.1
amps = np.arange(a_min, a_max, a_step)


with program() as iq_blobs_amp:

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

    a = declare(fixed)

    with for_(n, 0, n < n_runs, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):
            
            # GROUND iq blobs preparation
            wait(thermalization_time * u.ns)
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

            align()

            # EXCITED iq blobs preparation
            wait(thermalization_time * u.ns)
            play("x180", 'qubit')
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
        I_g_st.buffer(len(amps)).save_all(f"I_g")
        Q_g_st.buffer(len(amps)).save_all(f"Q_g")
        I_e_st.buffer(len(amps)).save_all(f"I_e")
        Q_e_st.buffer(len(amps)).save_all(f"Q_e")


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
    job = qmm.simulate(config, iq_blobs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try: 
        # Open the quantum machine
        qm = qmm.open_qm(config, close_other_machines=False)
        print("Open QMs: ", qmm.list_open_quantum_machines())
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(iq_blobs_amp)

        fetch_names = ["iteration"]
        results = fetching_tool(job, fetch_names, mode="live")
        
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_runs, start_time=results.start_time)

        fetch_names.append(f"I_g")
        fetch_names.append(f"Q_g")
        fetch_names.append(f"I_e")
        fetch_names.append(f"Q_e")

        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, fetch_names)

        res = results.fetch_all()

        angle_val, threshold_val, fidelity_val = iq_blobs_analysis(res[1], res[2], res[3], res[4], method="fidelity")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=False)
        
    plt.figure()
    plt.plot(amps, fidelity_val)
    plt.title("fidelity")

    plt.tight_layout()
    plt.show(block=False)

    plt.figure()

    threshold_coef = {}
    angle_coef = {}

    plt.plot(amps, threshold_val)
    threshold_coef, polyy = fit_polynomial(amps, threshold_val, 1)
    plt.plot(amps, polyy(amps))
    print("Fitting coefficients are slope:", threshold_coef[0], "and intercept:", threshold_coef[1])
    plt.title(f"thresholds")

    plt.tight_layout()
    plt.show(block=False)

    plt.figure()

    plt.plot(amps, np.unwrap(angle_val) * 180 / np.pi)
    plt.title("angles")
    angle_coef, polyy = fit_polynomial(amps, np.unwrap(angle_val) * 180 / np.pi, 0)
    plt.plot(amps, polyy(amps))
    print("The fitting coefficients:", angle_coef)

    plt.tight_layout()
    plt.show(block=False)

    np.savez('th_angle_vs_amp.npz', amps=amps, angle_coef=angle_coef, threshold_coef=threshold_coef)

# %%
