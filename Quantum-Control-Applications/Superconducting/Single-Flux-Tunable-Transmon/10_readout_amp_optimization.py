from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.analysis import two_state_discriminator
from qualang_tools.loops import from_array

###################
# The QUA program #
###################
qubit_operation = "x180"

n_runs = 1000

cooldown_time = 5 * qubit_T1

amps = np.arange(0.1, 1.7, 0.1)


with program() as ro_amp_opt:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    amps_st = declare_stream()
    counter = declare(int, value=0)
    a = declare(fixed)

    with for_(*from_array(a, amps)):
        save(counter, amps_st)
        with for_(n, 0, n < n_runs, n + 1):
            measure(
                "readout" * amp(a),
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)
            wait(cooldown_time * u.ns, "resonator")

            align()  # global align

            play(qubit_operation, "qubit")
            align("qubit", "resonator")
            measure(
                "readout" * amp(a),
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
            )
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)
            wait(cooldown_time * u.ns, "resonator")
        assign(counter, counter + 1)

    with stream_processing():
        amps_st.save("iteration")
        # mean values
        I_g_st.buffer(n_runs).buffer(len(amps)).save("I_g")
        Q_g_st.buffer(n_runs).buffer(len(amps)).save("Q_g")
        I_e_st.buffer(n_runs).buffer(len(amps)).save("I_e")
        Q_e_st.buffer(n_runs).buffer(len(amps)).save("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, ro_amp_opt, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)

    job = qm.execute(ro_amp_opt)  # execute QUA program

    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Get progress counter to monitor runtime of the program
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration[0], len(amps), start_time=results.get_start_time())

    # Fetch the results at the end
    results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e"])
    I_g, Q_g, I_e, Q_e = results.fetch_all()

    # Process the data
    fidelities = []
    for i in range(len(amps)):
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            I_g[i], Q_g[i], I_e[i], Q_e[i], b_print=False, b_plot=False
        )
        fidelities.append(fidelity)

    # Plot the data
    plt.figure()
    plt.plot(amps, fidelities)
    plt.title("Readout amplitude optimization")
    plt.xlabel("Readout amp pre-factor [a.u.]")
    plt.ylabel("Fidelity [%]")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
