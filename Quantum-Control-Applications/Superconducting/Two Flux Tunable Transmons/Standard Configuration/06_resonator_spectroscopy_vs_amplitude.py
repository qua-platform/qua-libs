from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
from qm.simulate import LoopbackInterface
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration

amps = np.arange(0.05, 1.99, 0.10)
dfs = np.arange(-1.0e6, + 1.0e6, 0.01e6)
n_avg = 2000

depletion_time = 1000

###################
# The QUA program #
###################
with program() as multi_res_spec_vs_amp:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            with for_(*from_array(a, amps)):
                # resonator 1
                wait(depletion_time * u.ns, "rr1")  # wait for the resonator to relax
                measure("readout" * amp(a), "rr1", None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[0]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[0]))
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

                # align("rr1", "rr2") # sequential to avoid overflow

                # resonator 2
                wait(depletion_time * u.ns, "rr2")  # wait for the resonator to relax
                measure("readout" * amp(a), "rr2", None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[1]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[1]))
                save(I[1], I_st[1])
                save(Q[1], Q_st[1])

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, multi_res_spec_vs_amp, SimulationConfig(11000,
    simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2) ], latency=250)))
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec_vs_amp)
    # Prepare the figures for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Data analysis
        s1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        s2 = u.demod2volts(I2 + 1j * Q2, readout_len)

        A1 = np.abs(s1)
        A2 = np.abs(s2)
        # Normalize data
        row_sums = A1.sum(axis=0)
        A1 = A1 / row_sums[np.newaxis, :]
        row_sums = A2.sum(axis=0)
        A2 = A2 / row_sums[np.newaxis, :]
        # Plot
        plt.subplot(121)
        plt.cla()
        plt.title(f"resonator 1 - f_cent: {(resonator_LO + resonator_IF_q1) / u.MHz})")
        plt.xlabel("amplitude pre-factor")
        plt.ylabel("detuning (MHz)")
        plt.pcolor(amps, dfs / u.MHz, A1)
        plt.subplot(122)
        plt.cla()
        plt.title(f"resonator 2 - f_cent: {(resonator_LO + resonator_IF_q2) / u.MHz})")
        plt.xlabel("amplitude pre-factor")
        plt.ylabel("detuning (MHz)")
        plt.pcolor(amps, dfs / u.MHz, A2)
        plt.tight_layout()

        plt.pause(0.1)