#%%
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.simulate import LoopbackInterface
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)

# Update the readout amplitude for the sweep
prev_amp1 = machine.resonators[active_qubits[0]].readout_pulse_amp
prev_amp2 = machine.resonators[active_qubits[1]].readout_pulse_amp
machine.resonators[active_qubits[0]].readout_pulse_amp = 0.01
machine.resonators[active_qubits[1]].readout_pulse_amp = 0.01

config = build_config(machine)

rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].qubit_name + "_z"
q2_z = machine.qubits[active_qubits[1]].qubit_name + "_z"

res_if_1 = rr1.f_res - machine.local_oscillators.readout[0].freq
res_if_2 = rr2.f_res - machine.local_oscillators.readout[0].freq

###################
# The QUA program #
###################
amps = np.arange(0.05, 1.99, 0.01)
dfs = np.arange(-10e6, +10e6, 0.1e6)
n_avg = 100
depletion_time = 1000

with program() as multi_res_spec_vs_amp:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    a = declare(fixed)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)
    
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency(rr1.resonator_name, df + res_if_1)
            update_frequency(rr2.resonator_name, df + res_if_2)

            with for_(*from_array(a, amps)):
                # resonator 1
                wait(depletion_time * u.ns, rr1.resonator_name)  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    rr1.resonator_name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[0]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[0]),
                )
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

                # align(rr1.resonator_name, rr2.resonator_name) # sequential to avoid overflow

                # resonator 2
                wait(depletion_time * u.ns, rr2.resonator_name)  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    rr2.resonator_name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[1]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[1]),
                )
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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        multi_res_spec_vs_amp,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
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
        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)
        # Normalize data
        row_sums = A1.sum(axis=0)
        A1 = A1 / row_sums[np.newaxis, :]
        row_sums = A2.sum(axis=0)
        A2 = A2 / row_sums[np.newaxis, :]
        # Plot
        plt.suptitle("Resonator spectroscopy vs amplitude")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{rr1.resonator_name} - f_cent: {int(rr1.f_res / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr1.readout_pulse_amp, dfs / u.MHz, A1)
        plt.axhline(0, color="k", linestyle='--')
        plt.axvline(prev_amp1, color="k", linestyle='--')
        plt.subplot(122)
        plt.cla()
        plt.title(f"{rr2.resonator_name} - f_cent: {int(rr2.f_res / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr2.readout_pulse_amp, dfs / u.MHz, A2)
        plt.axhline(0, color="k", linestyle='--')
        plt.axvline(prev_amp2, color="k", linestyle='--')
        plt.tight_layout()

        plt.pause(3)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # machine._save("current_state.json")

# %%
