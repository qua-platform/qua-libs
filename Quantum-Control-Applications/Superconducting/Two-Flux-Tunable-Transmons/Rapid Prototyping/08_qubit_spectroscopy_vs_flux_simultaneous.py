from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from quam import QuAM
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
cooldown_time = 1 * u.us
t = 1 * u.us  # Qubit drive
n_avg = 400

dfs = np.arange(-100e6, 100e6, 0.1e6)
dcs = np.arange(-0.49, 0.49, 0.1)

qb_if_1 = machine.qubits[0].xy.f_01 - machine.local_oscillators.qubits[0].freq
qb_if_2 = machine.qubits[1].xy.f_01 - machine.local_oscillators.qubits[0].freq

with program() as multi_qubit_spec_vs_flux:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    dc = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency("q0_xy", df + qb_if_1)
            update_frequency("q1_xy", df + qb_if_2)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset("q0_z", "single", dc + 0.05)
                set_dc_offset("q1_z", "single", dc)
                wait(10)  # Wait for the flux to settle

                # Saturate qubit
                play("cw" * amp(1), "q0_xy", duration=t * u.ns)
                play("cw" * amp(1), "q1_xy", duration=t * u.ns)

                align()

                # readout
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], amplitude=0.9)
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(multi_qubit_spec_vs_flux)

    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        plt.subplot(231)
        plt.cla()
        plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.abs(s1))
        plt.xlabel("flux (V)")
        plt.ylabel("q1_ifreq (MHz)")
        plt.title(f"q1 (f_res1: {machine.qubits[0].xy.f_01 / u.MHz} MHz)")
        plt.subplot(235)
        plt.cla()
        plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.unwrap(np.angle(s1)))
        plt.xlabel("flux (V)")
        plt.ylabel("q1_ifreq (MHz)")
        plt.subplot(232)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.abs(s2))
        plt.title(f"q2 (f_res2: {machine.qubits[1].xy.f_01 / u.MHz} MHz)")
        plt.ylabel("q2_ifreq (MHz)")
        plt.xlabel("flux (V)")
        plt.subplot(234)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.unwrap(np.angle(s2)))
        plt.xlabel("flux (V)")
        plt.ylabel("q2_ifreq (MHz)")
        # Add both to compare:
        plt.subplot(233)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.abs(s1) + np.abs(s2))
        plt.title(f"Sum of the two resonators")
        plt.ylabel("q2_ifreq (MHz)")
        plt.xlabel("flux (V)")
        plt.subplot(236)
        plt.cla()
        plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.unwrap(np.angle(s1)) + np.unwrap(np.angle(s2)))
        plt.xlabel("flux (V)")
        plt.ylabel("q2_ifreq (MHz)")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()