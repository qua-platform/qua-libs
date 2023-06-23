from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
from qm.simulate import LoopbackInterface
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
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

depletion_time = 1000
n_avg = 2000
flux_pts = 50

dcs = np.linspace(-0.49, 0.49, flux_pts)
dfs = np.arange(-5e6, 5e6, 0.05e6)

res_if_1 = machine.resonators[0].f_res - machine.local_oscillators.readout[0].freq
res_if_2 = machine.resonators[1].f_res - machine.local_oscillators.readout[0].freq

with program() as multi_res_spec_vs_flux:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    i = declare(int)
    df = declare(int)
    dc = declare(fixed)

    set_dc_offset("q1_z", "single", 0)
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency("rr0", df + res_if_1)
            update_frequency("rr1", df + res_if_2)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset("q0_z", "single", dc)
                # set_dc_offset("q1_z", "single", dc)
                wait(10)  # Wait for the flux to settle

                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], sequential=False)

                wait(depletion_time * u.ns, "rr0", "rr1")  # wait for the resonators to relax

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
    job = qmm.simulate(
        config,
        multi_res_spec_vs_flux,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(multi_res_spec_vs_flux)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)
        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)

        plt.subplot(121)
        plt.cla()
        plt.title(f"rr1 (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux (V)")
        plt.ylabel("freq (MHz)")
        plt.pcolor(dcs, res_if_1 / u.MHz + dfs / u.MHz, A1)
        plt.subplot(122)
        plt.cla()
        plt.title(f"rr2 (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux (V)")
        plt.ylabel("freq (MHz)")
        plt.pcolor(dcs, res_if_2 / u.MHz + dfs / u.MHz, A2)
        plt.tight_layout()
        plt.pause(0.1)

plt.show()
# Update machine with max frequency point for both resonator and qubit
# machine.qubits[0].z.max_frequency_point =
# machine.qubits[1].z.max_frequency_point =
# machine._save("quam_bootstrap_state.json")
