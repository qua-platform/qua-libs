from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from qm.simulate import LoopbackInterface
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)

# machine.resonators[2].readout_pulse_amp = 0.001
# machine.resonators[4].readout_pulse_amp = 0.002
config = build_config(machine)

###################
# The QUA program #
###################

depletion_time = 1000
n_avg = 200
flux_pts = 50

dcs = np.linspace(-0.49, 0.49, flux_pts)
dfs = np.arange(-50e6, 5e6, 0.1e6)

rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"

res_if_1 = rr1.f_res - machine.local_oscillators.readout[0].freq
res_if_2 = rr2.f_res - machine.local_oscillators.readout[0].freq

# res_if_1 = 170e6
# res_if_2 = 225e6
with program() as multi_res_spec_vs_flux:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    i = declare(int)
    df = declare(int)
    dc = declare(fixed)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency(rr1.name, df + res_if_1)
            update_frequency(rr2.name, df + res_if_2)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset(q1_z, "single", dc)
                set_dc_offset(q2_z, "single", dc)
                wait(100)  # Wait for the flux to settle

                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, sequential=False)

                wait(depletion_time * u.ns, rr1.name, rr2.name)  # wait for the resonators to relax

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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

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

        plt.suptitle("Resonator specrtoscopy vs flux")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{rr1.name} (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{rr1.name} IF [MHz]")
        plt.pcolor(dcs, res_if_1 / u.MHz + dfs / u.MHz, A1)
        plt.plot(machine.qubits[active_qubits[0]].z.max_frequency_point, res_if_1/ u.MHz, "r*" )
        plt.subplot(122)
        plt.cla()
        plt.title(f"{rr2.name} (LO: {machine.local_oscillators.readout[0].freq / u.MHz} MHz)")
        plt.xlabel("flux [V]")
        plt.ylabel(f"{rr2.name} IF [MHz]")
        plt.pcolor(dcs, res_if_2 / u.MHz + dfs / u.MHz, A2)
        plt.plot(machine.qubits[active_qubits[1]].z.max_frequency_point, res_if_2/ u.MHz, "r*" )
        plt.tight_layout()
        plt.pause(1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
plt.show()
# Update machine with max frequency point for both resonator and qubit
# qb1.z.max_frequency_point =
# qb2.z.max_frequency_point =
# machine._save("current_state.json")
