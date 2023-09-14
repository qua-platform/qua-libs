
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
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json")

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq
qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

config = build_config(machine)

###################
# The QUA program #
###################
cooldown_time = 5 * max(qb1.T1, qb2.T1)
t = 10 * u.us  # Qubit drive
n_avg = 100

dfs = np.arange(-50e6, 100e6, 0.1e6)
dcs = np.linspace(-0.05, 0.05, 40)  # flux



# qb_if_1 = 340e6
# qb_if_2 = 0

with program() as multi_qubit_spec_vs_flux:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)
    f_q1 = declare(int)
    f_q2 = declare(int)
    dc = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset(qb1.name + "_z", "single", dc)
                set_dc_offset(qb2.name + "_z", "single", dc)
                wait(100)  # Wait for the flux to settle

                # Saturate qubit
                # play("cw" * amp(1), qb1.name + "_xy", duration=t * u.ns)
                # play("cw" * amp(1), qb2.name + "_xy", duration=t * u.ns)
                play("x180", qb1.name + "_xy")
                play("x180", qb2.name + "_xy")
                # readout
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, amplitude=0.9)
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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # qm = qmm.open_qm(config)
    # job = qm.execute(multi_qubit_spec_vs_flux)
    from qm.QmJob import QmJob
    from qualang_tools.multi_user import qm_session

    with qm_session(qmm, config, timeout=100) as qm:
        job: QmJob = qm.execute(multi_qubit_spec_vs_flux)

        fig = plt.figure()
        interrupt_on_close(fig, job)
        results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
        while results.is_processing():
            n, I1, Q1, I2, Q2 = results.fetch_all()
            progress_counter(n, n_avg, start_time=results.start_time)

            s1 = u.demod2volts(I1 + 1j * Q1, rr1.readout_pulse_length)
            s2 = u.demod2volts(I2 + 1j * Q2, rr2.readout_pulse_length)

            plt.suptitle("Qubit spectroscopy vs flux")
            plt.subplot(221)
            plt.cla()
            plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.abs(s1))
            plt.plot(qb1.z.max_frequency_point, qb_if_1 / u.MHz, "r*")
            plt.xlabel("Flux [V]")
            plt.ylabel(f"{qb1.name} IF [MHz]")
            plt.title(f"{qb1.name} (f_01: {int(qb1.xy.f_01 / u.MHz)} MHz)")
            plt.subplot(223)
            plt.cla()
            plt.pcolor(dcs, (qb_if_1 + dfs) / u.MHz, np.unwrap(np.angle(s1)))
            plt.plot(qb1.z.max_frequency_point, qb_if_1 / u.MHz, "r*")
            plt.xlabel("Flux [V]")
            plt.ylabel(f"{qb1.name} IF [MHz]")
            plt.subplot(222)
            plt.cla()
            plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.abs(s2))
            plt.plot(qb2.z.max_frequency_point, qb_if_2 / u.MHz, "r*")
            plt.title(f"{qb2.name} (f_01: {int(qb2.xy.f_01 / u.MHz)} MHz)")
            plt.ylabel(f"{qb2.name} IF [MHz]")
            plt.xlabel("flux [V]")
            plt.subplot(224)
            plt.cla()
            plt.pcolor(dcs, (qb_if_2 + dfs) / u.MHz, np.unwrap(np.angle(s2)))
            plt.plot(qb2.z.max_frequency_point, qb_if_2 / u.MHz, "r*")
            plt.xlabel("Flux [V]")
            plt.ylabel(f"{qb2.name} IF [MHz]")
            plt.tight_layout()
            plt.pause(1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    # qm.close()
# Set the relevant flux points
# qb1.z. =
# qb2.z. =
# machine._save("quam_bootstrap_state.json")


