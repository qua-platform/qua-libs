from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from macros import multiplexed_readout
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

###################
# The QUA program #
###################
n_avg = 4000
dfs = np.arange(-2e6, 2e6, 0.02e6)
cooldown_time = 5 * max(qb1.T1, qb2.T1)

res_if_1 = rr1.f_opt - machine.local_oscillators.readout[0].freq
res_if_2 = rr2.f_opt - machine.local_oscillators.readout[0].freq

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(2)]
    Q_g = [declare(fixed) for _ in range(2)]
    I_e = [declare(fixed) for _ in range(2)]
    Q_e = [declare(fixed) for _ in range(2)]
    DI = declare(fixed)
    DQ = declare(fixed)
    D = [declare(fixed) for _ in range(2)]
    df = declare(int)
    D_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            update_frequency(rr1.name, df + res_if_1)
            update_frequency(rr2.name, df + res_if_2)

            # ground iq blobs for both qubits
            wait(cooldown_time * u.ns)
            align()
            multiplexed_readout(I_g, None, Q_g, None, resonators=active_qubits, weights="rotated_")

            # excited iq blobs for both qubits
            align()
            # Wait for thermalization again in case of measurement induced transitions
            wait(cooldown_time * u.ns)
            play("x180", qb1.name + "_xy")
            play("x180", qb2.name + "_xy")
            align()
            multiplexed_readout(I_e, None, Q_e, None, resonators=active_qubits, weights="rotated_")
            for i in range(len(active_qubits)):
                assign(DI, (I_e[i] - I_g[i]) * 100)
                assign(DQ, (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI * DI + DQ * DQ)
                save(D[i], D_st[i])

    with stream_processing():
        for i in range(len(active_qubits)):
            D_st[i].buffer(len(dfs)).average().save(f"D{i+1}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, ro_freq_opt, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    # open quantum machine
    qm = qmm.open_qm(config)
    # run job
    job = qm.execute(ro_freq_opt)
    # fetch data
    results = fetching_tool(job, ["D1", "D2"])
    D1, D2 = results.fetch_all()

    plt.subplot(211)
    plt.plot(dfs, D1)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{qb1.name} - f_opt = {int(rr1.f_opt / u.MHz)} MHz")
    plt.subplot(212)
    plt.plot(dfs, D2)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{qb2.name} - f_opt = {int(rr2.f_opt / u.MHz)} MHz")
    plt.tight_layout()
    print(f"{rr1.name}: Shift readout frequency by {dfs[np.argmax(D1)]} Hz")
    print(f"{rr2.name}: Shift readout frequency by {dfs[np.argmax(D2)]} Hz")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # machine.resonators[0].f_opt += dfs[np.argmax(D)]
    # machine.resonators[1].f_opt += dfs[np.argmax(D)]
    # machine._save("quam_bootstrap_state.json")
