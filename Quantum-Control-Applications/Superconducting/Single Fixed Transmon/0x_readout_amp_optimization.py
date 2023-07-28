# %%
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.analysis import two_state_discriminator

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
qubit_operation = "x180"

n_avg = 100

cooldown_time = 20_000

amps = np.arange(0.1, 1.7, 0.1)

qubit_index = 0

with program() as IQ_blobs:
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
        with for_(n, 0, n < n_avg, n + 1):
            measure(
                "readout" * amp(a),
                machine.resonators[qubit_index].name,
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)
            wait(cooldown_time * u.ns, machine.resonators[qubit_index].name)

            align()  # global align

            play(qubit_operation, machine.qubits[qubit_index].name)
            align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)
            measure(
                "readout" * amp(a),
                machine.resonators[qubit_index].name,
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
            )
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)
            wait(cooldown_time * u.ns, machine.resonators[qubit_index].name)
        assign(counter, counter + 1)

    with stream_processing():
        amps_st.save("iteration")
        # mean values
        I_g_st.buffer(n_avg).buffer(len(amps)).save("I_g")
        Q_g_st.buffer(n_avg).buffer(len(amps)).save("Q_g")
        I_e_st.buffer(n_avg).buffer(len(amps)).save("I_e")
        Q_e_st.buffer(n_avg).buffer(len(amps)).save("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)  # execute QUA program

# Get results from QUA program
results = fetching_tool(
    job,
    data_list=["iteration"],
    mode="live",
)

while results.is_processing():
    # Fetch results
    iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration[0], len(amps), start_time=results.get_start_time())

results = fetching_tool(job, data_list=["I_g", "Q_g", "I_e", "Q_e"])

I_g, Q_g, I_e, Q_e = results.fetch_all()

fidelities = []
for i in range(len(amps)):
    angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
        I_g[i], Q_g[i], I_e[i], Q_e[i], b_print=False, b_plot=False
    )
    fidelities.append(fidelity)

plt.figure()
plt.plot(amps, fidelities)
plt.title("Readout optimization")
plt.xlabel("Readout amp [a.u.]")
plt.ylabel("Fidelity")
plt.show()
# %%
