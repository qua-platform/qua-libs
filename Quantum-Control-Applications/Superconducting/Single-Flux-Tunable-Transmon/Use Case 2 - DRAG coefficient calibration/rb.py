from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from scipy.optimize import curve_fit
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close


inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
# max_circuit_depth = int(3 * qubit_T1 / x180_len)
max_circuit_depth = 700
delta_depth = 1  # must be 1!!
num_of_sequences = 75
n_avg = 100
seed = 345324
cooldown_time = 5 * qubit_T1 // 4


def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                wait(x180_len // 4, "qubit")
            with case_(1):
                play("x180", "qubit")
            with case_(2):
                play("y180", "qubit")
            with case_(3):
                play("y180", "qubit")
                play("x180", "qubit")
            with case_(4):
                play("x90", "qubit")
                play("y90", "qubit")
            with case_(5):
                play("x90", "qubit")
                play("-y90", "qubit")
            with case_(6):
                play("-x90", "qubit")
                play("y90", "qubit")
            with case_(7):
                play("-x90", "qubit")
                play("-y90", "qubit")
            with case_(8):
                play("y90", "qubit")
                play("x90", "qubit")
            with case_(9):
                play("y90", "qubit")
                play("-x90", "qubit")
            with case_(10):
                play("-y90", "qubit")
                play("x90", "qubit")
            with case_(11):
                play("-y90", "qubit")
                play("-x90", "qubit")
            with case_(12):
                play("x90", "qubit")
            with case_(13):
                play("-x90", "qubit")
            with case_(14):
                play("y90", "qubit")
            with case_(15):
                play("-y90", "qubit")
            with case_(16):
                play("-x90", "qubit")
                play("y90", "qubit")
                play("x90", "qubit")
            with case_(17):
                play("-x90", "qubit")
                play("-y90", "qubit")
                play("x90", "qubit")
            with case_(18):
                play("x180", "qubit")
                play("y90", "qubit")
            with case_(19):
                play("x180", "qubit")
                play("-y90", "qubit")
            with case_(20):
                play("y180", "qubit")
                play("x90", "qubit")
            with case_(21):
                play("y180", "qubit")
                play("-x90", "qubit")
            with case_(22):
                play("x90", "qubit")
                play("y90", "qubit")
                play("x90", "qubit")
            with case_(23):
                play("-x90", "qubit")
                play("y90", "qubit")
                play("-x90", "qubit")


resonator_cooldown = 500

with program() as rb:
    depth = declare(int)
    saved_gate = declare(int)
    m = declare(int)
    n = declare(int)
    res = declare(bool)
    res_st = declare_stream()
    I = declare(fixed)
    Q = declare(fixed)
    state = declare(bool)
    state_st = declare_stream()
    n_st = declare_stream()
    Q_st = declare_stream()
    I_st = declare_stream()
    I_g = declare(fixed)

    with for_(m, 0, m < num_of_sequences, m + 1):
        sequence_list, inv_gate_list = generate_sequence()
        save(m, n_st)

        with for_(depth, 1, depth <= max_circuit_depth, depth + delta_depth):
            with for_(n, 0, n < n_avg, n + 1):
                # Replacing the last gate in the sequence with the sequence's inverse gate
                # The original gate is saved in 'saved_gate' and is being restored at the end
                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])

                # qubit cooldown based on state discrimination
                measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                # To prepare the ground state we used -0.0003 which is a more strict threshold (3 sigma)
                # to guarantee higher ground state fidelity
                with while_(I_g > -0.0003):
                    measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                align()
                wait(resonator_cooldown)

                align("resonator", "qubit")

                play_sequence(sequence_list, depth)
                align("qubit", "resonator")
                # Make sure you updated the ge_threshold
                state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)

                save(state, state_st)
                save(Q, Q_st)
                save(I, I_st)

                assign(sequence_list[depth], saved_gate)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(
            num_of_sequences, max_circuit_depth
        ).save("res")
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences, max_circuit_depth).save("Q")
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences, max_circuit_depth).save("I")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="192.168.88.10", port="80")
# Open quantum machine
qm = qmm.open_qm(config, close_other_machines=False)
# Execute QUA program
job = qm.execute(rb)
# Get results from QUA program
results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

while results.is_processing():
    # Fetch results
    state, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    value = 1 - np.average(state, axis=0)
    plt.cla()
    plt.plot(value)
    plt.xlabel("Number of cliffords")
    plt.ylabel("Sequence Fidelity")


def power_law(m, a, b, p):
    return a * (p**m) + b


x = np.linspace(1, max_circuit_depth, max_circuit_depth)
plt.xlabel("Number of cliffords")
plt.ylabel("Sequence Fidelity")

pars, cov = curve_fit(
    f=power_law,
    xdata=x,
    ydata=value,
    p0=[0.5, 0.5, 0.9],
    bounds=(-np.inf, np.inf),
    maxfev=2000,
)

plt.plot(x, power_law(x, *pars), linestyle="--", linewidth=2)

stdevs = np.sqrt(np.diag(cov))

print("#########################")
print("### Fitted Parameters ###")
print("#########################")
print(f"A = {pars[0]:.3} ({stdevs[0]:.1}), B = {pars[1]:.3} ({stdevs[1]:.1}), p = {pars[2]:.3} ({stdevs[2]:.1})")
print("Covariance Matrix")
print(cov)

one_minus_p = 1 - pars[2]
r_c = one_minus_p * (1 - 1 / 2**1)
r_g = r_c / 1.875  # 1.875 is the average number of gates in clifford operation
r_c_std = stdevs[2] * (1 - 1 / 2**1)
r_g_std = r_c_std / 1.875

print("#########################")
print("### Useful Parameters ###")
print("#########################")
print(
    f"Error rate: 1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1})\n"
    f"Clifford set infidelity: r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1})\n"
    f"Gate infidelity: r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
)

np.savez("rb_values", value)

qm.close()
