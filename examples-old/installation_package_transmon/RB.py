from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import numpy as np
from matplotlib import pyplot as plt
from qm.qua import *
from scipy.optimize import curve_fit
from qm import LoopbackInterface


################################
# Open quantum machine manager #
################################

qmm = QuantumMachinesManager()

########################
# Open quantum machine #
########################

qm = qmm.open_qm(config)

##########
# Macros #
##########

cayley_table = np.int_(np.genfromtxt("c1_cayley_table.csv", delimiter=","))[1:, 1:]
inv_gates = [int(np.where(cayley_table[i, :] == 0)[0][0]) for i in range(24)]
max_circuit_depth = 180  # maximum number of Cliffords
delta_depth = 1  # must be 1!! - step in number of Cliffords
num_of_sequences = 50  # average over RB sequences
seed = 345324


def generate_sequence():

    cayley = declare(int, value=cayley_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        # assign(step, rand.rand_int(24))
        assign(step, np.random.randint(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


# def play_sequence(sequence_list, depth):
#     i = declare(int)
#     with for_(i, 0, i <= depth, i+1):
#         with switch_(sequence_list[i], unsafe=True):
#             for k in range(24):
#                 with case_(k):
#                     play(f'gate_{k}', 'qubit')


# pulse_len = 10


def play_sequence(sequence_list, depth):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):

        with switch_(sequence_list[i], unsafe=True):
            """
            for j in range(24):
                with case_(j):
                    wait(pulse_len, 'qubit')
            """
            with case_(0):
                wait(pi_len, "qubit")
            with case_(1):
                play("X", "qubit")
            with case_(2):
                play("Y", "qubit")
            with case_(3):
                play("Y", "qubit")
                play("X", "qubit")
            with case_(4):
                play("X/2", "qubit")
                play("Y/2", "qubit")
            with case_(5):
                play("X/2", "qubit")
                play("-Y/2", "qubit")
            with case_(6):
                play("-X/2", "qubit")
                play("Y/2", "qubit")
            with case_(7):
                play("-X/2", "qubit")
                play("-Y/2", "qubit")
            with case_(8):
                play("Y/2", "qubit")
                play("X/2", "qubit")
            with case_(9):
                play("Y/2", "qubit")
                play("-X/2", "qubit")
            with case_(10):
                play("-Y/2", "qubit")
                play("X/2", "qubit")
            with case_(11):
                play("-Y/2", "qubit")
                play("-X/2", "qubit")
            with case_(12):
                play("X/2", "qubit")
            with case_(13):
                play("-X/2", "qubit")
            with case_(14):
                play("Y/2", "qubit")
            with case_(15):
                play("-Y/2", "qubit")
            with case_(16):
                play("-X/2", "qubit")
                play("Y/2", "qubit")
                play("X/2", "qubit")
            with case_(17):
                play("-X/2", "qubit")
                play("-Y/2", "qubit")
                play("X/2", "qubit")
            with case_(18):
                play("X", "qubit")
                play("Y/2", "qubit")
            with case_(19):
                play("X", "qubit")
                play("-Y/2", "qubit")
            with case_(20):
                play("Y", "qubit")
                play("X/2", "qubit")
            with case_(21):
                play("Y", "qubit")
                play("-X/2", "qubit")
            with case_(22):
                play("X/2", "qubit")
                play("Y/2", "qubit")
                play("X/2", "qubit")
            with case_(23):
                play("-X/2", "qubit")
                play("Y/2", "qubit")
                play("-X/2", "qubit")
            ###


###################
# The QUA program #
###################


avgs = 1

cooldown_time = 50000 // 4  # decay time for qubit

with program() as rb:

    # Declare QUA variables
    ###################
    depth = declare(int)  # variable for number of Clifford gates
    saved_gate = declare(int)
    m = declare(int)  # loop over repeats of the whole Clifford sequence
    m_st = declare_stream()  # to save iteration
    n = declare(int)  # loop over one Clifford gate
    res_st = declare_stream()  # save sigma_z measurement
    I = declare(fixed)  # demodulated and integrated signal
    Q = declare(fixed)  # demodulated and integrated signal
    I_st = declare_stream()  # stream for I
    Q_st = declare_stream()  # stream for Q

    # Pulse sequence
    ################
    with for_(m, 0, m < num_of_sequences, m + 1):
        sequence_list, inv_gate_list = generate_sequence()
        save(m, m_st)

        with for_(depth, 1, depth <= max_circuit_depth, depth + delta_depth):

            with for_(n, 0, n < avgs, n + 1):

                assign(saved_gate, sequence_list[depth])
                assign(sequence_list[depth], inv_gate_list[depth - 1])

                wait(cooldown_time, "qubit")

                align("resonator", "qubit")

                play_sequence(sequence_list, depth)

                align("qubit", "resonator")

                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                save(Q, res_st)

                assign(sequence_list[depth], saved_gate)

    # Stream processing
    ###################
    with stream_processing():
        res_st.buffer(avgs).map(FUNCTIONS.average()).buffer(num_of_sequences, max_circuit_depth).save("res")
        m_st.save("iteration")

"""
simulation_config = SimulationConfig(
    duration=10000
)
job = qmm.simulate(config, rb, simulation_config, flags=['between-block-variable-count'])
job.result_handles.wait_for_all_values()

plt.figure()
job.get_simulated_samples().con1.plot()
plt.show()
"""

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    # simulation properties
    simulate_config = SimulationConfig(
        duration=5000,
        simulation_interface=LoopbackInterface(([("con1", 1, "con1", 1)])),
    )
    job = qmm.simulate(config, rb, simulate_config)  # do simulation with qmm
    job.get_simulated_samples().con1.plot()  # visualize played pulses

else:

    job = qm.execute(rb)  # execute QUA program

    res_handles = job.result_handles  # get access to handles
    resvalue_handle = res_handles.get("res")
    resvalue_handle.wait_for_values(1)
    iteration_handle = res_handles.get("iteration")
    iteration_handle.wait_for_values(1)

    resvalue_ground = 1

    plt.figure()

    while res_handles.is_processing():
        try:
            resvalue = resvalue_handle.fetch_all()
            iteration = iteration_handle.fetch_all() + 1
            print(iteration)
            value = 1 - (np.average(resvalue, axis=0) / resvalue_ground)  # need to normalize by ground state value
            plt.plot(value)
            plt.xlabel("Number of gates")
            plt.ylabel("Fidelity")
            plt.pause(0.1)
            plt.clf()

        except Exception as e:
            pass

    resvalue = res_handles.res.fetch_all()

    # seq = job.result_handles.get('seq').fetch_all()['value']
    # i_seq = job.result_handles.get('i_seq').fetch_all()['value']

    # for a, b in zip(i_seq, seq):
    #     print('seq = ', b, 'i_seq', a)

    value = 1 - (np.average(resvalue, axis=0) / resvalue_ground)  # need to normalize by ground state value
    plt.plot(value)
    plt.xlabel("Number of gates")
    plt.ylabel("Fidelity")
    # analysis: we should average on seq and remain with depth on the x-axis.

    # qmm = QuantumMachinesManager()
    # qmm.close_all_quantum_machines()
    # qm = qmm.open_qm(config)
    # job = qm.simulate(rb, SimulationConfig(90000))
    #
    # plt.figure()
    # job.get_simulated_samples().con1.plot()
    # plt.show()

    def power_law(m, a, b, p):
        return a * (p**m) + b

    x = np.linspace(1, max_circuit_depth, delta_depth)
    # Generate dummy dataset
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
    r_g = r_c / 1.875
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / 1.875

    print("#########################")
    print("### Useful Parameters ###")
    print("#########################")
    print(
        f"1-p = {np.format_float_scientific(one_minus_p, precision=2)} ({stdevs[2]:.1}), "
        f"r_c = {np.format_float_scientific(r_c, precision=2)} ({r_c_std:.1}), "
        f"r_g = {np.format_float_scientific(r_g, precision=2)}  ({r_g_std:.1})"
    )

    # np.savez('rb_values', value)
