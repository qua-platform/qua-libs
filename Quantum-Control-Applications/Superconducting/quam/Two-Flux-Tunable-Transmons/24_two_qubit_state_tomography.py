"""
        CZ CHEVRON - 1ns granularity
The goal of this protocol is to find the parameters of the CZ gate between two flux-tunable qubits.
The protocol consists in flux tuning one qubit to bring the |11> state on resonance with |20>.
The two qubits must start in their excited states so that, when |11> and |20> are on resonance, the state |11> will
start acquiring a global phase when varying the flux pulse duration.

By scanning the flux pulse amplitude and duration, the CZ chevron can be obtained and post-processed to extract the
CZ gate parameters corresponding to a single oscillation period such that |11> pick up an overall phase of pi (flux
pulse amplitude and interation time).

This version sweeps the flux pulse duration using the baking tool, which means that the flux pulse can be scanned with
a 1ns resolution, but must be shorter than ~260ns. If you want to measure longer flux pulse, you can either reduce the
resolution (do 2ns steps instead of 1ns) or use the 4ns version (CZ.py).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having found the qubits maximum frequency point (qubit_spectroscopy_vs_flux).
    - Having calibrated qubit gates (x180) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - (Optional) having corrected the flux line distortions by running the Cryoscope protocol and updating the filter taps in the state.

Next steps before going to the next node:
    - Update the CZ gate parameters in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import fitting
from qualang_tools.loops import from_array
from qualang_tools.bakery import baking
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM, Transmon
from macros import qua_declaration, multiplexed_readout


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]


####################
# Helper functions #
####################
##################
# State and QuAM #
##################

experiment = "arb_two_qst"
debug = True
simulate = False
fit_data = False
save_raw = False

flux_lines = [0, 1]

digital = [1, 2, 3, 9]
machine = QuAM("latest_quam.json")
# gate_shape = "drag_cosine"
gate_shape = "square"

qubit_list = [0, 1, 2]  # you can shuffle the order at which you perform the experiment

# machine.qubits[q].driving.drag_cosine.length = 300e-9
# machine.qubits[2].f_01 = 4.164803e9 #4.16480e9
# machine.get_qubit_gate(2, gate_shape).angle2volt.deg90 = 0.041 #0.041

config = machine.build_config(digital, qubit_list, flux_lines, gate_shape)

qubit_list = [1, 2]

q1_gate = q2_gate = "y90"

###################
# The QUA program #
###################

# Defining two dictionaries for the operators and the expectation values
I = [[1, 0], [0, 1]]
sigma_x = [[0, 1], [1, 0]]
sigma_y = [[0, -1j], [1j, 0]]
sigma_z = [[1, 0], [0, -1]]
ops = np.zeros((2, 2, 4), dtype=complex)

ops[:, :, 0] = I
ops[:, :, 1] = sigma_x
ops[:, :, 2] = sigma_y
ops[:, :, 3] = sigma_z

name_op = ["I", "X", "Y", "Z"]
name_res = ["i", "x", "y", "z"]
ops_dict = {}
res_dict = {}

for i in range(len(name_op)):
    for j in range(len(name_op)):
        temp_res = {name_res[i] + name_res[j]: []}
        temp_op = {name_op[i] + name_op[j]: np.kron(ops[:, :, i], ops[:, :, j])}

        res_dict.update(temp_res)
        ops_dict.update(temp_op)


def two_qb_QST(qb1: Transmon, qb2: Transmon, operation: str):
    len1 = qb1.xy.operations[operation].length
    len2 = qb2.xy.operations[operation].length
    with switch_(c):
        with case_(0):  # XX
            qb1.xy.play("-y90")
            qb2.xy.play("-y90")
        with case_(1):  # XY
            qb1.xy.play("-y90")
            qb2.xy.play("x90")
        with case_(2):  # XZ
            qb1.xy.play("-y90")
            qb2.xy.wait(int(len2 // 4))
        with case_(3):  # YX
            qb1.xy.play("x90")
            qb2.xy.play("-y90")
        with case_(4):  # YY
            qb1.xy.play("x90")
            qb2.xy.play("x90")
        with case_(5):  # YZ
            qb1.xy.play("x90")
            qb2.xy.wait(int(len2 // 4))
        with case_(6):  # ZX
            qb1.xy.wait(int(len1 // 4))
            qb2.xy.play("-y90")
        with case_(7):  # ZY
            qb1.xy.wait(int(len1 // 4))
            qb2.xy.play("x90")
            wait(int(len1 // 4), qb1)
        with case_(8):  # ZZ
            qb1.xy.wait(int(len1 // 4))
            qb2.xy.wait(int(len2 // 4))


def prob_2q(data_q1, data_q2):
    # Input: qubit_data["state_all_shot_Q1"] with dims (n_avg, 9 msrmt)
    # Return P00, P01, P10, P11 (the joint probabilities)

    P = np.zeros((1, 4, 9))

    for j in range(np.shape(data_q1)[1]):

        N_00 = N_01 = N_10 = N_11 = 0

        for i in range(np.shape(data_q1)[0]):

            if data_q1[i, j] == 0 and data_q2[i, j] == 0:
                N_00 += 1

            elif data_q1[i, j] == 0 and data_q2[i, j] == 1:
                N_01 += 1

            elif data_q1[i, j] == 1 and data_q2[i, j] == 0:
                N_10 += 1

            elif data_q1[i, j] == 1 and data_q2[i, j] == 1:
                N_11 += 1

        N_tot = N_00 + N_01 + N_10 + N_11

        P_00 = N_00 / N_tot
        P_01 = N_01 / N_tot
        P_10 = N_10 / N_tot
        P_11 = N_11 / N_tot

        P[:, :, j] = np.array([P_00, P_01, P_10, P_11])

    return P


def prob_2q_decimal(data_joint):
    # Input: Joint probability of (n_avg, measurement#)
    # Return P00, P01, P10, P11 (the joint probabilities)

    P = np.zeros((1, 4, 9))

    for j in range(np.shape(data_joint)[1]):

        N_00 = N_01 = N_10 = N_11 = N_tot = 0

        for i in range(np.shape(data_joint)[0]):

            if data_joint[i, j] == 0:
                N_00 += 1

            elif data_joint[i, j] == 1:
                N_01 += 1

            elif data_joint[i, j] == 2:
                N_10 += 1

            elif data_joint[i, j] == 3:
                N_11 += 1

        N_tot = N_00 + N_01 + N_10 + N_11

        P_00 = N_00 / N_tot
        P_01 = N_01 / N_tot
        P_10 = N_10 / N_tot
        P_11 = N_11 / N_tot

        P[:, :, j] = np.array([P_00, P_01, P_10, P_11])

    return P


def T_mat(t):
    # Defining the lower triangular matrix for MLE given vector t
    T_mat = np.zeros((4, 4), dtype=complex)
    T_mat = np.matrix(
        [
            [t[0], 0, 0, 0],
            [t[4] + (1j * t[5]), t[1], 0, 0],
            [t[6] + (1j * t[7]), t[8] + (1j * t[9]), t[2], 0],
            [t[10] + (1j * t[11]), t[12] + (1j * t[13]), t[14] + (1j * t[15]), t[3]],
        ]
    )

    return T_mat


def loss_func(x, res_dict, ops_dict=ops_dict):
    # loss function for each measurement
    # Inputs: M:: measurement matrix p:: measured outcome t:: 16-element complex vector
    # Output: MLE minimization function

    name = ["I", "X", "Y", "Z"]

    rho = T_dag = np.zeros((4, 4), dtype=complex)
    loss = 0.0 + 0.0 * 1j

    T_dag = T_mat(x).getH()
    rho = np.matmul(T_dag, T_mat(x))  # /(np.trace(np.matmul(T_dag, T_mat(x))))

    for i in range(len(name)):
        for j in range(len(name)):
            if name[i] + name[j] == "II":
                continue
            else:
                loss = (
                    loss
                    + (
                        np.trace(np.matmul(ops_dict[name[i] + name[j]], rho))
                        - res_dict[name[i].lower() + name[j].lower()]
                    )
                    ** 2
                )  # /(2*np.trace(np.matmul(ops_dict[name[i]+name[j]], rho)))

    return [loss.real, loss.imag]


def rho_2q(t):
    # output rho with the optimal t vector
    rho = T_dag = np.zeros((4, 4), dtype=complex)

    T_dag = T_mat(t).getH()
    rho = np.matmul(T_dag, T_mat(t)) / (np.trace(np.matmul(T_dag, T_mat(t))))

    return rho


def fidelity(rho, vect):
    # function to calculate the fidelity of the constructed density matrix
    # inputs:: rho: constructed density matrix vect: column of the desired prepared state

    sigma = np.zeros(np.shape(rho), dtype=complex)
    sigma = np.matmul(vect, vect.getH()) / np.trace(np.matmul(vect.getH(), vect))

    sqr_rho = sqrtm(rho)
    sqr_rho_sig = np.matmul(sqr_rho, sigma)
    sqr_rho_sig_sqr_rho = np.matmul(sqr_rho_sig, sqr_rho)

    F = (np.trace(sqrtm(sqr_rho_sig_sqr_rho))) ** 2

    print(f"Fidelity = {F.real * 100:.2f}%")

    return F


def concurrence(rho):
    sysy = ops_dict["YY"]

    rho_tilde = np.matmul(np.matmul(sysy, rho.conjugate()), sysy)

    rho_rho_tilde = np.matmul(rho, rho_tilde)

    evals, _ = np.linalg.eig(rho_rho_tilde)

    # abs to avoid problems with sqrt for very small negative numbers
    evals = abs(np.sort(evals.real))

    lsum = np.sqrt(evals[3]) - np.sqrt(evals[2]) - np.sqrt(evals[1]) - np.sqrt(evals[0])

    print(f"Concurrence = {max(0, lsum):.2f}")

    return max(0, lsum)


n_avg = 10e3

with program() as arb_two_qst:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    c = declare(int)  # variable for switch case
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]
    iters = declare(int)
    state_decimal = declare(int)
    aux = declare(int)
    state_decimal_st = declare_stream()

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_flux_relation):
            if q == z:
                set_dc_offset(
                    machine.qubits[q].name + "_flux",
                    "single",
                    machine.get_flux_bias_point(j, "working_point").value,
                )

    with for_(iters, 0, iters < n_avg, iters + 1):
        # with for_(*from_array(t, lengths)):

        with for_(c, 0, c < 9, c + 1):
            assign(aux, 0)
            assign(state_decimal, 0)

            play(q1_gate, machine.qubits[qubit_list[0]].name)
            play(q2_gate, machine.qubits[qubit_list[1]].name)
            # wait(4, machine.qubits[qubit_list[0]].name)
            # wait(20, machine.qubits[qubit_list[0]].name)
            # wait(20, machine.qubits[qubit_list[1]].name)
            # align()  # global align

            # wait(4, machine.qubits[qubit_list[0]].name)
            # wait(4, machine.qubits[qubit_list[1]].name)

            two_qb_QST(
                machine.qubits[qubit_list[0]].name,
                machine.qubits[qubit_list[1]].name,
                machine.qubits[qubit_list[0]].driving.drag_cosine.length,
            )

            align(machine.qubits[qubit_list[0]].name, machine.readout_resonators[qubit_list[0]].name)
            align(machine.qubits[qubit_list[1]].name, machine.readout_resonators[qubit_list[1]].name)

            # align()

            for i, q in enumerate(qubit_list):
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    demod.full("rotated_cos", I[i], "out1"),
                )

                assign(state[i], I[i] > machine.readout_resonators[q].ge_threshold)
                save(state[i], state_st[i])

                save(iters, n_st[i])

            for i, q in enumerate(qubit_list):
                assign(aux, Cast.to_int(state[i]) * 2**i)
                assign(state_decimal, state_decimal + aux)

            save(state_decimal, state_decimal_st)
            wait_cooldown_time_fivet1(q, machine, simulate)

    align()

    with stream_processing():

        state_decimal_st.buffer(9).buffer(n_avg).save("state_decimal")

        for i, q in enumerate(qubit_list):
            # state_st[i].boolean_to_int().buffer(2).buffer(3).buffer(len(lengths)).average().save(f"states{q}")\
            # state_st[i].boolean_to_int().buffer(9).average().save(f"states{q}")
            # state_st[i].boolean_to_int().buffer(3).buffer(len(lengths)).save(f"states_no_avg{q}")
            state_st[i].boolean_to_int().buffer(9).buffer(n_avg).save(f"state_all_shots{q}")
            # state_st[i].boolean_to_int().buffer(len(lengths)).average().save(f"state{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
with JobQueue("Daria", message="2QST"):
    if simulate:
        simulation_config = SimulationConfig(duration=20000)
        job = qmm.simulate(config, arb_two_qst, simulation_config)
        job.get_simulated_samples().con1.plot()

    else:
        qm = qmm.open_qm(config)
        job = qm.execute(arb_two_qst)

        # Initialize dataset
        qubit_data = {}
        figures = []
        # Create the fitting object
        Fit = fitting.Fit()

        print(f"Arb state 2QST on Q{qubit_list[0]} and Q{qubit_list[1]}")
        qubit_data["iteration"] = 0

        # Get results from QUA program
        my_results = fetching_tool(
            job,
            [f"state_all_shots{qubit_list[0]}", f"state_all_shots{qubit_list[1]}", "state_decimal", f"iteration{q}"],
            mode="live",
        )

        while my_results.is_processing() and qubit_data["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data["state_all_shot_Q1"] = data[0]
            qubit_data["state_all_shot_Q2"] = data[1]
            qubit_data["state_decimal"] = data[2]
            qubit_data["iteration"] = data[-1]

            # Progress bar
            progress_counter(qubit_data["iteration"], n_avg, start_time=my_results.start_time)

        # initiate res_dict[*]=0 for each msrmt round
        for k, v in res_dict.items():
            res_dict[k] = []

        P = prob_2q(qubit_data["state_all_shot_Q1"], qubit_data["state_all_shot_Q2"])
        # P = prob_2q_decimal(qubit_data["state_decimal"])

        expec_name_1 = []
        expec_name_1.append(["i", "x"])
        expec_name_1.append(["i", "x"])
        expec_name_1.append(["i", "x"])
        expec_name_1.append(["i", "y"])
        expec_name_1.append(["i", "y"])
        expec_name_1.append(["i", "y"])
        expec_name_1.append(["i", "z"])
        expec_name_1.append(["i", "z"])
        expec_name_1.append(["i", "z"])

        expec_name_2 = []
        expec_name_2.append(["i", "x"])
        expec_name_2.append(["i", "y"])
        expec_name_2.append(["i", "z"])
        expec_name_2.append(["i", "x"])
        expec_name_2.append(["i", "y"])
        expec_name_2.append(["i", "z"])
        expec_name_2.append(["i", "x"])
        expec_name_2.append(["i", "y"])
        expec_name_2.append(["i", "z"])

        IZ = ["I", "Z"]

        for k in range(9):
            for i in range(len(IZ)):
                for j in range(len(IZ)):
                    res_dict[expec_name_1[k][i] + expec_name_2[k][j]].append(
                        np.sum(np.matmul(ops_dict[IZ[i] + IZ[j]], np.matrix(P[:, :, k]).transpose()))
                    )
        # print(res_dict)
        for k, v in res_dict.items():
            res_dict[k] = np.mean(res_dict[k])

        x0 = np.ones((16))
        x0[:] = 0.25

        # result = minimize(loss_func, x0=x0, args=(res_dict, ops_dict), method="BFGS", tol=1e-40, options={'maxiter':1000000, 'gtol':1e-30})
        result = least_squares(loss_func, x0=x0, args=(res_dict, ops_dict), xtol=1e-15, gtol=1e-15, method="trf")
        # print(f'Cost = {result.cost}')
        # print(f'Optimality = {result.optimality}')
        # result.x

        rho = rho_2q(result.x) / np.trace(rho_2q(result.x))
        # print(rho)

        if q1_gate == "y90" and q2_gate == "y90":
            fid = fidelity(rho, np.matrix([[1, 1, 1, 1]]).transpose())  # +x
        elif q1_gate == "y-90" and q2_gate == "y-90":
            fid = fidelity(rho, np.matrix([[1, -1, -1, 1]]).transpose())  # -x
        elif q1_gate == "x-90" and q2_gate == "x-90":
            fid = fidelity(rho, np.matrix([[1, 1j, 1j, -1]]).transpose())  # +y
        elif q1_gate == "x90" and q2_gate == "x90":
            fid = fidelity(rho, np.matrix([[1, -1j, -1j, -1]]).transpose())  # -y
        else:
            print("Manually calculate the fidelity!!")

        conc = concurrence(rho)

        if save_raw:
            data_dir = r"C:\Data\2023\20230108_Muninnv200\OPX\2QST"

            file_name = (
                "_"
                + q1_gate
                + q2_gate
                + f"_2QST_SquarePulse_[qubit_data,res_dict]_RawData_Q{qubit_list}_ROdur_{machine.readout_lines[0].length}_avg_{n_avg}_Fid_{np.real(fid):.2f}_"
            )
            os.makedirs(data_dir + "\\", exist_ok=True)
            np.save(data_dir + "\\" + "/" + datetime.now().strftime("%H%M%S") + file_name, [qubit_data, res_dict])

            file_name = (
                "_"
                + q1_gate
                + q2_gate
                + f"_2QST_[rho]_RawData_Q{qubit_list}_ROdur_{machine.readout_lines[0].length}_avg_{n_avg}_Fid_{np.real(fid):.2f}_"
            )
            np.save(data_dir + "\\" + "/" + datetime.now().strftime("%H%M%S") + file_name, rho)
        # Visualize the density matrix

        xlabels = ["|00>", "|01>", "|10>", "|11>"]

        fig, ax1 = matrix_histogram(
            np.array(rho.real),
            xlabels=xlabels,
            ylabels=xlabels,
            limits=[-0.3, 0.3],
            options={"cmap": "winter_r", "bars_alpha": 0.5, "figsize": (7, 5), "zticks": [-0.2, 0, 0.2]},
        )
        ax1.set_title("$\\mathcal{Re}~(\\rho$)", fontsize=17)
        ax1.view_init(azim=-38, elev=20)

        fig, ax2 = matrix_histogram(
            np.array(rho.imag),
            xlabels=xlabels,
            ylabels=xlabels,
            limits=[-0.3, 0.3],
            options={"cmap": "winter_r", "bars_alpha": 0.5, "figsize": (7, 5), "zticks": [-0.2, 0, 0.2]},
        )
        ax2.set_title("$\\mathcal{Im}~(\\rho$)", fontsize=17)
        ax2.view_init(azim=-38, elev=20)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    # qb.z.cz.length =
    # qb.z.cz.level =
# machine.save("quam")
