from configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np
from scipy.optimize import minimize

Navg = 40000
thresh = 0.001  # threshold on the Q value to determine the state e/g
res_time = 10000


def reset_qubit():
    qv = declare(fixed)
    align("res", "qubit")
    wait(res_time, "res")  # a little bit of buffer
    align("res", "qubit")
    measure("meas_op_res", "res", None, demod.full("integ_w_s", qv))

    with while_(qv > -0.005):  # tight threshold to get high fidelity of reset
        play("pi_op_qubit", "qubit")
        measure("meas_op_res", "res", None, demod.full("integ_w_s", qv))

        # rotation name to operation conversion


rot_dict = {
    "X": (0.0, 1.0),
    "x": (0.0, 0.5),
    "Y": (0.25, 1.0),
    "y": (0.25, 0.5),
    "id": (0.0, 0.0),
}

sequence = [  # based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
    ("id", "id"),
    ("X", "X"),
    ("Y", "Y"),
    ("X", "Y"),
    ("Y", "X"),
    ("x", "id"),
    ("y", "id"),
    ("x", "y"),
    ("y", "x"),
    ("x", "Y"),
    ("y", "X"),
    ("X", "y"),
    ("Y", "x"),
    ("x", "X"),
    ("X", "x"),
    ("y", "Y"),
    ("Y", "y"),
    ("X", "id"),
    ("Y", "id"),
    ("x", "x"),
    ("y", "y"),
]

##############################################
# convert the sequence of pulses into lists of angles and amplitudes
# X - pi pulse amp - 1, x - pi/2 pulse amp - 0.5, rotation angle - 0
# Y - pi pulse amp - 1, y - pi/2 pulse amp - 0.5, rotation angle - pi/2
##############################################
angle_array1 = [rot_dict[element[0]][0] for element in sequence]
angle_array2 = [rot_dict[element[1]][0] for element in sequence]
amp_array1 = [rot_dict[element[0]][1] for element in sequence]
amp_array2 = [rot_dict[element[1]][1] for element in sequence]


def all_xy(amplitude):
    with program() as prog:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        angle1 = declare(fixed)
        angle2 = declare(fixed)
        amp1 = declare(fixed)
        amp2 = declare(fixed)
        state_estimate = declare(int)
        sigma_z = declare_stream()
        with for_(n, 0, n < Navg, n + 1):
            with for_each_(
                (angle1, angle2, amp1, amp2),
                (angle_array1, angle_array2, amp_array1, amp_array2),
            ):
                reset_qubit()

                align("qubit", "res")
                frame_rotation_2pi(
                    angle1, "qubit"
                )  # rotate by pi/2 (relative to X) to achieve Y rotation using pi pulse
                play("pi_gauss_op_qubit" * amp(amp1 * amplitude), "qubit")
                frame_rotation_2pi(-angle1 + angle2, "qubit")

                play("pi_gauss_op_qubit" * amp(amp2 * amplitude), "qubit")
                frame_rotation_2pi(-angle2, "qubit")

                align("qubit", "res")
                measure(
                    "meas_op_res",
                    "res",
                    None,
                    demod.full("integ_w_c", I),  # cos integration weights for I
                    demod.full("integ_w_s", Q),  # sin integration weights for Q
                )

                save(I, "I")
                save(Q, "Q")
                with if_(Q > thresh):
                    assign(state_estimate, 1)  # excited state
                    save(state_estimate, sigma_z)
                with else_():
                    assign(state_estimate, -1)  # ground state
                    save(state_estimate, sigma_z)
        with stream_processing():
            sigma_z.buffer(
                len(sequence)
            ).average().save()  # calculate the expectation of the pauli z operator for all combination of pulses

    return prog


qmm = QuantumMachinesManager()


def cost(freq, amplitude):
    config["elements"]["qubit"]["intermediate_frequency"] = freq
    qm1 = qmm.open_qm(config)
    job = qm1.execute(all_xy(amplitude), data_limit=int(1e9), duration_limit=0)

    print("waiting for values...")
    job.result_handles.wait_for_all_values(timeout=120)
    print("done.")

    target = np.array([-1] * 5 + [0] * 12 + [1] * 4)  # the goal values for the sigma z expectation values

    result = job.result_handles.sigma_z.fetch_all()

    fit = np.norm(target - result)

    return fit


# optimize over the frequency and amplitude to for the pi pulse
res = minimize(cost, np.array([1.53e6, 1]))
IF = res.x[0]
amplitude = res.x[1]
