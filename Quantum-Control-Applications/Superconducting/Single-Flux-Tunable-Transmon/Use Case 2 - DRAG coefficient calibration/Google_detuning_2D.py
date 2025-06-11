from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

n_avg = 1000

cooldown_time = 5 * qubit_T1 // 4

a_min = -1
a_max = 1
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

iter_min = 0
iter_max = 50
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

resonator_cooldown = 500

with program() as drag:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    state = declare(bool)
    state_st = declare_stream()
    it = declare(int)
    pulses = declare(int)
    I_g = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's + da/2 to include a_max (This is only for fixed!)
        # with for_(a, a_min, a < a_max + da / 2, a + da):
        with for_(it, iter_min, it <= iter_max, it + d):
            measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            # To prepare the ground state we used -0.0003 which is a more strict threshold (3 sigma)
            # to guarantee higher ground state fidelity
            with while_(I_g > -0.0003):
                measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            align()
            wait(resonator_cooldown)

            with for_(pulses, iter_min, pulses <= it, pulses + d):
                play("x180" * amp(1), "qubit")
                play("x180" * amp(-1), "qubit")

            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(iters)).average().save("I")
        Q_st.buffer(len(iters)).average().save("Q")
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(len(iters)).average().save("state")

qmm = QuantumMachinesManager(host="192.168.88.10", port=80)

xaxis = []
yaxis = []

for j in range(-20, 20):
    det = j * 0.1e6
    print(det)
    xaxis.append(det)

    x180_wf, x180_der_wf = np.array(
        drag_cosine_pulse_waveforms(x180_amp, x180_len, alpha=drag_coef, anharmonicity=(-0.163e9), detuning=det)
    )

    config["waveforms"]["x180_wf"]["samples"] = x180_wf.tolist()
    config["waveforms"]["x180_der_wf"]["samples"] = x180_der_wf.tolist()

    qm = qmm.open_qm(config, close_other_machines=False)

    job = qm.execute(drag)
    res_handles = job.result_handles
    I_handle = res_handles.get("I")
    Q_handle = res_handles.get("Q")
    iteration_handle = res_handles.get("iteration")
    I_handle.wait_for_values(1)
    Q_handle.wait_for_values(1)
    iteration_handle.wait_for_values(1)
    state_handle = res_handles.get("state")
    state_handle.wait_for_values(1)
    next_percent = 0.1  # First time print 10%

    def on_close(event):
        event.canvas.stop_event_loop()
        job.halt()

    while res_handles.is_processing():
        # plt.cla()
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        iteration = iteration_handle.fetch_all()
        if iteration / n_avg > next_percent:
            percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
            print(f"{percent}%", end=" ")
            next_percent = percent / 100 + 0.1  # Print every 10%

    plt.cla()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    state = state_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    print(f"{round(iteration / n_avg * 100)}%")
    yaxis.append(state)
    qm.close()

plt.pcolor(iters, xaxis, yaxis)

np.savez("coarse_det.npz", iters, xaxis, yaxis)
