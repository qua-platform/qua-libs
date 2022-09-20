"""
An experiment to calibrate the AC stark shift using DRAG waveforms
This protocol is described in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from macros import readout_macro

###################
# The QUA program #
###################

n_avg = 1000

cooldown_time = 5 * qubit_T1 // 4

number_of_pulses = 20

with program() as ac_stark_shift:
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

    with for_(n, 0, n < n_avg, n + 1):
        with for_(it, 0, it < number_of_pulses, it + 1):
            play("x180" * amp(1), "qubit")
            play("x180" * amp(-1), "qubit")
        align("qubit", "resonator")
        state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
        save(I, I_st)
        save(Q, Q_st)
        save(state, state_st)
        wait(cooldown_time, "resonator")
        save(n, n_st)

    with stream_processing():
        I_st.average().save("I")
        Q_st.average().save("Q")
        n_st.save("iteration")
        state_st.boolean_to_int().average().save("state")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, ac_stark_shift, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    xaxis = []
    yaxis = []

    detunings = np.arange(-3e6, 0e6, 0.06e6)

    for det in detunings:
        xaxis.append(det)

        x180_wf, x180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                x180_amp, x180_len, x180_sigma, alpha=drag_coef, anharmonicity=anharmonicity, detuning=det
            )
        )
        x180_I_wf = x180_wf
        x180_Q_wf = x180_der_wf

        config["waveforms"]["x180_I_wf"]["samples"] = x180_I_wf.tolist()
        config["waveforms"]["x180_Q_wf"]["samples"] = x180_Q_wf.tolist()

        qm = qmm.open_qm(config)

        job = qm.execute(ac_stark_shift)
        # Get results from QUA program
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"])

        while results.is_processing():
            # Fetch results
            I, Q, state, iteration = results.fetch_all()
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())

        yaxis.append(state)

    plt.plot(xaxis, yaxis)
    plt.xlabel("Detuning [Hz]")
    plt.ylabel("State population")
    plt.title("AC stark shift calibration")
