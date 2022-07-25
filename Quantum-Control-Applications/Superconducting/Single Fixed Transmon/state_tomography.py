"""
A template to perform state tomography
"""
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from macros import readout_macro

###################
# The QUA program #
###################

n_avg = 100

cooldown_time = 5 * qubit_T1 // 4

with program() as state_tomography:
    n = declare(int)  # variable for average loop
    n_st = declare_stream()  # stream for 'n'
    c = declare(int)  # variable for switch case
    state = declare(bool)
    state_st = declare_stream()
    with for_(n, 0, n < n_avg, n + 1):
        with for_(c, 0, c <= 2, c + 1):
            # Add here whatever state you want to characterize
            with switch_(c):
                with case_(0):  # projection along X
                    play("-x90", "qubit")
                    align("qubit", "resonator")
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    save(state, state_st)
                    wait(cooldown_time, "qubit", "resonator")
                with case_(1):  # projection along Y
                    play("x90", "qubit")
                    align("qubit", "resonator")
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    save(state, state_st)
                    wait(cooldown_time, "qubit")
                with case_(2):  # projection along Z
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    save(state, state_st)
                    wait(cooldown_time, "qubit")
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(3).average().save("states")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(state_tomography)  # execute QUA program

    res_handles = job.result_handles
    res_handles.wait_for_all_values()

    states = res_handles.get("states").fetch_all()
    states = -2 * (states - 0.5)  # Converts the (0,1) -> |g>,|e> convention to (1,-1) -> |g>,|e>

    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Zero order approximation
    rho = 0.5 * (I + states[0] * sigma_x + states[1] * sigma_y + states[2] * sigma_z)
    print(f"The density matrix is:\n{rho}")
