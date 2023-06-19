"""
wigner_tomography.py: A template for performing Wigner tomography using a superconducting qubit
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig, LoopbackInterface


##############################
# Program-specific variables #
##############################
cavity_element = "resonator"
threshold = ge_threshold
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)
chi = 10 * u.MHz / u.GHz  # cavity  coupling strength in GHz
revival_time = int(np.pi / chi) // 4  # Revival time in multiples of 4 ns
# range to sample alpha
n_points = 20
alpha = np.linspace(-2, 2, n_points)
# scale alpha to get the required power amplitude for the pulse
amp_displace = list(-alpha / np.sqrt(2 * np.pi) / 4)
n_avg = 100

###################
# The QUA program #
###################
with program() as wigner_tomo:
    amp_dis = declare(fixed, value=amp_displace)
    n = declare(int)
    i = declare(int)
    r = declare(int)
    ground = declare(int)
    excited = declare(int)
    I = declare(fixed)
    Q = declare(fixed)

    ground_st = declare_stream()
    excited_st = declare_stream()

    with for_(r, 0, r < n_points, r + 1):
        with for_(i, 0, i < n_points, i + 1):
            assign(ground, 0)
            assign(excited, 0)
            with for_(n, 0, n < n_avg, n + 1):
                # Displace the cavity
                play("displace" * amp(amp_dis[r], 0, 0, amp_dis[i]), cavity_element)
                align(cavity_element, "qubit")
                # The Ramsey sequence with idle time set to pi / chi
                play("x90", "qubit")
                wait(revival_time, "qubit")
                play("x90", "qubit")
                # Readout the resonator
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Single shot detection and ground/excited state assignment
                with if_(I < threshold):
                    assign(ground, ground + 1)
                with else_():
                    assign(excited, excited + 1)
                # wait and let all elements relax
                wait(cooldown_time, cavity_element, "qubit", "resonator")
            save(ground, ground_st)
            save(excited, excited_st)

    with stream_processing():
        ground_st.buffer(n_points, n_points).save("ground")
        excited_st.buffer(n_points, n_points).save("excited")

######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(qop_ip)

qm = qmm.open_qm(config)

simulate = True
if simulate:
    simulation_config = SimulationConfig(
        duration=int(2e5),  # need to run the simulation for long enough to get all points
        simulation_interface=LoopbackInterface(
            [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
            latency=200,
            noisePower=0.05**2,
        ),
    )
    job = qm.simulate(wigner_tomo, simulation_config)
    job.get_simulated_samples().con1.plot()  # to see the output pulses
else:
    job = qm.execute(wigner_tomo)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["ground", "excited"], mode="wait_for_all")
    # Fetch results
    ground, excited = results.fetch_all()
    # Plot results
    fig = plt.figure()
    parity = excited - ground  # derive the parity
    wigner = 2 / np.pi * parity / n_avg  # derive the average wigner function
    plt.cla()
    ax = plt.subplot()
    pos = ax.imshow(
        wigner,
        cmap="Blues",
        vmin=-2 / np.pi,
        vmax=2 / np.pi,
    )
    fig.colorbar(pos, ax=ax)
    ax.set_xticks(range(len(alpha)))
    ax.set_xticklabels(
        f"{alpha[i]:.2f}" if ((i / (len(alpha) / 5)).is_integer() or (i == len(alpha) - 1)) else ""
        for i in range(len(alpha))
    )
    ax.set_yticks(range(len(alpha)))
    ax.set_yticklabels(
        f"{alpha[i]:.2f}" if ((i / (len(alpha) / 5)).is_integer() or (i == len(alpha) - 1)) else ""
        for i in range(len(alpha))
    )
    ax.set_xlabel("Im(alpha)")
    ax.set_ylabel("Re(alpha)")
    ax.set_title("Wigner function")
