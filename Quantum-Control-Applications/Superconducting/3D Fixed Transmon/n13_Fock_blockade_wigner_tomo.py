"""
wigner_tomography.py: A template for performing Wigner tomography using a superconducting qubit
reference: supplemental material of arxiv:2001. 03217
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig, LoopbackInterface

from quam import QuAM

machine = QuAM("quam_bootstrap_state.json", flat_data=False)
resonator = machine.resonators[0]
qubit = machine.qubits[0]
storage = machine.storage[0]

##############################
# Program-specific variables #
##############################
cooldown_time = int(5 * qubit.T1 * 1e9 // 4)
# range to sample alpha
n_points = 20
alpha = np.linspace(-1, 1, n_points)
n_avg = 100

# choose photon number
N = 2  # N = 0 is vacuum
dynamics_duration = 100

###################
# The QUA program #
###################
with program() as wigner_tomo:
    amp_dis = declare(fixed, value=alpha)
    n = declare(int)
    i = declare(int)
    r = declare(int)
    state_plus = declare(bool)
    state_minus = declare(bool)
    state = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    n_st = declare_stream()

    state_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        update_frequency(qubit.name, qubit.f_01 - qubit.lo - N*qubit.storage_chi)
        with for_(r, 0, r < n_points, r + 1):
            with for_(i, 0, i < n_points, i + 1):
                # Displace the cavity
                play('x180', qubit.name, duration=dynamics_duration+100)  # in clock cycles, does it require any pre-time for settling down?
                wait(100, storage.name)
                play('displace' * amp(amp_dis[r], 0, amp_dis[i], 0), storage.name, duration=dynamics_duration)

                align(storage.name, qubit.name)
                # The Ramsey sequence with idle time set to pi / chi
                play("short_x90", qubit.name)
                wait(int(1e9/(2 * qubit.storage_chi)), qubit.name)
                play("short_x90", qubit.name)
                # Readout the resonator
                align(qubit.name, resonator.name)
                measure(
                    "readout",
                    resonator.name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Single shot detection and ground/excited state assignment
                assign(state_plus, I > resonator.readout_params.ge_threshold)
                # wait and let all elements relax
                # wait(cooldown_time)

                align()

                # Displace the cavity
                play('x180', qubit.name, duration=dynamics_duration+100)  # in clock cycles, does it require any pre-time for settling down?
                wait(100, storage.name)
                play('displace' * amp(amp_dis[r], 0, amp_dis[i], 0), storage.name, duration=dynamics_duration)

                align(storage.name, qubit.name)
                # The Ramsey sequence with idle time set to pi / chi
                play("short_x90", qubit.name)
                wait(int(1e9/(2 * qubit.storage_chi)), qubit.name)
                play("short_-x90", qubit.name)
                # Readout the resonator
                align(qubit.name, resonator.name)
                measure(
                    "readout",
                    resonator.name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Single shot detection and ground/excited state assignment
                assign(state_minus, I > resonator.readout_params.ge_threshold)
                # wait and let all elements relax
                # wait(cooldown_time)

                assign(state, Cast.to_int(state_plus) - Cast.to_int(state_minus))
                save(state, state_st)

    with stream_processing():
        state_st.buffer(n_points).buffer(n_points).average().save("state")
        n_st.save('iteration')

######################################
#  Open Communication with the QOP  #
######################################
qmm = QuantumMachinesManager(machine.opx_ip, port=83)

qm = qmm.open_qm(build_config(machine))

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=3000)  # in clock cycles
    job = qmm.simulate(build_config(machine), wigner_tomo, simulation_config)
    job.get_simulated_samples().con1.plot()  # to see the output pulses
    plt.show()
else:
    job = qm.execute(wigner_tomo)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["state", 'iteration'], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        state, iteration = results.fetch_all()
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        wigner = 2 / np.pi * state  # derive the wigner function
        plt.cla()
        plt.pcolor(alpha, alpha, wigner, cmap='magma')
        plt.xlabel('Re(alpha)')
        plt.ylabel('Im(alpha)')
        plt.title('Wigner tomography')
        plt.pause(0.1)
