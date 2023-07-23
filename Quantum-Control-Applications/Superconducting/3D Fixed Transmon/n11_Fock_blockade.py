"""
Fock_blockade.py: reference arXiv:1505.04238
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.loops import from_array
from quam import QuAM

machine = QuAM("quam_bootstrap_state.json", flat_data=False)
resonator = machine.resonators[0]
qubit = machine.qubits[0]
storage = machine.storage[0]

###################
# The QUA program #
###################

n_avg = 100

cooldown_time = int(5 * qubit.T1 * 1e9 // 4)

t_min = 4
t_max = 804
dt = 16
taus = np.arange(t_min, t_max + 0.1, dt)  # + 0.1 to add t_max to taus

# choose photon number
N = 2  # N = 0 is vacuum
k_index = 6  # up to  which level to do conditional-pi

with program() as fock_blockade:

    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    k = declare(int)


    with for_(n, 0, n < n_avg, n + 1):

        with for_(t, t_min, t <= t_max + dt / 2, t + dt):

            with for_(k, 0, k < k_index, k+1):  # conditional pi-pulse selection
                update_frequency(qubit.name, qubit.f_01 - qubit.lo - N*qubit.storage_chi)
                play('x180', qubit.name, duration=t+100)  # in clock cycles, does it require any pre-time for settling down?
                wait(100, storage.name)
                play('displace', storage.name, duration=t)
                align()
                update_frequency(qubit.name, qubit.f_01 - qubit.lo - k*qubit.storage_chi)
                play('x180', qubit.name)
                align()
                measure(
                    "readout",
                    resonator.name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                wait(cooldown_time)
            save(n, n_st)

    with stream_processing():
        I_st.buffer(k_index).buffer(len(taus)).average().save('I')
        Q_st.buffer(k_index).buffer(len(taus)).average().save('Q')
        n_st.save('iteration')


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.opx_ip, port=83)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=3000)  # in clock cycles
    job = qmm.simulate(build_config(machine), fock_blockade, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(build_config(machine))

    job = qm.execute(fock_blockade)

    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        a, b = I.shape
        plt.cla()
        for j in range(b):
            plt.plot(4 * taus, I[:, j], '.-', label=f'P_{j}')
        plt.legend()
        plt.xlabel("t [us]")
        plt.ylabel("P_k")
        plt.pause(0.1)
