from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.loops import from_array

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################

n_avg = 1_000

cooldown_time = 20_000

taus = np.arange(4, 100, 1)

qubit_index = 0

with program() as power_rabi:
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, taus)):
            # qubit cooldown
            wait(cooldown_time * u.ns, machine.resonators[qubit_index].name)

            measure(
                "readout",
                machine.resonators[qubit_index].name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )

            wait(t, machine.resonators[qubit_index].name)

            align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)

            play("x90", machine.qubits[qubit_index].name)

            wait(100)  # fixed time ramsey

            play("x90", machine.qubits[qubit_index].name)

            align(machine.qubits[qubit_index].name, machine.resonators[qubit_index].name)

            measure(
                "readout",
                machine.resonators[qubit_index].name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)

        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)

    job = qm.execute(power_rabi)
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
        # Plot results
        plt.cla()
        plt.plot(4 * taus, I, ".", label="I")
        plt.plot(4 * taus, Q, ".", label="Q")
        plt.xlabel("Delay [clock cycles]")
        plt.ylabel("I & Q amplitude [a.u.]")
        plt.legend()
        plt.pause(0.1)
    plt.show()
