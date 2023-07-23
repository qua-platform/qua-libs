"""
time_rabi.py: A Rabi experiment sweeping the duration of the MW pulse
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig

from quam import QuAM

machine = QuAM("quam_bootstrap_state.json", flat_data=False)
resonator = machine.resonators[0]
qubit = machine.qubits[0]

# by default
# gauss_amp = 0.4
# gauss_len = 16

###################
# The QUA program #
###################

n_avg = 10000

cooldown_time = int(5 * qubit.T1 * 1e9 // 4)

t_min = 10
t_max = 1000
dt = 10
taus = np.arange(t_min, t_max + 0.1, dt)  # + 0.1 to add t_max to taus


with program() as time_rabi:
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's <= to include t_max (This is only for integers!)
        with for_(t, t_min, t <= t_max + dt / 2, t + dt):
            play("gauss" * amp(qubit.pulse_params.amplitude.x180 / gauss_amp), qubit.name, duration=t)
            align(qubit.name, resonator.name)
            measure(
                "readout",
                resonator.name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            wait(cooldown_time, resonator.name)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.opx_ip, port=83)

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(build_config(machine), time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(build_config(machine))

    job = qm.execute(time_rabi)
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
        plt.xlabel("Rabi pulse duration [ns]")
        plt.ylabel("I & Q amplitude [a.u.]")
        plt.title("Time Rabi")
        plt.legend()
        plt.pause(0.1)
