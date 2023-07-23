"""
ramsey.py: Measures T2*
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

###################
# The QUA program #
###################

tau_min = 4  # in clock cycles
tau_max = 100  # in clock cycles
d_tau = 2  # in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus

f_min = qubit.f_01 - qubit.lo - 10e6
f_max = qubit.f_01 - qubit.lo + 10e6
df = 0.1e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

n_avg = 1e2
cooldown_time = int(5 * qubit.T1 * 1e9 // 4)

with program() as ramsey:
    n = declare(int)
    n_st = declare_stream()
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)
    f = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            update_frequency(qubit.name, f)
            with for_(*from_array(tau, taus)):
                play("pi_half", qubit.name)
                wait(tau, qubit.name)
                play("pi_half", qubit.name)
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
        I_st.buffer(len(taus)).buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(taus)).buffer(len(freqs)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.opx_ip)

#######################
# Simulate or execute #
#######################

simulate = False

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(build_config(machine), ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(build_config(machine))

    job = qm.execute(ramsey)
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
        plt.pcolor(4 * taus, freqs, I, label="I")
        # plt.pcolor(4 * taus, freqs, Q, label="Q")
        plt.xlabel("Idle time [ns]")
        plt.ylabel("Freqs [MHz]")
        plt.title("Chevron Ramsey")
        plt.legend()
        plt.pause(0.1)

# update parameters
####################
ready = False
qubit.f_01 = qubit.lo + 150e6

if ready:
    machine._save("quam_bootstrap_state.json", flat_data=False)
