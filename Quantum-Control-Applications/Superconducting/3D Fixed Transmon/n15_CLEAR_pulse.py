"""
n15_CLEAR_pulse.py: insipired on PRL 5, 011001 (2016)
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

relax_min = 4  # in clock cycles
relax_max = 100  # in clock cycles
d_relax = 2  # in clock cycles
relaxs = np.arange(relax_min, relax_max + 0.1, d_relax)  # + 0.1 to add tau_max to taus

n_avg = 1e4
cooldown_time = int(5 * qubit.T1 * 1e9 // 4)
buffer = 1000 

detuning = 1 * u.MHz  # in Hz

with program() as ramsey:
    n = declare(int)
    n_st = declare_stream()
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)
    relax = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(relax, relaxs)):
            with for_(*from_array(tau, taus)):
                # populate the cavity
                play('readout', resonator.name)
                align()
                wait(relax)
                align()
                # ramsey sequence
                play("pi_half", qubit.name)
                wait(tau, qubit.name)
                play("pi_half", qubit.name)
                align(qubit.name, resonator.name)
                wait(buffer//4)
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
                reset_frame(qubit.name)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(taus)).buffer(len(relaxs)).average().save("I")
        Q_st.buffer(len(taus)).buffer(len(relaxs)).average().save("Q")
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
        plt.pcolor(4 * taus, 4 * relaxs, I, label="I")
        # plt.plot(4 * taus, Q, ".", label="Q")
        plt.xlabel("Ramsey time [ns]")
        plt.ylabel("Relax time [ns]")
        plt.title("Ramsey with frame rotation")
        plt.legend()
        plt.pause(0.1)

# update parameters
###################
ready = False
qubit.T2 = 11e-06
# qubit.f_01 =

if ready:
    machine._save("quam_bootstrap_state.json", flat_data=False)
