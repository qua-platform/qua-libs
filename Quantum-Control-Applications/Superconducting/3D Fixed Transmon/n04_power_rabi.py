"""
power_rabi.py: A Rabi experiment sweeping the amplitude of the MW pulse
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

# by default
# gauss_amp = 0.4
# gauss_len = 16

###################
# The QUA program #
###################

n_avg = 1000

cooldown_time = int(5 * qubit.T1 * 1e9 // 4)

a_min = 0.0
a_max = 1.0
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

with program() as power_rabi:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            play("gauss" * amp(a), qubit.name, duration=qubit.pulse_params.length * 1e9 // 4)
            align(qubit.name, resonator.name)
            measure(
                "CLEAR",
                resonator.name,
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            # wait(cooldown_time, resonator.name)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.opx_ip, port=83)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(build_config(machine), power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    qm = qmm.open_qm(build_config(machine))

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
        plt.plot(amps * gauss_amp, I, ".", label="I")
        plt.plot(amps * gauss_amp, Q, ".", label="Q")
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("I & Q amplitude [a.u.]")
        plt.legend()
        plt.pause(0.1)

# update parameters
###################
ready = True
pi_amp = 0.15
qubit.pulse_params.amplitude.x180 = pi_amp
qubit.pulse_params.amplitude.x90 = pi_amp / 2
qubit.pulse_params.amplitude.minus_x90 = -pi_amp / 2
qubit.pulse_params.amplitude.y180 = pi_amp
qubit.pulse_params.amplitude.y90 = pi_amp / 2
qubit.pulse_params.amplitude.minus_y90 = -pi_amp / 2

if ready:
    machine._save("quam_bootstrap_state.json", flat_data=False)
