from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_avg = 10000

cooldown_time = 5 * qubit_T1

a_min = 0.8
a_max = 1.2
da = 0.005
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

max_nb_of_pulses = 80  # Number of played qubit pulses for getting a better estimate of the pi amplitude
nb_of_pulses = np.arange(0, max_nb_of_pulses, 2)  # Always play a odd/even number of pulses to end up in the same state

with program() as power_rabi:
    n = declare(int)
    n2 = declare(int)
    n_rabi = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(n_rabi, nb_of_pulses)):
            with for_(*from_array(a, amps)):
                # Loop for error amplification (perform many qubit pulses)
                with for_(n2, 0, n2 < n_rabi, n2 + 1):
                    play("x180" * amp(a), "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                wait(cooldown_time * u.ns, "resonator")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amps)).buffer(len(nb_of_pulses)).average().save("I")
        Q_st.buffer(len(amps)).buffer(len(nb_of_pulses)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

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
        plt.pcolor(amps * x180_amp, nb_of_pulses, I)
        plt.xlabel("Rabi pulse amplitude [V]")
        plt.ylabel("# of Rabi pulses")
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()