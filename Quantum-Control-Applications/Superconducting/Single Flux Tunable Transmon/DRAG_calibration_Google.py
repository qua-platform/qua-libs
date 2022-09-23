"""
An experiment to calibrate the DRAG coefficient: drag_coef
This protocol is described in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from macros import single_measurement
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

# set the drag_coef in the configuration
drag_coef = 1

n_avg = 1000

cooldown_time = 5 * qubit_T1 // 4

a_min = 0.0
a_max = 1.0
da = 0.1
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

with program() as drag:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    state = declare(bool)
    state_st = declare_stream()
    it = declare(int)
    pulses = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            with for_(*from_array(it, iters)):
                with for_(pulses, iter_min, pulses <= it, pulses + d):
                    play("x180" * amp(1, 0, 0, a), "qubit")
                    play("x180" * amp(-1, 0, 0, -a), "qubit")
            align("qubit", "resonator")
            state, I, Q = single_measurement(threshold=ge_threshold, state=state, I=I, Q=Q)
            save(I, I_st)
            save(Q, Q_st)
            save(state, state_st)
            wait(cooldown_time, "resonator")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(iters)).buffer(len(amps)).average().save("I")
        Q_st.buffer(len(iters)).buffer(len(amps)).average().save("Q")
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(len(iters)).buffer(len(amps)).average().save("state")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)  # in clock cycles
    job = qmm.simulate(config, drag, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)

    job = qm.execute(drag)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.cla()
        plt.pcolor(iters, amps * drag_coef, state, cmap="magma")
        plt.xlabel("Number of iterations")
        plt.ylabel("Drag coef")
