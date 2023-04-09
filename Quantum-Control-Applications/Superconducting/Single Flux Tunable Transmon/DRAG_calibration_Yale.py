"""
An experiment to calibrate the DRAG coefficient: drag_coef
This protocol is described in Reed's thesis (Fig. 5.8) https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf
This protocol was also cited in: https://doi.org/10.1103/PRXQuantum.2.040202
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

with program() as drag:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I1_st = declare_stream()
    Q1_st = declare_stream()
    I2_st = declare_stream()
    Q2_st = declare_stream()
    state = declare(bool)
    state1_st = declare_stream()
    state2_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            play("x180" * amp(1, 0, 0, a), "qubit")
            play("y90" * amp(a, 0, 0, 1), "qubit")
            align("qubit", "resonator")
            state, I, Q = single_measurement(threshold=ge_threshold, state=state, I=I, Q=Q)
            save(I, I1_st)
            save(Q, Q1_st)
            save(state, state1_st)
            wait(cooldown_time, "resonator")

            align()

            play("y180" * amp(a, 0, 0, 1), "qubit")
            play("x90" * amp(1, 0, 0, a), "qubit")
            align("qubit", "resonator")
            state, I, Q = single_measurement(threshold=ge_threshold, state=state, I=I, Q=Q)
            save(I, I2_st)
            save(Q, Q2_st)
            save(state, state2_st)
            wait(cooldown_time, "resonator")
        save(n, n_st)

    with stream_processing():
        I1_st.buffer(len(amps)).average().save("I1")
        Q1_st.buffer(len(amps)).average().save("Q1")
        I2_st.buffer(len(amps)).average().save("I2")
        Q2_st.buffer(len(amps)).average().save("Q2")
        n_st.save("iteration")
        state1_st.boolean_to_int().buffer(len(amps)).average().save("state1")
        state2_st.boolean_to_int().buffer(len(amps)).average().save("state2")

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
    results = fetching_tool(job, data_list=["state1", "state2", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        state1, state2, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.cla()
        plt.plot(amps * drag_coef, state1, label="x180y90")
        plt.plot(amps * drag_coef, state2, label="y180x90")
        plt.xlabel("Drag coef")
        plt.ylabel("g-e transition probability")
        plt.legend()
        plt.tight_layout()
