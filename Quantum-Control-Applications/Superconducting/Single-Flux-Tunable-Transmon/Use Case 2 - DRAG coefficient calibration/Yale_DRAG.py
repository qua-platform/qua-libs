from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close

n_avg = 2500 * 4

cooldown_time = 5 * qubit_T1 // 4

a_min = -1
a_max = 1
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

resonator_cooldown = 500

with program() as drag2:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    state = declare(bool)
    state_st = declare_stream()
    state2_st = declare_stream()
    I_g = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's + da/2 to include a_max (This is only for fixed!)
        with for_(*from_array(a, amps)):
            measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            # To prepare the ground state we used -0.0003 which is a more strict threshold (3 sigma)
            # to guarantee higher ground state fidelity
            with while_(I_g > -0.0003):
                measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            align()
            wait(resonator_cooldown)
            play("x180" * amp(1, 0, 0, a), "qubit")
            play("y90" * amp(a, 0, 0, 1), "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state_st)
            # wait(cooldown_time, "resonator")

            align()

            measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            with while_(I_g > -0.0003):
                measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
            align()
            wait(resonator_cooldown)

            play("y180" * amp(a, 0, 0, 1), "qubit")
            play("x90" * amp(1, 0, 0, a), "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
            assign(state, I > ge_threshold)
            save(state, state2_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(len(amps)).average().save("state")
        state2_st.boolean_to_int().buffer(len(amps)).average().save("state2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="192.168.88.10", port=80)
# Open quantum machine
qm = qmm.open_qm(config, close_other_machines=False)
# Execute QUA program
job = qm.execute(drag2)
# Get results from QUA program
results = fetching_tool(job, data_list=["state", "state2", "iteration"], mode="live")
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

while results.is_processing():
    # Fetch results
    state, state2, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    plt.cla()
    plt.plot(amps * drag_coef, state, label="x180y90")
    plt.plot(amps * drag_coef, state2, label="y180x90")
    plt.xlabel("drag coef")
    plt.legend()

# Close quantum machine
qm.close()
