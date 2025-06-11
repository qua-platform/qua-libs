from qm.qua import *
from configuration import *
from qm import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close

n_avg = 100

cooldown_time = 5 * qubit_T1 // 4

a_min = -1
a_max = 1
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

resonator_cooldown = 500

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
    I_g = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's + da/2 to include a_max (This is only for fixed!)
        with for_(*from_array(a, amps)):
            with for_(it, iter_min, it <= iter_max, it + d):
                measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                # To prepare the ground state we used -0.0003 which is a more strict threshold (3 sigma)
                # to guarantee higher ground state fidelity
                with while_(I_g > -0.0003):
                    measure("readout", "resonator", None, dual_demod.full("rotated_cos", "rotated_sin", I_g))
                align()
                wait(resonator_cooldown)
                with for_(pulses, iter_min, pulses <= it, pulses + d):
                    play("x180" * amp(1, 0, 0, a), "qubit")
                    play("x180" * amp(-1, 0, 0, -a), "qubit")
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
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(iters)).buffer(len(amps)).average().save("I")
        Q_st.buffer(len(iters)).buffer(len(amps)).average().save("Q")
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(len(iters)).buffer(len(amps)).average().save("state")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="192.168.88.10", port=80)
# Open quantum machine
qm = qmm.open_qm(config, close_other_machines=False)
# Execute QUA program
job = qm.execute(drag)

# Get results from QUA program
results = fetching_tool(job, data_list=["state", "iteration"], mode="live")
# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

while results.is_processing():
    # Fetch results
    state, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    plt.cla()
    plt.pcolor(iters, amps * drag_coef, state, cmap="magma")
    plt.axhline(y=0.01)
    plt.xlabel("drag coef")

# Close quantum machine
qm.close()
