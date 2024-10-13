"""
        DRAG PULSE CALIBRATION (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively while varying the DRAG
coefficient alpha. After such a sequence, the qubit is expected to always be in the ground state if the DRAG
coefficient has the correct value. Note that the idea is very similar to what is done in power_rabi_error_amplification.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the DRAG coefficient to a non-zero value in the config: such as drag_coef = 1
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the DRAG coefficient (drag_coef) in the configuration.
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib

matplotlib.use("TKAgg")

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
qubits = machine.active_qubits
num_qubits = len(qubits)

qb_idx = 1
qb = qubits[qb_idx]
qb.xy.operations["x180"].alpha = 1.0

config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
###################
# The QUA program #
###################
n_avg = 1000

# Scan the DRAG coefficient pre-factor
a_min = 0.0
a_max = 1.0
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

# Scan the number of pulses
iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)
print(qb.xy.operations["x180"])

with program() as drag:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the DRAG coefficient pre-factor
    it = declare(int)  # QUA variable for the number of qubit pulses
    pulses = declare(int)  # QUA variable for counting the qubit pulses
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=num_qubits)

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the pulse amplitude
            with for_(*from_array(it, iters)):  # QUA for_ loop for sweeping the number of pulses
                # Loop for error amplification (perform many qubit pulses with varying DRAG coefficients)
                with for_(pulses, iter_min, pulses <= it, pulses + d):
                    play("x180" * amp(1, 0, 0, a), qb.xy.name)
                    play("x180" * amp(-1, 0, 0, -a), qb.xy.name)
                    # qb.xy.play("x180", amplitude_scale=(1, 0, 0, a))
                    # qb.xy.play("x180", amplitude_scale=(-1, 0, 0, -a))
                # Align the two elements to measure after playing the qubit pulses.
                align()
                # Measure the state of the resonators
                multiplexed_readout(qubits, I, I_st, Q, Q_st, sequential=False)
                # Wait for the qubits to decay to the ground state
                wait(machine.get_thermalization_time * u.ns)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        for i in range(num_qubits):
            I_st[i].buffer(len(iters)).buffer(len(amps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(iters)).buffer(len(amps)).average().save(f"Q{i + 1}")
        n_st.save("n")

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, drag, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(drag)
    # Get results from QUA program
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        I = [u.demod2volts(I[i], qubits[i].resonator.operations["readout"].length) for i in range(num_qubits)]
        Q = fetched_data[2::2]
        Q = [u.demod2volts(Q[i], qubits[i].resonator.operations["readout"].length) for i in range(num_qubits)]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("DRAG calibration (Google)")
        plt.subplot(231)
        plt.cla()
        plt.pcolor(iters, amps * qb.xy.operations["x180"].alpha, I[qb_idx], cmap="magma")
        plt.xlabel("Number of iterations")
        plt.ylabel(r"Drag coefficient $\alpha$")
        plt.title("I [V]")
        plt.subplot(232)
        plt.cla()
        plt.pcolor(iters, amps * qb.xy.operations["x180"].alpha, Q[qb_idx], cmap="magma")
        plt.xlabel("Number of iterations")
        plt.title("Q [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot(amps * qb.xy.operations["x180"].alpha, np.sum(I[qb_idx], axis=1))
        plt.xlabel(r"Drag coefficient $\alpha$")
        plt.ylabel("Sum along the iterations")
        plt.tight_layout()
        plt.pause(0.1)

    optimal_drag_coef = qb.xy.operations["x180"].alpha * amps[np.argmin(np.sum(I[qb_idx], axis=1))]
    print(f"Optimal drag_coef = {optimal_drag_coef:.3f}")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Save data from the node
    data = {"figure": plt.gcf(), "optimal_drag_coef": optimal_drag_coef}
    node_save(machine, "drag_calibration", data, additional_files=True)
    plt.show()
