# %%
"""
ALL-XY MEASUREMENT
The program consists in playing a random sequence of predefined gates after which the theoretical qubit state is known.
See [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details.

The sequence of gates defined below is based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
This protocol checks that the single qubit gates (x180, x90, y180 and y90) are properly defined and calibrated and can
thus be used as a preliminary step before randomized benchmarking.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""

from pathlib import Path

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM, Transmon, ReadoutResonator
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save

import matplotlib.pyplot as plt
import numpy as np

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
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)

##############################
# Program-specific variables #
##############################
n_points = 1_000


# All XY sequences. The sequence names must match corresponding operation in the config
sequence = [
    ("I", "I"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "I"),
    ("y90", "I"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "I"),
    ("y180", "I"),
    ("x90", "x90"),
    ("y90", "y90"),
]


# All XY macro generating the pulse sequences from a python list.
def allXY(pulses, qubit: Transmon, resonator: ReadoutResonator):
    """
    Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
    **Example:** I, Q = allXY(['I', 'y90'])
    :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
        operations in the config except for the identity operation that must be called 'I'.
    :param qubit: The qubit element as defined in the config.
    :param resonator: The resonator element as defined in the config.
    :return: two QUA variables for the 'I' and 'Q' quadratures measured after the sequence.
    """
    I_xy = declare(fixed)
    Q_xy = declare(fixed)
    for pulse in pulses:
        if pulse != "I":
            qubit.xy.play(pulse)  # Either play the sequence
        else:
            qubit.xy.wait(qubit.xy.operations["x180"].length * u.ns)  # or wait if sequence is identity

    align()
    # Play the readout on the other resonator to measure in the same condition as when optimizing readout
    for other_qubit in qubits:
        if other_qubit.resonator != resonator:
            other_qubit.resonator.play("readout")
    resonator.measure("readout", qua_vars=(I_xy, Q_xy))
    return I_xy, Q_xy


###################
# The QUA program #
###################
# Define the QUA program in a function so that one can call it for the two qubits successively
def get_prog(qubit, resonator):
    with program() as all_xy:
        n = declare(int)
        n_st = declare_stream()
        r = Random()  # Pseudo random number generator
        r_ = declare(int)  # Index of the sequence to play
        # The result of each set of gates is saved in its own stream
        I_st = [declare_stream() for _ in range(21)]
        Q_st = [declare_stream() for _ in range(21)]

        # Bring the active qubits to the minimum frequency point
        machine.apply_all_flux_to_min()

        with for_(n, 0, n < n_points, n + 1):
            save(n, n_st)
            # Get a value from the pseudo-random number generator on the OPX FPGA
            assign(r_, r.rand_int(21))
            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
            wait(machine.thermalization_time * u.ns)
            # Plays a random XY sequence
            # The switch/case method allows to map a python index (here "i") to a QUA number (here "r_") in order to switch
            # between elements in a python list (here "sequence") that cannot be converted into a QUA array (here because it
            # contains strings).
            with switch_(r_):
                for i in range(21):
                    with case_(i):
                        # Play the all-XY sequence corresponding to the drawn random number
                        I, Q = allXY(sequence[i], qubit, resonator)
                        # Save the 'I' & 'Q' quadratures to their respective streams
                        save(I, I_st[i])
                        save(Q, Q_st[i])

        with stream_processing():
            n_st.save("iteration")
            for i in range(21):
                I_st[i].average().save(f"I{i + 1}")
                Q_st[i].average().save(f"Q{i + 1}")
    return all_xy


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    program = get_prog(qubits[0], qubits[0].resonator)
    job = qmm.simulate(config, program, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Perform AllXY for each qubit separately
    for qubit in qubits:
        # Prepare data for saving
        data = {}
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Calibrate the active qubits
        # machine.calibrate_octave_ports(qm)
        all_xy = get_prog(qubit, qubit.resonator)
        job = qm.execute(all_xy)
        data_list = ["iteration"] + np.concatenate([[f"I{i + 1}", f"Q{i + 1}"] for i in range(21)]).tolist()
        results = fetching_tool(job, data_list, mode="live")
        fig, ax = plt.subplots(2, 1)

        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            n = res[0]
            I = np.array(res[1::2])
            Q = np.array(res[2::2])
            # Progress bar
            progress_counter(n, n_points, start_time=results.start_time)
            # Plot results
            ax[0].cla()
            ax[0].plot(-I, "-*")
            ax[0].plot([np.max(-I)] * 5 + [(np.mean(-I))] * 12 + [np.min(-I)] * 4, "-")
            ax[0].set_ylabel("I quadrature [a.u.]")
            ax[0].set_xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=45)
            ax[1].cla()
            ax[1].plot(-Q, "-*")
            ax[1].plot([np.max(-Q)] * 5 + [(np.mean(-Q))] * 12 + [np.min(-Q)] * 4, "-")
            ax[1].set_ylabel("Q quadrature [a.u.]")
            ax[1].set_xticks(ticks=range(21), labels=[str(el) for el in sequence], rotation=45)
            plt.suptitle(f"All XY {qubit.name}")
            plt.tight_layout()
            plt.pause(0.1)

            data[qubit.name] = {"I": I, "Q": Q, "figure": fig}
        plt.show()
        # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
        qm.close()

        # Save data from the node
        node_save(machine, "all_xy", data, additional_files=True)

# %%
