"""
        ALL-XY MEASUREMENT
The program consists in playing a random sequence of predefined gates after which the theoretical qubit state is known.
See [Reed's Thesis](https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf) for more details.

The sequence of gates defined below is based on https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf
This protocol checks that the single qubit gates (x180, x90, y180 and y90) are properly defined and calibrated and can
thus be used as a preliminary step before randomized benchmarking.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt


##############################
# Program-specific variables #
##############################
n_avg = 1e4

# All-XY sequences. The sequence names must match corresponding operation in the config
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


# All-XY macro generating the pulse sequences from a python list.
def allXY(pulses):
    """
    Generate a QUA sequence based on the two operations written in pulses. Used to generate the all XY program.
    **Example:** I, Q = allXY(['I', 'y90'])

    :param pulses: tuple containing a particular set of operations to play. The pulse names must match corresponding
        operations in the config except for the identity operation that must be called 'I'.
    :return: two QUA variables for the 'I' and 'Q' quadratures measured after the sequence.
    """
    I_xy = declare(fixed)
    Q_xy = declare(fixed)
    # Play the 1st gate or wait if the gate is identity
    if pulses[0] != "I":
        play(pulses[0], "qubit")  # Either play the sequence
    else:
        wait(x180_len // 4, "qubit")  # or wait if sequence is identity
    # Play the 2nd gate or wait if the gate is identity
    if pulses[1] != "I":
        play(pulses[1], "qubit")  # Either play the sequence
    else:
        wait(x180_len // 4, "qubit")  # or wait if sequence is identity
    # Align the two elements to measure after playing the qubit gates.
    align("qubit", "resonator")
    # Measure the state of the resonator and return the two quadratures
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "rotated_sin", I_xy),
        dual_demod.full("rotated_minus_sin", "rotated_cos", Q_xy),
    )
    return I_xy, Q_xy


###################
# The QUA program #
###################
with program() as ALL_XY:
    n = declare(int)
    r = Random()  # Pseudo random number generator
    r_ = declare(int)  # Index of the sequence to play
    # The result of each set of gates is saved in its own stream
    I_st = [declare_stream() for _ in range(len(sequence))]
    Q_st = [declare_stream() for _ in range(len(sequence))]
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Get a value from the pseudo-random number generator on the OPX FPGA
        assign(r_, r.rand_int(len(sequence)))
        # Wait for the qubit to decay to the ground state - Can be replaced by active reset
        wait(thermalization_time * u.ns, "qubit")
        # Plays a random XY sequence
        # The switch/case method allows to map a python index (here "i") to a QUA number (here "r_") in order to switch
        # between elements in a python list (here "sequence") that cannot be converted into a QUA array (here because it
        # contains strings).
        with switch_(r_):
            for i in range(len(sequence)):
                with case_(i):
                    # Play the all-XY sequence corresponding to the drawn random number
                    I, Q = allXY(sequence[i])
                    # Save the 'I' & 'Q' quadratures to their respective streams
                    save(I, I_st[i])
                    save(Q, Q_st[i])
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        for i in range(len(sequence)):
            I_st[i].average().save(f"I{i}")
            Q_st[i].average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ALL_XY, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ALL_XY)
    # Get results from QUA program
    data_list = ["iteration"] + list(np.concatenate([[f"I{i}", f"Q{i}"] for i in range(len(sequence))]))
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        res = results.fetch_all()
        I = -np.array(res[1::2])
        Q = -np.array(res[2::2])
        n = res[0]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        plt.suptitle("All XY")
        plt.subplot(211)
        plt.cla()
        plt.plot(I, "bx", label="Experimental data")
        plt.plot([np.max(I)] * 5 + [(np.mean(I))] * 12 + [np.min(I)] * 4, "r-", label="Expected value")
        plt.ylabel("I quadrature [a.u.]")
        plt.xticks(ticks=range(len(sequence)), labels=["" for _ in sequence], rotation=45)
        plt.legend()
        plt.subplot(212)
        plt.cla()
        plt.plot(Q, "bx", label="Experimental data")
        plt.plot([np.max(Q)] * 5 + [(np.mean(Q))] * 12 + [np.min(Q)] * 4, "r-", label="Expected value")
        plt.ylabel("Q quadrature [a.u.]")
        plt.xticks(ticks=range(len(sequence)), labels=[str(el) for el in sequence], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.pause(0.1)
