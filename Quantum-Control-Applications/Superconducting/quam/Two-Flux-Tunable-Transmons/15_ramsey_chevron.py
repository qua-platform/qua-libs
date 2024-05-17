"""
        RAMSEY CHEVRON (IDLE TIME VS FREQUENCY)
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different qubit intermediate
frequencies and idle times.
From the results, one can estimate the qubit frequency more precisely than by doing Rabi and also gets a rough estimate
of the qubit coherence time.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the qubit frequency (f_01) in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]


###################
# The QUA program #
###################
n_avg = 100  # Number of averaging loops

# Frequency detuning sweep in Hz
dfs = np.arange(-10e6, 10e6, 0.1e6)
# Idle time sweep (Must be a list of integers) - in clock cycles (4ns)
t_delay = np.arange(4, 300, 4)


with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the idle time
    df = declare(int)  # QUA variable for the qubit detuning

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubits frequency
            update_frequency(q1.xy.name, df + q1.xy.intermediate_frequency)
            update_frequency(q2.xy.name, df + q2.xy.intermediate_frequency)

            with for_(*from_array(t, t_delay)):
                # qubit 1
                q1.xy.play("x90")
                q1.xy.wait(t)
                q1.xy.play("x90")

                # qubit 2
                q2.xy.play("x90")
                q2.xy.wait(t)
                q2.xy.play("x90")

                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(machine, I, I_st, Q, Q_st)
                # Wait for the qubits to decay to the ground state
                wait(machine.get_thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(t_delay)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(t_delay)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(t_delay)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(t_delay)).buffer(len(dfs)).average().save("Q2")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ramsey)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1 = u.demod2volts(I1, q1.resonator.operations["readout"].length)
        Q1 = u.demod2volts(Q1, q1.resonator.operations["readout"].length)
        I2 = u.demod2volts(I2, q2.resonator.operations["readout"].length)
        Q2 = u.demod2volts(Q2, q2.resonator.operations["readout"].length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        plt.suptitle("Ramsey chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, I1)
        plt.title(f"{q1.name} - I, f_01={int(q1.f_01 / u.MHz)} MHz")
        plt.ylabel("detuning [MHz]")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, Q1)
        plt.title(f"{q1.name} - Q")
        plt.xlabel("Idle time [ns]")
        plt.ylabel("detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, I2)
        plt.title(f"{q2.name} - I, f_01={int(q2.f_01 / u.MHz)} MHz")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(4 * t_delay, dfs / u.MHz, Q2)
        plt.title(f"{q2.name} - Q")
        plt.xlabel("Idle time [ns]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {
        f"{q1.name}_amplitude": 4 * t_delay,
        f"{q1.name}_frequency": dfs + q1.xy.intermediate_frequency,
        f"{q1.name}_I": np.abs(I1),
        f"{q1.name}_Q": np.angle(Q1),
        f"{q2.name}_amplitude": 4 * t_delay,
        f"{q2.name}_frequency": dfs + q2.xy.intermediate_frequency,
        f"{q2.name}_I": np.abs(I2),
        f"{q2.name}_Q": np.angle(Q2),
        "figure": fig,
    }
    node_save("ramsey_chevron", data, machine)
