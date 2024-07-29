# %%
"""
RABI CHEVRON (DURATION VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse durations.
By analyzing the results, one can determine the qubit and estimate the x180 pulse duration for a specified amplitude.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse amplitude (labeled as "pi_amp").
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "f_01", in the state.
    - Modify the qubit pulse amplitude setting, labeled as "pi_len", in the state.
    - Save the current state by calling machine.save("quam")
"""

from pathlib import Path

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
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
qubits = machine.active_qubits
num_qubits = len(qubits)

###################
# The QUA program #
###################

operation = "x180"  # The qubit operation to play
n_avg = 2  # The number of averages

# The frequency sweep with respect to the qubits resonance frequencies
dfs = np.arange(-100e6, +100e6, 1e6)
# Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
durations = np.arange(4, 100, 2)

with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit detuning
    t = declare(int)  # QUA variable for the qubit pulse duration

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            for qubit in qubits:
                update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)

            with for_(*from_array(t, durations)):
                for qubit in qubits:
                    qubit.xy.play(operation, duration=t)
                align()
                multiplexed_readout(qubits, I, I_st, Q, Q_st)
                wait(machine.thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(len(durations)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(durations)).buffer(len(dfs)).average().save(f"Q{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rabi_chevron, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_chevron)
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_qubits)], [])
    results = fetching_tool(job, data_list, mode="live")
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        I = fetched_data[1::2]
        Q = fetched_data[2::2]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        I_volts, Q_volts = [], []
        for i, qubit in enumerate(qubits):
            # Convert results into Volts
            I_volts.append(u.demod2volts(I[i], qubit.resonator.operations["readout"].length))
            Q_volts.append(u.demod2volts(Q[i], qubit.resonator.operations["readout"].length))
            # Plot results
            plt.suptitle("Rabi chevron")
            plt.subplot(2, num_qubits, i + 1)
            plt.cla()
            plt.pcolor(durations * 4, dfs / u.MHz, I_volts[i])
            plt.plot(qubit.xy.operations[operation].length, 0, "r*")
            plt.xlabel("Qubit pulse duration [ns]")
            plt.ylabel("Qubit detuning [MHz]")
            # plt.title(f"{qubit.name} (f_01: {int(qubit.f_01 / u.MHz)} MHz)")
            plt.subplot(2, num_qubits, i + num_qubits + 1)
            plt.cla()
            plt.pcolor(durations * 4, dfs / u.MHz, Q_volts[i])
            plt.plot(qubit.xy.operations[operation].length, 0, "r*")
            plt.xlabel("Qubit pulse duration [ns]")
            plt.ylabel("Qubit detuning [MHz]")
            plt.tight_layout()
            plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    data = {}
    for i, qubit in enumerate(qubits):
        data[f"{qubit.name}_duration"] = durations * 4
        data[f"{qubit.name}_frequency"] = dfs + qubit.xy.intermediate_frequency
        data[f"{qubit.name}_I"] = np.abs(I_volts[i])
        data[f"{qubit.name}_Q"] = np.angle(Q_volts[i])
    data["figure"] = fig
    node_save(machine, "rabi_chevron_duration", data, additional_files=True)

# %%
