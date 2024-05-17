"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency (f_opt) in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import multiplexed_readout


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("quam")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]
rr1 = machine.active_qubits[0].resonator
rr2 = machine.active_qubits[1].resonator

###################
# The QUA program #
###################
n_avg = 100  # The number of averages

# The frequency sweep parameters with respect to the resonators resonance frequencies
dfs = np.arange(-2e6, 2e6, 0.02e6)

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(2)]
    Q_g = [declare(fixed) for _ in range(2)]
    I_e = [declare(fixed) for _ in range(2)]
    Q_e = [declare(fixed) for _ in range(2)]
    DI = declare(fixed)
    DQ = declare(fixed)
    D = [declare(fixed) for _ in range(2)]
    df = declare(int)
    D_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the resonator frequencies
            update_frequency(rr1.name, df + rr1.intermediate_frequency)
            update_frequency(rr2.name, df + rr2.intermediate_frequency)

            # Wait for the qubit to decay to the ground state
            wait(machine.get_thermalization_time * u.ns)
            align()
            # Measure the state of the resonators
            multiplexed_readout(machine, I_g, None, Q_g, None)

            align()
            # Wait for thermalization again in case of measurement induced transitions
            wait(machine.get_thermalization_time * u.ns)
            # Play the x180 gate to put the qubits in the excited state
            q1.xy.play("x180")
            q2.xy.play("x180")
            # Align the elements to measure after playing the qubit pulses.
            align()
            # Measure the state of the resonator
            multiplexed_readout(machine, I_e, None, Q_e, None)

            # Derive the distance between the blobs for |g> and |e>
            for i in range(len(machine.active_qubits)):
                assign(DI, (I_e[i] - I_g[i]) * 100)
                assign(DQ, (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI * DI + DQ * DQ)
                save(D[i], D_st[i])

    with stream_processing():
        for i in range(len(machine.active_qubits)):
            D_st[i].buffer(len(dfs)).average().save(f"D{i+1}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_freq_opt)
    # Get results from QUA program
    results = fetching_tool(job, ["D1", "D2"])
    # fetch data
    D1, D2 = results.fetch_all()
    # Plot the results
    plt.subplot(211)
    plt.plot(dfs, D1)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{q1.name} - f_opt = {int(rr1.f_01 / u.MHz)} MHz")
    plt.subplot(212)
    plt.plot(dfs, D2)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{q2.name} - f_opt = {int(rr2.f_01 / u.MHz)} MHz")
    plt.tight_layout()
    print(f"{rr1.name}: Shift readout frequency by {dfs[np.argmax(D1)]} Hz")
    print(f"{rr2.name}: Shift readout frequency by {dfs[np.argmax(D2)]} Hz")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    rr1.intermediate_frequency += dfs[np.argmax(D1)]
    rr2.intermediate_frequency += dfs[np.argmax(D2)]
    # machine.save("quam")
