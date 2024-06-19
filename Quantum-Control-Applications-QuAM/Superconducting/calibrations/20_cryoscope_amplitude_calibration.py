"""
        CRYOSCOPE AMPLITUDE
The goal of this protocol is to measure the frequency shift induced by a flux pulse of a given duration.
Since the flux line ends on the qubit chip, it is not possible to measure the flux pulse after propagation through the
fridge.

The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a fixed dephasing time.
A flux pulse with varying duration is played during the idle time. The Sx and Sy components of the Bloch vector are
measured by alternatively closing the Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing
 as a function of the flux pulse duration.

The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more details about the sequence and
the post-processing of the data.


Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - Having calibrated the IQ blobs for state discrimination.
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
import os

from quam_components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load(os.path.join('..', 'configuration', 'quam_state'))
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]


###################
# The QUA program #
###################

qb = q1
n_avg = 500

# Flux amplitude sweep (as a pre-factor of the flux amplitude) - must be within [-2; 2)
flux_amp_array = np.linspace(0, 0.45, 1001)

with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    flux_amp = declare(fixed)  # Flux amplitude pre-factor
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(flux_amp, flux_amp_array)):
            with for_each_(flag, [True, False]):
                # Play first X/2
                qb.xy.play("x90")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play the flux pulse
                qb.z.play("const", amplitude_scale=flux_amp)
                qb.xy.align(qb.z.name)
                # Wait some time to ensure that the 2nd x90 pulse will arrive after the flux pulse
                wait(20 * u.ns)
                align()
                with if_(flag):
                    qb.xy.play("x90")
                with else_():
                    qb.xy.play("y90")

                # Measure resonators state after the sequence
                align()
                multiplexed_readout([q1, q2], I, I_st, Q, Q_st)
                # State discrimination
                assign(state[0], I[0] > q1.resonator.operations["readout"].threshold)
                assign(state[1], I[1] > q2.resonator.operations["readout"].threshold)
                # Wait cooldown time and save the results
                wait(machine.thermalization_time * u.ns)
                save(state[0], state_st[0])
                save(state[1], state_st[1])

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # Qubit state
        state_st[0].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state1")
        state_st[1].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state2")
        # I_st[0].buffer(2).buffer(len(flux_amp_array)).average().save("I1")
        # I_st[1].buffer(2).buffer(len(flux_amp_array)).average().save("I2")
        # Q_st[0].buffer(2).buffer(len(flux_amp_array)).average().save("Q1")
        # Q_st[1].buffer(2).buffer(len(flux_amp_array)).average().save("Q2")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cryoscope, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cryoscope)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "state1", "state2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    xplot = flux_amp_array * qb.z.operations["const"].amplitude
    while results.is_processing():
        # Fetch results
        n, state1, state2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Get the state of the qubit under study
        if qb == q1:
            state = state1
        else:
            state = state2
        # Derive the Bloch vector components from the two projections
        Sx = state[:, 0] * 2 - 1
        Sy = state[:, 1] * 2 - 1
        qubit_state = Sx + 1j * Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[0]
        # Filtering and derivative of the phase to get the averaged frequency
        coarse_detuning = qubit_phase / (2 * np.pi * qb.z.operations["const"].length / u.s)
        # Quadratic fit of detuning versus flux pulse amplitude
        pol = np.polyfit(xplot, coarse_detuning, deg=2)

        # Plot the results
        plt.suptitle("Cryoscope")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, state1, ".-")
        plt.title(f"{q1.name}")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("State")
        plt.legend(("Sx", "Sy"))
        plt.subplot(222)
        plt.cla()
        plt.title(f"{q2.name}")
        plt.plot(xplot, state2, ".-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.legend(("Sx", "Sy"))
        plt.subplot(212)
        plt.cla()
        plt.plot(xplot, coarse_detuning / u.MHz, ".")
        plt.plot(xplot, np.polyval(pol, xplot) / u.MHz, "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [MHz]")
        plt.title(f"{qb.name}")
        plt.legend(("data", "Fit"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # TODO: store the fit parameters

    # Save data from the node
    data = {
        f"{qb.name}_time": xplot,
        f"{qb.name}_coarse_detuning": coarse_detuning,
        f"{qb.name}_fit": np.polyval(pol, xplot),
        f"{qb.name}_fit_coef": pol,
        f"{qb.name}_state": state,
        "figure": fig,
    }
    node_save("cryoscope_vs_amplitude", data, machine)
