"""
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
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
n_avg = 1000

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(4, 1000, 5)

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = 1e6

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the idle time
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(t, idle_times)):
            # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
            # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
            assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
            align()
            # Strict_timing ensures that the sequence will be played without gaps
            with strict_timing_():
                q1.xy.play("x90")
                q1.xy.wait(t)
                q1.xy.frame_rotation(phi)
                q1.xy.play("x90")

                q2.xy.play("x90")
                q2.xy.wait(t)
                q2.xy.frame_rotation(phi)
                q2.xy.play("x90")

            # Align the elements to measure after playing the qubit pulse.
            align()
            # Measure the state of the resonators
            multiplexed_readout(machine, I, I_st, Q, Q_st)
            # Wait for the qubits to decay to the ground state
            wait(machine.get_thermalization_time * u.ns)
            # Reset the frame of the qubits in order not to accumulate rotations
            reset_frame(q1.xy.name, q2.xy.name)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(idle_times)).average().save("I1")
        Q_st[0].buffer(len(idle_times)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(idle_times)).average().save("I2")
        Q_st[1].buffer(len(idle_times)).average().save("Q2")


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
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
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
        plt.suptitle("Ramsey")
        plt.subplot(221)
        plt.cla()
        plt.plot(4 * idle_times, I1)
        plt.ylabel("I [V]")
        plt.title(f"{q1.name}")
        plt.subplot(223)
        plt.cla()
        plt.plot(4 * idle_times, Q1)
        plt.title(f"{q1.name}")
        plt.xlabel("Idle time [ns]")
        plt.ylabel("Q [V]")
        plt.subplot(222)
        plt.cla()
        plt.plot(4 * idle_times, I2)
        plt.title(f"{q2.name}")
        plt.subplot(224)
        plt.cla()
        plt.plot(4 * idle_times, Q2)
        plt.title(f"{q2.name}")
        plt.xlabel("Idle time [ns]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {
        f"{q1.name}_amplitude": 4 * idle_times,
        f"{q1.name}_I": np.abs(I1),
        f"{q1.name}_Q": np.angle(Q1),
        f"{q2.name}_amplitude": 4 * idle_times,
        f"{q2.name}_I": np.abs(I2),
        f"{q2.name}_Q": np.angle(Q2),
        "figure": fig,
    }

    # Fit data to extract the qubits frequency and T2*
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        plt.suptitle("Ramsey")
        plt.subplot(121)
        fit_I1 = fit.ramsey(4 * idle_times, I1, plot=True)
        plt.xlabel("Idle time [ns]")
        plt.ylabel("I [V]")
        plt.title(f"{q1.name}")
        plt.legend((f"T2* = {int(fit_I1['T2'][0])} ns\n df = {int(fit_I1['f'][0] * u.GHz - detuning)/u.kHz} kHz",))
        plt.subplot(122)

        # Update the state
        qubit_detuning_q1 = fit_I1["f"][0] * u.GHz - detuning
        q1.T2ramsey = int(fit_I1["T2"][0])
        q1.xy.intermediate_frequency -= qubit_detuning_q1
        data[f"{q1.name}"] = {"T2*": q1.T2ramsey, "if_01": q1.xy.intermediate_frequency, "fit_successful": True}
        print(f"Detuning to add to {q1.name}: {-qubit_detuning_q1 / u.kHz:.3f} kHz")

        fit_I2 = fit.ramsey(4 * idle_times, I2, plot=True)
        plt.xlabel("idle_times [ns]")
        plt.title(f"{q2.name}")
        plt.legend((f"T2* = {int(fit_I2['T2'][0])} ns\n df = {int(fit_I2['f'][0] * u.GHz - detuning)/u.kHz} kHz",))
        plt.tight_layout()

        # Update the state
        qubit_detuning_q2 = fit_I2["f"][0] * u.GHz - detuning
        q2.T2ramsey = int(fit_I2["T2"][0])
        q2.xy.intermediate_frequency -= qubit_detuning_q2
        data[f"{q2.name}"] = {"T2*": q2.T2ramsey, "if_01": q2.xy.intermediate_frequency, "fit_successful": True}
        print(f"Detuning to add to {q2.name}: {-qubit_detuning_q2 / u.kHz:.3f} kHz")
    except (Exception,):
        data["fit_successful"] = False
        pass

# Save data from the node
node_save("ramsey", data, machine)
