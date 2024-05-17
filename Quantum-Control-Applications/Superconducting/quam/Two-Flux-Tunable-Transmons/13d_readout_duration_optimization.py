"""
        READOUT OPTIMISATION: DURATION
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). The "demod.accumulated" method is employed to assess the state of the resonator over varying durations.
Reference: (https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#accumulated-demodulation)
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to determine
the Signal-to-Noise Ratio (SNR). The readout duration that offers the highest SNR is identified as the optimal choice.
Note: To observe the resonator's behavior during ringdown, the integration weights length should exceed the readout_pulse length.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit in focus (referred to as "resonator_spectroscopy").
    - Calibration of the qubit pi pulse (x180) using methods like qubit spectroscopy, rabi_chevron, and power_rabi,
      followed by an update to the state.
    - Calibration of both the readout frequency and amplitude, with subsequent state updates.
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the readout duration setting, labeled as "readout_len", in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import node_save


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
rr1 = q1.resonator
rr2 = q2.resonator
# Select the resonator and qubit to measure (no multiplexing here)
rr = rr2
qb = q2


####################
# Helper functions #
####################
# def update_readout_length(qubit, new_readout_length, ringdown_length):
#     config["pulses"][f"readout_pulse_{qubit}"]["length"] = new_readout_length
#     config["integration_weights"][f"cosine_weights_{qubit}"] = {
#         "cosine": [(1.0, new_readout_length + ringdown_length)],
#         "sine": [(0.0, new_readout_length + ringdown_length)],
#     }
#     config["integration_weights"][f"sine_weights_{qubit}"] = {
#         "cosine": [(0.0, new_readout_length + ringdown_length)],
#         "sine": [(1.0, new_readout_length + ringdown_length)],
#     }
#     config["integration_weights"][f"minus_sine_weights_{qubit}"] = {
#         "cosine": [(0.0, new_readout_length + ringdown_length)],
#         "sine": [(-1.0, new_readout_length + ringdown_length)],
#     }


###################
# The QUA program #
###################

n_avg = 1e4  # number of averages
# Set maximum readout duration for this scan and update the configuration accordingly
readout_len = 7 * u.us
ringdown_len = 0 * u.us
rr1.operations["readout"].length = readout_len
rr2.operations["readout"].length = readout_len
config = machine.generate_config()
# update_readout_length(q1.name, readout_len, ringdown_len)
# update_readout_length(q2.name, readout_len, ringdown_len)
# Set the accumulated demod parameters
division_length = 10  # size of one demodulation slice in clock cycles
number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)


with program() as ro_duration_opt:
    n = declare(int)
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)
    I = declare(fixed, size=number_of_divisions)
    Q = declare(fixed, size=number_of_divisions)
    ind = declare(int)

    n_st = declare_stream()
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state.
        # Play on the second resonator to be in the same conditions as with multiplexed readout
        if rr == rr1:
            measure("readout", rr2.name, None)
        else:
            measure("readout", rr1.name, None)
        # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
        rr.measure_accumulated("readout", segment_length=division_length, qua_vars=(II, IQ, QI, QQ))
        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ig_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qg_st)
        # Wait for the qubit to decay to the ground state
        wait(machine.get_thermalization_time * u.ns, rr.name)

        align()

        # Measure the excited state.
        qb.xy.play("x180")
        align()
        # Play on the second resonator to be in the same conditions as with multiplexed readout
        if rr == rr1:
            measure("readout", rr2.name, None)
        else:
            measure("readout", rr1.name, None)
        # With demod.accumulated, the results are QUA vectors with 1 point for each accumulated chunk
        rr.measure_accumulated("readout", segment_length=division_length, qua_vars=(II, IQ, QI, QQ))
        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ie_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qe_st)

        # Wait for the qubit to decay to the ground state
        wait(machine.get_thermalization_time * u.ns, rr.name)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # mean values
        Ig_st.buffer(number_of_divisions).average().save("Ig_avg")
        Qg_st.buffer(number_of_divisions).average().save("Qg_avg")
        Ie_st.buffer(number_of_divisions).average().save("Ie_avg")
        Qe_st.buffer(number_of_divisions).average().save("Qe_avg")
        # variances
        (
            ((Ig_st.buffer(number_of_divisions) * Ig_st.buffer(number_of_divisions)).average())
            - (Ig_st.buffer(number_of_divisions).average() * Ig_st.buffer(number_of_divisions).average())
        ).save("Ig_var")
        (
            ((Qg_st.buffer(number_of_divisions) * Qg_st.buffer(number_of_divisions)).average())
            - (Qg_st.buffer(number_of_divisions).average() * Qg_st.buffer(number_of_divisions).average())
        ).save("Qg_var")
        (
            ((Ie_st.buffer(number_of_divisions) * Ie_st.buffer(number_of_divisions)).average())
            - (Ie_st.buffer(number_of_divisions).average() * Ie_st.buffer(number_of_divisions).average())
        ).save("Ie_var")
        (
            ((Qe_st.buffer(number_of_divisions) * Qe_st.buffer(number_of_divisions)).average())
            - (Qe_st.buffer(number_of_divisions).average() * Qe_st.buffer(number_of_divisions).average())
        ).save("Qe_var")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ro_duration_opt, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_duration_opt)
    # Get results from QUA program
    results = fetching_tool(
        job,
        data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
        mode="live",
    )
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Derive the SNR
        ground_trace = Ig_avg + 1j * Qg_avg
        excited_trace = Ie_avg + 1j * Qe_avg
        var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4
        SNR = (np.abs(excited_trace - ground_trace) ** 2) / (2 * var)
        # Plot results
        plt.subplot(221)
        plt.cla()
        plt.plot(x_plot, ground_trace.real, label="ground")
        plt.plot(x_plot, excited_trace.real, label="excited")
        plt.xlabel("Readout duration [ns]")
        plt.ylabel("demodulated traces [V]")
        plt.title("Real part")
        plt.legend()

        plt.subplot(222)
        plt.cla()
        plt.plot(x_plot, ground_trace.imag, label="ground")
        plt.plot(x_plot, excited_trace.imag, label="excited")
        plt.xlabel("Readout duration [ns]")
        plt.title("Imaginary part")
        plt.legend()

        plt.subplot(212)
        plt.cla()
        plt.plot(x_plot, SNR, ".-")
        plt.xlabel("Readout duration [ns]")
        plt.ylabel("SNR")
        plt.title("SNR")
        plt.pause(0.1)
        plt.tight_layout()

    # Get the optimal readout length in ns
    opt_readout_length = int(np.round(np.argmax(SNR) * division_length / 4) * 4 * 4)
    print(f"The optimal readout length is {opt_readout_length} ns (SNR={max(SNR)})")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    rr.operations["readout"].length = opt_readout_length

    # Save data from the node
    data = {
        f"{rr.name}_time": x_plot,
        f"{rr.name}_SNR": SNR,
        f"{rr.name}_opt_readout_length": opt_readout_length,
        "figure": fig,
    }
    node_save("readout_duration_optimization", data, machine)
