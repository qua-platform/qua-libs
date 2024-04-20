"""
        READOUT OPTIMISATION: INTEGRATION WEIGHTS
This sequence involves assessing the state of the resonator in two distinct scenarios: first, after thermalization
(with the qubit in the |g> state) and then following the application of a pi pulse to the qubit (transitioning the
qubit to the |e> state).
The "demod.sliced" method is employed to capture the time trace of the demodulated data, providing insight into the
resonator's response.
Reference: https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/features/?h=accumulated#sliced-demodulation

From the average I & Q quadratures for the qubit states |g> and |e>, along with their variances,
the Signal-to-Noise Ratio (SNR) is determined. The readout duration that yields the highest SNR is selected as
the optimal choice.
It's important to note that if you aim to observe the resonator's behavior during its ringdown phase,
the length of the integration weights should surpass that of the readout_pulse.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Having calibrated the readout frequency, amplitude and duration and updated the state.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the integration weights in the state by following the steps at the end of the script.
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


####################
# Helper functions #
####################
def divide_array_in_half(arr):
    split_index = len(arr) // 2
    arr1 = arr[:split_index]
    arr2 = arr[split_index:]
    return arr1, arr2


def normalize_complex_array(arr):
    # Calculate the simple norm of the complex array
    norm = np.sqrt(np.sum(np.abs(arr) ** 2))

    # Normalize the complex array by dividing it by the norm
    normalized_arr = arr / norm

    # Rescale the normalized array so that the maximum value is 1
    max_val = np.max(np.abs(normalized_arr))
    rescaled_arr = normalized_arr / max_val

    return rescaled_arr


def plot_three_complex_arrays(x, arr1, arr2, arr3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(x, arr1.real, label="real")
    ax1.plot(x, arr1.imag, label="imag")
    ax1.set_title("ground state")
    ax1.set_xlabel("Readout time [ns]")
    ax1.set_ylabel("demod traces [a.u.]")
    ax1.legend()
    ax2.plot(x, arr2.real, label="real")
    ax2.plot(x, arr2.imag, label="imag")
    ax2.set_title("excited state")
    ax2.set_xlabel("Readout time [ns]")
    ax2.set_ylabel("demod traces [a.u.]")
    ax2.legend()
    ax3.plot(x, arr3.real, label="real")
    ax3.plot(x, arr3.imag, label="imag")
    ax3.set_title("SNR")
    ax3.set_xlabel("Readout time [ns]")
    ax3.set_ylabel("subtracted traces [a.u.]")
    ax3.legend()
    plt.tight_layout()
    plt.show()


def update_readout_length(qubit, new_readout_length, ringdown_length):
    config["pulses"][f"readout_pulse_{qubit}"]["length"] = new_readout_length
    config["integration_weights"][f"cosine_weights_{qubit}"] = {
        "cosine": [(1.0, new_readout_length + ringdown_length)],
        "sine": [(0.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"][f"sine_weights_{qubit}"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(1.0, new_readout_length + ringdown_length)],
    }
    config["integration_weights"][f"minus_sine_weights_{qubit}"] = {
        "cosine": [(0.0, new_readout_length + ringdown_length)],
        "sine": [(-1.0, new_readout_length + ringdown_length)],
    }


###################
# The QUA program #
###################
# Select the resonator and qubit to measure (no multiplexing here)
qb = q2
rr = rr2

n_avg = 1e4  # number of averages

# Set maximum readout duration for this scan and update the configuration accordingly
readout_len = rr.operations["readout"].length
ringdown_len = 0 * u.us
rr1.operations["readout"].length = readout_len
rr2.operations["readout"].length = readout_len
config = machine.generate_config()

# Set the sliced demod parameters
division_length = 8  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len + ringdown_len) / (4 * division_length))
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + ringdown_len + 1, division_length * 4)


with program() as opt_weights:
    n = declare(int)
    ind = declare(int)
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)

    n_st = declare_stream()
    II_st = declare_stream()
    IQ_st = declare_stream()
    QI_st = declare_stream()
    QQ_st = declare_stream()

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        # Measure the ground state.
        # Play on the second resonator to be in the same conditions as with multiplexed readout
        if rr == rr1:
            measure("readout", rr2.name, None)
        else:
            measure("readout", rr1.name, None)
        # With demod.sliced, the results are QUA vectors with 1 point for each chunk
        rr.measure_sliced("readout", segment_length=division_length, qua_vars=(II, IQ, QI, QQ))

        # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)
        # Wait for the qubit to decay to the ground state
        wait(machine.get_thermalization_time * u.ns, rr.name)

        align()

        # Measure the excited state.
        qb.xy.play("x180")
        align()
        # Play on the second resonator to be in the same conditions as with multiplexed readout
        if rr == rr1:
            rr2.measure("readout")
        else:
            rr1.measure("readout")
        # With demod.sliced, the results are QUA vectors with 1 point for each chunk
        rr.measure_sliced("readout", segment_length=division_length, qua_vars=(II, IQ, QI, QQ))

        # Save the sliced data (time trace of the demodulated data with a resolution equals to the division length)
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            save(II[ind], II_st)
            save(IQ[ind], IQ_st)
            save(QI[ind], QI_st)
            save(QQ[ind], QQ_st)

        # Wait for the qubit to decay to the ground state
        wait(machine.get_thermalization_time * u.ns, rr.name)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        II_st.buffer(2 * number_of_divisions).average().save("II")
        IQ_st.buffer(2 * number_of_divisions).average().save("IQ")
        QI_st.buffer(2 * number_of_divisions).average().save("QI")
        QQ_st.buffer(2 * number_of_divisions).average().save("QQ")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, opt_weights, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(opt_weights)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()[0]
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # Fetch and reshape the data
    res_handles = job.result_handles
    IIg, IIe = divide_array_in_half(res_handles.get("II").fetch_all())
    IQg, IQe = divide_array_in_half(res_handles.get("IQ").fetch_all())
    QIg, QIe = divide_array_in_half(res_handles.get("QI").fetch_all())
    QQg, QQe = divide_array_in_half(res_handles.get("QQ").fetch_all())
    # Sum the quadrature to fully demodulate the traces
    Ie = IIe + IQe
    Ig = IIg + IQg
    Qe = QIe + QQe
    Qg = QIg + QQg
    # Derive and normalize the ground and excited traces
    ground_trace = Ig + 1j * Qg
    excited_trace = Ie + 1j * Qe
    subtracted_trace = excited_trace - ground_trace
    norm_subtracted_trace = normalize_complex_array(subtracted_trace)  # <- these are the optimal weights :)
    # Plot the results
    plot_three_complex_arrays(x_plot, ground_trace, excited_trace, norm_subtracted_trace)
    # Reshape the optimal integration weights to match the configuration
    weights_real = norm_subtracted_trace.real
    weights_imag = norm_subtracted_trace.imag

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    from quam.components.pulses import SquareReadoutPulse

    rr.operations["readout_opt"] = SquareReadoutPulse(
        length=rr.operations["readout"].length,
        amplitude=rr.operations["readout"].amplitude,
        threshold=rr.operations["readout"].threshold,
        digital_marker="ON",
        integration_weights=weights_real,
    )

    # Save data from the node
    data = {
        f"{rr.name}_time": x_plot,
        f"{rr.name}_ground_trace_real": ground_trace.real,
        f"{rr.name}_excited_trace_real": excited_trace.real,
        f"{rr.name}_norm_subtracted_trace_real": norm_subtracted_trace.real,
        f"{rr.name}_ground_trace_imag": ground_trace.imag,
        f"{rr.name}_excited_trace_imag": excited_trace.imag,
        f"{rr.name}_norm_subtracted_trace_imag": norm_subtracted_trace.imag,
        f"{rr.name}_opt_weights": weights_real,
        "figure": plt.gcf(),
    }
    node_save("readout_weights_optimization", data, machine)
