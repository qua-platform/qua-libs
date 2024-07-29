# %%
"""
        QUBIT SPECTROSCOPY AND CROSSTALK MATRIX POPULATION
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various qubit intermediate frequencies and flux biases
for every flux element's DC offset voltage. The qubit frequency as a function of flux bias is then extracted and fitted,
used to calculate the shift caused by flux biasing.

Additionally, this program populates the crosstalk matrix with the shift in resonator frequency for each combination of
qubit and flux element. The crosstalk compensation terms are then stored in each analog output's configuration.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points,
as well as to correct for crosstalk in the system.

Before proceeding to the next node:
    - Ensure that the crosstalk matrix is saved in the machine state
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from quam.components.ports import LFFEMAnalogOutputPort

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close, Fit
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, node_save

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
resonators = [qubit.resonator for qubit in machine.active_qubits]
couplers = [qubit_pair.coupler for qubit_pair in machine.active_qubit_pairs]
num_qubits = len(qubits)
num_resonators = len(resonators)
num_couplers = len(couplers)

all_flux_elements = [qubit.z for qubit in qubits] + [coupler.z for coupler in couplers]

# Set the element you would like to sweep
target_flux_elements = all_flux_elements
target_qubits = qubits[1:]  # ignore qubit 1 since it doesn't work

flux_elements_by_port, port_by_flux_element = {}, {}
for flux_element in target_flux_elements:
    port: LFFEMAnalogOutputPort = machine.ports.get_analog_ouptut(*flux_element.opx_output)
    flux_elements_by_port[port.port_id] = flux_element
    port_by_flux_element[flux_element.name] = port

# Crosstalk matrix initialization (ordered by port id)
crosstalk_matrix = np.ones((len(all_flux_elements), len(all_flux_elements)))


###################
# The QUA program #
###################
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
operation = "saturation"
saturation_len = 10 * u.us  # in ns
saturation_amp = 0.5  # scaling factor
cooldown_time = machine.thermalization_time
dfs = np.arange(-50e6, 100e6, 0.1e6)

n_avg = 20  # Number of averaging loops
# Flux bias sweep in V
dc_low = -0.5
dc_high = -0.5

for port_id, flux_element in flux_elements_by_port.items():
    with program() as multi_qubit_spec_vs_flux:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        # For instance, here 'I' is a python list containing two QUA fixed variables.
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(target_qubits))
        dc = declare(fixed)  # QUA variable for the flux bias
        df = declare(int)  # QUA variable for the readout frequency

        for i, q in enumerate(len(target_qubits)):
            # Bring the active qubits to the minimum frequency point
            # todo: bring to the "steepspot" (not sweetspot!)
            machine.apply_all_flux_to_min()
            # todo: bring to what?
            machine.apply_all_couplers_to_min()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    update_frequency(q.xy.name, df + q.xy.intermediate_frequency)

                    with for_(*from_array(dc, [dc_low, dc_high])):
                        # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                        flux_element.set_dc_offset(dc)
                        wait(100)  # Wait for the flux to settle
                        # Apply saturation pulse to all qubits
                        q.xy.play(
                            operation,
                            amplitude_scale=saturation_amp,
                            duration=saturation_len * u.ns,
                        )
                        # QUA macro to read the state of the active resonators
                        q.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

                        # Wait for the qubits to decay to the ground state
                        wait(cooldown_time * u.ns)

        with stream_processing():
            n_st.save("n")
            for i, q in enumerate(len(target_qubits)):
                I_st[i].buffer(2).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(2).buffer(len(dfs)).average().save(f"Q{i + 1}")

    #######################
    # Simulate or execute #
    #######################
    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
        job.get_simulated_samples().con1.plot()
    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(multi_qubit_spec_vs_flux)
        # Get results from QUA program
        data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_resonators)], [])
        results = fetching_tool(job, data_list, mode="live")
        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            I = fetched_data[1::2]
            Q = fetched_data[2::2]

            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

            plt.suptitle("Qubit spectroscopy vs flux element")
            S_data, R_data = [], []
            for i, q in enumerate(target_qubits):
                S = u.demod2volts(I[i] + 1j * Q[i], q.resonator.operations["readout"].length)
                R = np.abs(S)
                S_data.append(S)
                R_data.append(R)
                # Plot
                plt.subplot(1, len(target_qubits), i + 1)
                plt.cla()
                plt.title(f"{q.xy.name} (LO: {q.xy.frequency_converter_up.LO_frequency / u.MHz} MHz)")
                plt.xlabel("Flux Bias [V]")
                plt.ylabel(f"{q.xy.name} IF [MHz]")
                plt.plot(q.xy.intermediate_frequency / u.MHz + dfs / u.MHz, R[0])
                plt.plot(q.xy.intermediate_frequency / u.MHz + dfs / u.MHz, R[1])
                # plt.plot(coupler.z.min_offset, rr.intermediate_frequency / u.MHz, "r*")

            plt.tight_layout()
            plt.pause(0.1)

        # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat up
        qm.close()

        # Update machine with max frequency point for both resonator and qubit
        # q1.z.min_offset =
        # q2.z.min_offset =
        # Save data from the node
        data = {}
        for i, (q, rr) in enumerate(zip(qubits, resonators)):
            data[f"{q.name}_flux_element_bias_{flux_element.name}"] = [dc_low, dc_high]
            data[f"{q.name}_frequency"] = q.xy.intermediate_frequency + dfs
            data[f"{q.name}_S_{flux_element.name}"] = S_data[i]
            data[f"{q.name}_R_{flux_element.name}"] = R_data[i]
            # data[f"{rr.name}_min_offset"] = qubit.z.min_offset

            try:
                measured_qubit_frequencies = []
                for j, dc in enumerate([dc_low, dc_high]):
                    # Extract the resonator frequency shift for each combination
                    fit = Fit()
                    res = fit.reflection_resonator_spectroscopy(
                        (q.xy.intermediate_frequency + dfs) / u.MHz,
                        -np.angle(S_data[i][j]),
                        plot=True,
                    )
                    measured_qubit_frequencies.append(int(res["f"][0] * u.MHz))

                    plt.subplot(1, len(target_qubits), i + 1)
                    plt.axvline(measured_qubit_frequencies[j], color="r")
                    data[f"{q.xy.name}_frequency_at_{flux_element.name}_{dc:.3f}"] = qubit_frequency_shift

                # resonator frequency shift per unit volt
                qubit_frequency_shift = (measured_qubit_frequencies[1] - measured_qubit_frequencies[0]) / (
                    dc_high - dc_low
                )
            except:
                qubit_frequency_shift = 0

            data[f"{q.xy.name}_frequency_shift_{flux_element.name}"] = qubit_frequency_shift

            x_idx = port_by_flux_element[q.z.name].port_id
            y_idx = port_by_flux_element[flux_element.name].port_id
            crosstalk_matrix[x_idx, y_idx] = qubit_frequency_shift

            plt.show()

        data["figure"] = fig
        node_save(machine, f"resonator_spectroscopy_vs_flux_element_{port_id}", data, additional_files=True)

# Calculate the inverse of the crosstalk matrix
crosstalk_matrix_inverse = np.linalg.pinv(crosstalk_matrix)
crosstalk_matrix_inverse_relative = crosstalk_matrix_inverse / crosstalk_matrix_inverse.diagonal()

# Save or print the crosstalk matrix and its inverse
print("Crosstalk Matrix:")
print(crosstalk_matrix)
print("Inverse Crosstalk Matrix:")
print(crosstalk_matrix_inverse)

for port, flux_element in flux_elements_by_port.items():
    for i, q in enumerate(target_qubits):
        # todo: divide by diagonal element?
        x_idx = port_by_flux_element[q.z.name].port_id
        y_idx = port_by_flux_element[flux_element.name].port_id
        crosstalk_term = crosstalk_matrix_inverse_relative[y_idx, x_idx]
        # Update crosstalk term
        # port_by_flux_element[flux_element.name].crosstalk[x_idx] = crosstalk_term

data = {}
data["target_qubits"] = [q.z.name for q in target_qubits]
data["flux_elements"] = [elem.name for elem in target_flux_elements]
data["flux_elements_ports"] = [port_by_flux_element[elem.name].port_id for elem in target_flux_elements]
data["crosstalk_matrix"] = crosstalk_matrix
data["crosstalk_matrix_inverse"] = crosstalk_matrix_inverse
data["crosstalk_matrix_inverse_relative"] = crosstalk_matrix_inverse_relative

node_save(machine, f"resonator_spectroscopy_crosstalk", data, additional_files=True)
