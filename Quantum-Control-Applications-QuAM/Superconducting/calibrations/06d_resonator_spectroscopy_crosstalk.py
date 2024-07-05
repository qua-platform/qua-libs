# %%
"""
        RESONATOR SPECTROSCOPY AND CROSSTALK MATRIX POPULATION
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate frequencies and flux biases
for every flux element's DC offset voltage. The resonator frequency as a function of flux bias is then extracted and fitted,
allowing the parameters to be stored in the configuration.

Additionally, this program populates the crosstalk matrix with the shift in resonator frequency for each combination of resonator and flux element. The crosstalk compensation terms are then stored in each analog output's configuration.

This information can then be used to adjust the readout frequency for the maximum and minimum frequency points,
as well as to correct for crosstalk in the system.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from quam.components.ports import LFFEMAnalogOutputPort

from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close, Fit
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM, FluxLine, TunableCoupler
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
resonators = [qubit.resonator for qubit in machine.active_qubits]
couplers = [qubit_pair.coupler for qubit_pair in machine.active_qubit_pairs]
num_qubits = len(qubits)
num_resonators = len(resonators)
num_couplers = len(couplers)

elements_by_port = []
for flux_element in [qubit.z for qubit in qubits] + [coupler.z for coupler in couplers]:
    port: LFFEMAnalogOutputPort = machine.ports.get_analog_ouptut(*flux_element.opx_output)
    elements_by_port.append((port, flux_element))

flux_elements_by_port = dict(zip(elements_by_port))
port_by_flux_elements = {v: k for k, v in flux_elements_by_port.items()}

# Crosstalk matrix initialization
crosstalk_matrix = np.ones((len(flux_elements_by_port), len(flux_elements_by_port)))

###################
# The QUA program #
###################

n_avg = 20  # Number of averaging loops
# Flux bias sweep in V
dc_low = -0.5
dc_high = -0.5
# The frequency sweep around the resonator resonance frequency f_opt
dfs = np.arange(-50e6, 5e6, 0.1e6)

for port_id, flux_element in flux_elements_by_port.items():
    with program() as multi_res_spec_vs_flux:
        # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
        # For instance, here 'I' is a python list containing two QUA fixed variables.
        I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
        dc = declare(fixed)  # QUA variable for the flux bias
        df = declare(int)  # QUA variable for the readout frequency

        for i, q in enumerate(qubits):
            # Bring the active qubits to the minimum frequency point
            # todo: bring to the "steepspot" (not sweetspot!)
            machine.apply_all_flux_to_min()
            # todo: bring to what?
            machine.apply_all_couplers_to_min()

            # resonator of the qubit
            rr = resonators[i]

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):
                    # Update the resonator frequencies for all resonators
                    update_frequency(rr.name, df + rr.intermediate_frequency)

                    with for_(*from_array(dc, [dc_low, dc_high])):
                        # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                        flux_element.set_dc_offset(dc)
                        wait(100)  # Wait for the flux to settle

                        # readout the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))

                        # wait for the resonator to relax
                        rr.wait(machine.depletion_time * u.ns)

                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])

            align(*[rr.name for rr in resonators])

        with stream_processing():
            n_st.save("n")
            for i, rr in enumerate(resonators):
                I_st[i].buffer(2).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(2).buffer(len(dfs)).average().save(f"Q{i + 1}")

    #######################
    # Simulate or execute #
    #######################
    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
        job.get_simulated_samples().con1.plot()
    else:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Calibrate the active qubits
        # machine.calibrate_octave_ports(qm)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(multi_res_spec_vs_flux)
        # Get results from QUA program
        data_list = ["n"] + sum(
            [[f"I{i + 1}", f"Q{i + 1}"] for i in range(num_resonators)], []
        )
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

            plt.suptitle("Resonator spectroscopy vs flux element")
            S_data, R_data = [], []
            for i, rr in enumerate(resonators):
                S = u.demod2volts(I[i] + 1j * Q[i], rr.operations["readout"].length)
                R = np.abs(S)
                S_data.append(S)
                R_data.append(R)
                # Plot
                plt.subplot(1, num_resonators, i + 1)
                plt.cla()
                plt.title(
                    f"{rr.name} (LO: {rr.frequency_converter_up.LO_frequency / u.MHz} MHz)"
                )
                plt.xlabel("Flux Bias [V]")
                plt.ylabel(f"{rr.name} IF [MHz]")
                plt.plot(rr.intermediate_frequency / u.MHz + dfs / u.MHz, R[0])
                plt.plot(rr.intermediate_frequency / u.MHz + dfs / u.MHz, R[1])
                # plt.plot(coupler.z.min_offset, rr.intermediate_frequency / u.MHz, "r*")

            plt.tight_layout()
            plt.pause(0.1)
        plt.show()

        # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat up
        qm.close()

        # Update machine with max frequency point for both resonator and qubit
        # q1.z.min_offset =
        # q2.z.min_offset =
        # Save data from the node
        data = {}
        for i, (qubit, rr) in enumerate(zip(qubits, resonators)):
            data[f"{rr.name}_flux_element_bias_{flux_element.name}"] = [dc_low, dc_high]
            data[f"{rr.name}_frequency"] = qubit.resonator.intermediate_frequency + dfs
            data[f"{rr.name}_S_{flux_element.name}"] = S_data[i]
            data[f"{rr.name}_R_{flux_element.name}"] = R_data[i]
            # data[f"{rr.name}_min_offset"] = qubit.z.min_offset

            try:
                resonator_frequencies = []
                for j, dc in enumerate([dc_low, dc_high]):
                    # Extract the resonator frequency shift for each combination
                    fit = Fit()
                    res_spec_fit = fit.reflection_resonator_spectroscopy(
                        (qubit.resonator.intermediate_frequency + dfs) / u.MHz, R_data[i][j], plot=False
                    )
                    resonator_frequencies = res_spec_fit["f"][0] * u.MHz

                # resonator frequency shift per unit volt
                resonator_frequency_shift = (resonator_frequencies[1] - resonator_frequencies[0]) / (dc_high - dc_low)

            except:
                resonator_frequency_shift = 0

            data[f"{rr.name}_frequency_shift_{flux_element.name}"] = resonator_frequency_shift
            crosstalk_matrix[i, port_id] = resonator_frequency_shift

        data["figure"] = fig
        node_save(machine, f"resonator_spectroscopy_vs_flux_element_{port_id}", data, additional_files=True)

# Calculate the inverse of the crosstalk matrix
crosstalk_matrix_inverse = np.linalg.pinv(crosstalk_matrix)

# Save or print the crosstalk matrix and its inverse
print("Crosstalk Matrix:")
print(crosstalk_matrix)
print("Inverse Crosstalk Matrix:")
print(crosstalk_matrix_inverse)

for port, flux_element in flux_elements_by_port.items():
    for i, resonator in enumerate(resonators):
        # todo: divide by diagonal element?
        crosstalk_matrix_inverse_term = crosstalk_matrix_inverse[port.port_id, i]
        port_by_flux_elements[port].crosstalk[resonator] = crosstalk_matrix_inverse_term

# You can also save these matrices if needed, for example:
# np.save("crosstalk_matrix.npy", crosstalk_matrix)
# np.save("crosstalk_matrix_inverse.npy", crosstalk_matrix_inverse)

# %%
