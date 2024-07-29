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


q4 = machine.qubits["q4"]
q5 = machine.qubits["q5"]
coupler = (q4 @ q5).coupler
target_flux_elements = [q4.z, coupler, q5.z]
target_qubit = q4

###################
# The QUA program #
###################
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
operation = "x180"
cooldown_time = machine.thermalization_time

flux_offsets = np.linspace(-0.1, 0.1, 51)
compensation_scales = np.linspace(-1, 0, 51)
n_avg = 200  # Number of averaging loops
# intermediate_frequency =
dc_bias = 0.02

with program() as multi_qubit_spec_vs_flux:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=len(target_flux_elements))
    compensation_scale = declare(fixed)  # QUA variable for the flux bias
    flux = declare(int)  # QUA variable for the readout frequency

    machine.apply_all_flux_to_min()

    target_qubit.xy.update_frequency(intermediate_frequency)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        for flux_element in target_flux_elements:
            with for_(*from_array(flux, flux_offsets)):
                with for_(*from_array(compensation_scale, compensation_scales)):
                    # Flux sweeping by tuning the OPX dc offset associated with the flux_line element
                    flux_element.set_dc_offset(flux)
                    target_qubit.z.set_dc_offset(dc_bias + compensation_scale * flux)
                    wait(100)  # Wait for the flux to settle
                    align()
                    # Apply saturation pulse to all qubits
                    target_qubit.xy.play(operation)

                    align()
                    machine.apply_all_flux_to_min()
                    wait(100)
                    align()

                    # QUA macro to read the state of the active resonators
                    target_qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

                    # Wait for the qubits to decay to the ground state
                    wait(cooldown_time * u.ns)
            align()

    with stream_processing():
        n_st.save("n")
        for i, q in enumerate(target_flux_elements):
            I_st[i].buffer(len(compensation_scales)).buffer(len(flux_offsets)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(compensation_scales)).buffer(len(flux_offsets)).average().save(f"Q{i + 1}")

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
    data_list = ["n"] + sum([[f"I{i + 1}", f"Q{i + 1}"] for i in range(len(target_flux_elements))], [])
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

        plt.suptitle(f"Qubit spectroscopy vs {flux_element.name}")
        S_data, R_data = [], []
        for i, q in enumerate(target_flux_elements):
            S = u.demod2volts(I[i] + 1j * Q[i], q.resonator.operations["readout"].length)
            R = np.abs(S)
            S_data.append(S)
            R_data.append(R)
            # Plot
            plt.subplot(1, len(target_flux_elements), i + 1)
            plt.cla()
            plt.title(f"{q.xy.name} (LO: {q.xy.frequency_converter_up.LO_frequency / u.MHz} MHz)")
            plt.xlabel("Flux Bias [V]")
            plt.ylabel(f"{q.xy.name} RF [MHz]")
            plt.pcolormesh(q.xy.RF_frequency + intermediate_frequency / u.MHz, R)
            # plt.plot(coupler.z.min_offset, rr.intermediate_frequency / u.MHz, "r*")

        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat up
    plt.show()
    qm.close()

node_save(machine, f"resonator_spectroscopy_crosstalk", data, additional_files=True)
