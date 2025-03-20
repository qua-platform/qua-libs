"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""

# %% {Imports}
# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibrate.utils.logger_m import logger
from quam_config import QuAM
from quam_experiments.experiments.qubit_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from quam_libs.xarray_data_fetcher import XarrayDataFetcher


description = """
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a_resonator_spectroscopy.py).

State update:
    - The readout power.
    - The readout frequency for the optimal readout power. 
"""


# Be sure to include [Parameters, QuAM] so the node has proper type hinting
node = QualibrationNode[Parameters, QuAM](
    name="03a_Qubit_Spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the Qualibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, QuAM]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QuAM class from the state file
node.machine = QuAM.load()


# %% {QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    # %% {QUA_program}
    operation = node.parameters.operation  # The qubit operation to play
    n_avg = node.parameters.num_averages  # The number of averages
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_prefactor
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
        df = declare(int)  # QUA variable for the qubit frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Get the duration of the operation from the node parameters or the state
                        duration = operation_len if operation_len is not None else qubit.xy.operations[operation].length
                        # Update the qubit frequency
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        # Play the saturation pulse
                        qubit.xy.play(operation, amplitude_scale=operation_amp, duration=duration * u.ns)
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # readout the resonator
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to deplete
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, QuAM]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # print_progress_bar(job, iteration_variable="n", total_number_of_iterations=node.parameters.num_averages)
            progress_counter(data_fetcher["n"], node.parameters.num_averages, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Data_loading_and_dataset_creation}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node = node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Data_analysis}
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], logger)


# %% {Plotting}
@node.run_action(skip_if=node.parameters.simulate)
def data_plotting(node: QualibrationNode[Parameters, QuAM]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figure_amplitude"] = fig_raw_fit


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def state_update(node: QualibrationNode[Parameters, QuAM]):
    """Update the relevant parameters for each qubit only if the data analysis was a success."""

    # Update the state
    with node.record_state_updates():
        for index, q in enumerate(node.namespace["qubits"]):
            if node.results["fit_results"][q.name]["success"]:
                # Update the readout frequency for the given flux point
                q.f_01 = node.results["fit_results"][q.name]["frequency"]
                q.xy.RF_frequency = q.f_01
                # Update the integration weight angle
                q.resonator.operations["readout"].integration_weights_angle = node.results["fit_results"][q.name][
                    "iw_angle"
                ]
                # Update the saturation amplitude
                q.xy.operations["saturation"].amplitude = node.results["fit_results"][q.name]["saturation_amp"]
                # Update the x180 and x90 amplitudes
                q.xy.operations["x180"].amplitude = node.results["fit_results"][q.name]["x180_amp"]
                q.xy.operations["x900"].amplitude = node.results["fit_results"][q.name]["x180_amp"] / 2


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, QuAM]):
    node.save()


# %% {Plotting}
grid = QubitGrid(ds, [q.grid_location for q in qubits])
approx_peak = result.base_line + result.amplitude * (1 / (1 + ((ds.freq - result.position) / result.width) ** 2))
for ax, qubit in grid_iter(grid):
    # Plot the line
    (ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].I_rot * 1e3).plot(ax=ax, x="freq_GHz")
    # Identify the resonance peak
    if not np.isnan(result.sel(qubit=qubit["qubit"]).position.values):
        ax.plot(
            abs_freqs[qubit["qubit"]] / 1e9,
            ds.loc[qubit].sel(freq=result.loc[qubit].position.values, method="nearest").I_rot * 1e3,
            ".r",
        )
        # # Identify the width
        (approx_peak.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit] * 1e3).plot(
            ax=ax, x="freq_GHz", linewidth=0.5, linestyle="--"
        )
    ax.set_xlabel("Qubit freq [GHz]")
    ax.set_ylabel("Trans. amp. [mV]")
    ax.set_title(qubit["qubit"])
grid.fig.suptitle("Qubit spectroscopy (amplitude)")
plt.tight_layout()
plt.show()
node.results["figure"] = grid.fig
