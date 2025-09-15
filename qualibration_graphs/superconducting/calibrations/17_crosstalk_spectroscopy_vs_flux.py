# %% {Imports}
import warnings

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualang_tools.loops import from_array

from qualibrate import QualibrationNode
from quam_config import Quam

from calibration_utils.crosstalk_spectroscopy_vs_flux.program import (
    get_expected_frequency_at_flux_detuning,
    get_flux_detuning_in_v
)
from calibration_utils.crosstalk_spectroscopy_vs_flux import (
    Parameters, process_raw_dataset, fit_raw_data, log_fitted_results
)
from calibration_utils.crosstalk_spectroscopy_vs_flux.plotting import plot_analysis, add_node_info_subtitle

from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}

description = """
Qubit Spectroscopy for Crosstalk Calibration
This experiment performs qubit spectroscopy while sweeping the flux bias of a neighboring qubit or tunable coupler,
in order to map the target qubit’s frequency response as a function of the other element’s flux bias.
The resulting frequency–flux map is used to extract and compensate for flux crosstalk.

Purpose:
    - Measure the dependence of the target qubit’s f_01 on a neighboring qubit or coupler’s flux bias.
    - Determine the crosstalk slope (∂f_target/∂Φ_neighbor) for building a crosstalk compensation matrix.
    - Verify and refine flux bias settings to isolate control channels.

Prerequisites:
    - XY vs. Z channel delay correctly calibrated.
    - Mixer or Octave calibration completed (nodes 01a or 01b).
    - Readout parameters calibrated (nodes 02a, 02b, and/or 02c).
    - Target qubit frequency calibrated at its nominal flux point (03a_qubit_spectroscopy.py).
    - Flux operating points defined for both the target and the neighboring element
      (e.g., qubit.z.flux_point and coupler.z.flux_point).

State Update:
    - Measured f_01 of the target qubit vs. neighbor flux bias.
    - Extracted crosstalk coefficients for compensation.
    - Updated flux bias offsets for independent or joint control: q.z.independent_offset or q.z.joint_offset.
"""


node = QualibrationNode[Parameters, Quam](
    name="17_crosstalk_spectroscopy_vs_flux",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qC1", "qC4"]
    node.parameters.multiplexed = False
    node.parameters.frequency_num_points = 41
    node.parameters.frequency_span_in_mhz = 2
    node.parameters.flux_num_points = 19
    node.parameters.flux_span_in_v = 0.4
    node.parameters.num_shots = 200
    node.parameters.expected_crosstalk = 0.003
    node.parameters.flux_detuning_mode = "auto_for_linear_response"
    node.parameters.simulate = True
    assert node.parameters.multiplexed == False


# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["target_qubits"] = target_qubits = get_qubits(node)
    node.namespace["aggressor_qubits"] = aggressor_qubits = get_qubits(node)
    node.namespace["qubits"] = qubits = set(target_qubits._items + aggressor_qubits._items)
    # Check if the qubits have a z-line attached
    if any([q.z is None for q in qubits]):
        warnings.warn("Found qubits without a flux line. Skipping")

    operation = node.parameters.operation  # The qubit operation to play
    n_avg = node.parameters.num_shots
    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor

    flux_pulse_padding = node.parameters.flux_pulse_padding_in_ns

    # Setting the flux point of the qubits
    for qubit in qubits:
        qubit.z.flux_point = "joint"

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency
        dc = declare(fixed)  # QUA variable for the flux dc level

        # Qubit detuning sweep with respect to their resonance frequencies
        freq_span = node.parameters.frequency_span_in_mhz * u.MHz
        num_points = node.parameters.frequency_num_points
        dfs = np.linspace(-freq_span / 2, freq_span / 2, num_points)

        # Flux bias sweep in V
        flux_span = node.parameters.flux_span_in_v * u.V
        num_points = node.parameters.flux_num_points
        dcs = np.linspace(-flux_span / 2, +flux_span / 2, num_points)

        # Target qubit flux detunings
        node.namespace["flux_detunings"] = {
            q.name: get_flux_detuning_in_v(node.parameters, q) for q in target_qubits
        }

        for i, target_qubit in enumerate(target_qubits):
            flux_detuning = node.namespace["flux_detunings"][target_qubit.name]
            # set target qubit frequency to expected frequency at flux-sensitive point
            expected_frequency = get_expected_frequency_at_flux_detuning(target_qubit, flux_detuning)
            expected_frequency_offset = expected_frequency - target_qubit.xy.RF_frequency

            for j, aggressor_qubit in enumerate(aggressor_qubits):
                # Initialize the QPU in terms of flux points (flux-tunable transmons and/or tunable couplers)
                for qubit in qubits:
                    node.machine.initialize_qpu(target=qubit, flux_point=qubit.z.flux_point)

                if target_qubit.name == aggressor_qubit.name: continue

                if target_qubit.z.flux_point == "joint":
                    set_dc_offset(target_qubit.z.name, "single", target_qubit.z.joint_offset + flux_detuning)
                else:
                    raise NotImplementedError()

                assert abs(target_qubit.xy.intermediate_frequency + expected_frequency_offset) < 500e6

                target_qubit.align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs)):
                        with for_(*from_array(dc, dcs)):
                            # Update the qubit frequency
                            target_qubit.xy.update_frequency(df + target_qubit.xy.intermediate_frequency + expected_frequency_offset)
                            # Wait for the qubits to decay to the ground state
                            target_qubit.reset_qubit_thermal()
                            # Flux sweeping for a qubit
                            duration = (
                                operation_len * u.ns
                                if operation_len is not None
                                else target_qubit.xy.operations[operation].length * u.ns
                            )
                            align(target_qubit.xy.name, target_qubit.z.name,
                                  aggressor_qubit.xy.name, aggressor_qubit.z.name)

                            # Bring the aggresor qubit flux to the desired point during the saturation pulse
                            aggressor_qubit.z.play(
                                "const",
                                amplitude_scale=dc / aggressor_qubit.z.operations["const"].amplitude,
                                duration=duration + 2 * (flux_pulse_padding // 4)
                            )
                            # add some padding in case xy vs z delay is wrong
                            wait(flux_pulse_padding // 4, target_qubit.xy.name)
                            # Apply saturation pulse to all qubits
                            target_qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration,
                            )
                            target_qubit.align()

                            # Qubit readout
                            target_qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                align(*(
                    [q.xy.name for q in qubits] +
                    [q.z.name for q in qubits] +
                    [q.resonator.name for q in qubits]
                ))

        with stream_processing():
            n_st.save("n")
            for i, target_qubit in enumerate(target_qubits):
                I_st[i].buffer(len(dcs)).buffer(len(dfs)).buffer(len(aggressor_qubits)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dcs)).buffer(len(dfs)).buffer(len(aggressor_qubits)).average().save(f"Q{i + 1}")

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(target_qubits.get_names()),
        "aggressor": xr.DataArray(aggressor_qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "flux bias", "units": "V"}),
    }


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
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
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = node.namespace["target_qubits"] = node.namespace["aggressor_qubits"] = get_qubits(node)
    # ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    # ds_processed.IQ_abs.plot()


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: v for k, v in fit_results.items()}
    if "flux_detunings" in node.namespace:
        node.results["flux_detunings"] = node.namespace["flux_detunings"]
    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # node.outcomes = {
    #     pair_key: ("successful" if fit_result.success else "failed")
    #     for pair_key, fit_result in node.results["fit_results"].items()
    # }

# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    # Plot raw data
    fig = plot_analysis(
        node.results["ds_raw"], node.results["peak_results"], node.results["fit_results"],
        node.results.get("flux_detunings"), node.machine.qubits
    )
    add_node_info_subtitle(node, fig)

    node.results["figures"] = {"main": fig}

    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for target_qubit_name, target_qubit_results in node.results["fit_results"].items():
            for aggressor_qubit_name, fit_result in target_qubit_results.items():
                if target_qubit_name == aggressor_qubit_name:
                    continue

                # Find the qubit objects
                target_qubit = node.machine.qubits[target_qubit_name]
                aggressor_qubit = node.machine.qubits[aggressor_qubit_name]

                target_output = target_qubit.z.opx_output
                aggressor_output = aggressor_qubit.z.opx_output

                # Update crosstalk coefficients
                if target_output.fem_id == aggressor_output.fem_id and target_output.controller_id == aggressor_output.controller_id:
                    if not target_output.crosstalk:
                        target_output.crosstalk = {}
                    if not aggressor_output.port_id in target_output.crosstalk or np.isnan(target_output.crosstalk[aggressor_output.port_id]):
                        target_output.crosstalk[aggressor_output.port_id] = 0
                    target_output.crosstalk[aggressor_output.port_id] += fit_result["crosstalk_coefficient"]

                else:
                    node.log(f"Couldn't compensate crosstalk between {target_qubit.name} and {aggressor_qubit.name}, "
                             f"since they are on different fems ({target_output.controller_id, target_output.fem_id} and "
                             f"{aggressor_output.controller_id, aggressor_output.fem_id}) respectively.")

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    # Xarray can't yet serialize multi-index dimension "pair", so need to reset before saving
    node.results['peak_results'] = node.results['peak_results'].reset_index('pair')
    node.save()
