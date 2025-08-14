# %% {Imports}
import warnings
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode

from calibration_utils.crosstalk_spectroscopy_vs_flux.program import get_target_slope_from_parameter_ranges, \
    get_qubit_flux_detuning_for_target_slope, get_expected_frequency_at_flux_detuning
from quam_config import Quam
from calibration_utils.crosstalk_spectroscopy_vs_flux import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
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
    name="18_crosstalk_spectroscopy_vs_flux",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q3"]
    node.parameters.multiplexed = False
    node.parameters.frequency_step_in_mhz = 0.2
    node.parameters.frequency_span_in_mhz = 5
    node.parameters.num_flux_points = 19
    node.parameters.flux_offset_span_in_v = 0.2
    node.parameters.num_shots = 1000
    node.parameters.qubits = ["q1", "q2"]
    node.parameters.use_state_discrimination = True
    assert node.parameters.multiplexed == False
    pass


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
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    # Flux bias sweep in V
    span = node.parameters.flux_offset_span_in_v * u.V
    num = node.parameters.num_flux_points
    dcs = np.linspace(-span / 2, +span / 2, num)

    xy_vs_z_delay_padding = 2000  # ns
    target_slope = get_target_slope_from_parameter_ranges(node.parameters)

    # Setting the flux point of the qubits
    for qubit in qubits:
        qubit.z.flux_point = "joint"

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(target_qubits.get_names()),
        "aggressor": xr.DataArray(aggressor_qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "flux bias", "units": "V"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency
        dc = declare(fixed)  # QUA variable for the flux dc level

        for i, target_qubit in enumerate(target_qubits):
            for j, aggressor_qubit in enumerate(aggressor_qubits):
                # Initialize the QPU in terms of flux points (flux-tunable transmons and/or tunable couplers)
                for qubit in qubits:
                    node.machine.initialize_qpu(target=qubit, flux_point=qubit.z.flux_point)

                # set target qubit to just-right flux-sensitive point
                flux_detuning = get_qubit_flux_detuning_for_target_slope(
                    target_qubit=target_qubit,
                    target_slope_in_hz_per_v=target_slope,
                    expected_crosstalk=1 if i==j else node.parameters.expected_crosstalk,
                )
                if target_qubit.z.flux_point == "joint":
                    set_dc_offset(target_qubit.z.name, "single", target_qubit.z.joint_offset + flux_detuning)
                else:
                    raise NotImplementedError()

                # set target qubit frequency to expected frequency at flux-sensitive point
                expected_frequency = get_expected_frequency_at_flux_detuning(target_qubit, flux_detuning)
                expected_frequency_offset = expected_frequency - target_qubit.xy.RF_frequency
                # print("target: " + target_qubit.name)
                # print("aggressor: " + aggressor_qubit.name)
                # print(f"intermediate freq offset: {expected_frequency_offset/1e6} MHz")
                # print(f"flux_detuning: {flux_detuning} V")
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
                                duration=duration + 2 * xy_vs_z_delay_padding
                            )
                            # add some padding in case xy vs z delay is wrong
                            wait(xy_vs_z_delay_padding //4, target_qubit.xy.name)
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
    # node.namespace["qubits"] = get_qubits(node)
    # ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    # ds_processed.IQ_abs.plot()


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

    for target_qubit in node.namespace["target_qubits"]:
        for aggressor_qubit in node.namespace["aggressor_qubits"]:
            plt.figure()
            node.results["ds_raw"].sel(qubit=target_qubit.name).sel(aggressor=aggressor_qubit.name).IQ_abs.plot()
            plt.show()
    
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            else:
                fit_results = node.results["fit_results"][q.name]
                if q.z.flux_point == "independent":
                    q.z.independent_offset = fit_results["idle_offset"]
                elif q.z.flux_point == "joint":
                    q.z.joint_offset += fit_results["idle_offset"]
                q.xy.RF_frequency = fit_results["qubit_frequency"]
                q.f_01 = fit_results["qubit_frequency"]
                # q.freq_vs_flux_01_quad_term = fit_results["quad_term"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
