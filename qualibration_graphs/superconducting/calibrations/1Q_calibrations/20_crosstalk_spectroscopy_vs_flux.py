# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.crosstalk_spectroscopy_vs_flux import (
    Parameters,
    add_node_info_subtitle,
    build_crosstalk_pairs,
    fit_raw_data,
    get_expected_frequency_at_flux_detuning,
    get_flux_detuning_in_v,
    log_fitted_results,
    plot_analysis,
    plot_crosstalk_matrix,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
Qubit Spectroscopy for Crosstalk Calibration
This experiment performs qubit spectroscopy while sweeping the flux bias of a neighboring qubit or tunable coupler,
in order to map the target qubit's frequency response as a function of the other element's flux bias.
The resulting frequency-flux map is used to extract and compensate for flux crosstalk.

Purpose:
    - Measure the dependence of the target qubit's f_01 on a neighboring qubit or coupler's flux bias.
    - Determine the crosstalk slope (df_target/dPhi_neighbor) for building a crosstalk compensation matrix.
    - Verify and refine flux bias settings to isolate control channels.

Measurement schedule:
    1. Optional: serial (T, T) self-calibration per target when measure_self=True.
    2. Cross-talk: for each aggressor, 2D spectroscopy vs aggressor flux on all targets
       (parallel if multiplexed=True, serial if multiplexed=False).

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
    name="20_crosstalk_spectroscopy_vs_flux",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    node.parameters.target_qubits = ["qA2",]
    node.parameters.aggressor_qubits = ["qA1"]
    node.parameters.measure_self = False
    node.parameters.operation_amplitude_factor = 1.0
    node.parameters.frequency_num_points = 51
    node.parameters.target_qubit_frequency_span = 50
    node.parameters.aggressor_qubit_frequency_span = 20
    node.parameters.flux_detuning_mode = "manual"
    node.parameters.manual_flux_detuning_in_v = -0.01
    node.parameters.reset_type = "thermal"
    node.parameters.multiplexed = True
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):  # pylint: disable=too-many-statements
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    u = unit(coerce_to_integer=True)

    node.parameters.qubits = list(node.parameters.target_qubits or [])
    node.namespace["target_qubits"] = target_qubits = [
        node.machine.qubits[q] for q in node.parameters.qubits
    ]
    node.namespace["pairs_by_target"], _ = build_crosstalk_pairs(node)
    pairs_by_target = node.namespace["pairs_by_target"]
    n_pairs_by_target = {name: len(pairs) for name, pairs in pairs_by_target.items()}
    node.namespace["n_pairs_by_target"] = n_pairs_by_target
    if not node.parameters.measure_self and len(set(n_pairs_by_target.values())) > 1:
        raise ValueError("All targets must have the same number of measurement pairs.")

    node.namespace["qubits"] = targets = get_qubits(node)
    num_target_qubits = len(targets)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    operation_len = node.parameters.operation_len_in_ns
    operation_amp = node.parameters.operation_amplitude_factor or 1.0
    flux_pulse_padding = node.parameters.flux_pulse_padding_in_ns
    target_qubit_frequency_span = node.parameters.target_qubit_frequency_span * u.MHz
    aggressor_qubit_frequency_span = node.parameters.aggressor_qubit_frequency_span * u.MHz
    frequency_num_points = int(node.parameters.frequency_num_points)
    flux_num_points = int(node.parameters.flux_num_points)
    target_flux_span = node.parameters.target_flux_offset_span_in_v * u.V
    aggressor_flux_span = node.parameters.aggressor_flux_offset_span_in_v * u.V
    dfs_target = np.linspace(-target_qubit_frequency_span / 2, target_qubit_frequency_span / 2, frequency_num_points)
    dfs_aggressor = np.linspace(
        -aggressor_qubit_frequency_span / 2, aggressor_qubit_frequency_span / 2, frequency_num_points
    )
    dcs_target = np.linspace(-target_flux_span / 2, target_flux_span / 2, flux_num_points)
    dcs_aggressor = np.linspace(-aggressor_flux_span / 2, aggressor_flux_span / 2, flux_num_points)
    node.namespace["self_sweep_grids"] = {
        "detuning": dfs_target,
        "flux_bias": dcs_target,
    }

    flux_detunings = {q.name: get_flux_detuning_in_v(node.parameters, q) for q in target_qubits}
    expected_frequency_offsets = {}
    for target_qubit in target_qubits:
        flux_detuning = flux_detunings[target_qubit.name]
        expected_frequency = get_expected_frequency_at_flux_detuning(target_qubit, flux_detuning)
        expected_frequency_offsets[target_qubit.name] = expected_frequency - target_qubit.xy.RF_frequency

    node.namespace["flux_detunings"] = flux_detunings
    node.namespace["expected_frequency_offsets"] = expected_frequency_offsets

    aggressor_qubits = [node.machine.qubits[name] for name in (node.parameters.aggressor_qubits or [])]

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray([q.name for q in target_qubits]),
        "pair": xr.DataArray(np.arange(max(n_pairs_by_target.values())), dims="pair"),
        "detuning": xr.DataArray(dfs_aggressor, attrs={"long_name": "qubit frequency detuning", "units": "Hz"}),
        "flux_bias": xr.DataArray(dcs_aggressor, attrs={"long_name": "aggressor flux bias", "units": "V"}),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_target_qubits)
        df = declare(int)
        dc = declare(fixed)

        # Step 1: serial self-calibration (T, T) — one target at a time, own Z flux sweep
        if node.parameters.measure_self:
            for i, target_qubit in enumerate(targets):
                flux_detuning = flux_detunings[target_qubit.name]
                expected_frequency_offset = expected_frequency_offsets[target_qubit.name]
                duration = (
                    operation_len * u.ns
                    if operation_len is not None
                    else target_qubit.xy.operations[operation].length * u.ns
                )

                node.machine.initialize_qpu(target=target_qubit)
                align()
                set_dc_offset(
                    target_qubit.z.name,
                    "single",
                    target_qubit.z.independent_offset + flux_detuning,
                )
                target_qubit.z.settle()
                target_qubit.align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs_target)):
                        with for_(*from_array(dc, dcs_target)):
                            target_qubit.xy.update_frequency(
                                df + target_qubit.xy.intermediate_frequency + expected_frequency_offset,
                                keep_phase=True,
                            )
                            target_qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align(target_qubit.xy.name, target_qubit.z.name)

                            target_qubit.z.play(
                                "const",
                                amplitude_scale=dc / target_qubit.z.operations["const"].amplitude,
                                duration=duration + 2 * (flux_pulse_padding // 4),
                            )
                            wait(flux_pulse_padding // 4, target_qubit.xy.name)
                            target_qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration,
                            )
                            target_qubit.align()

                            target_qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        # Step 2: cross-talk (T, A) — sweep aggressor Z, spectroscopy on targets (multiplexed via targets.batch())
        for aggressor_qubit in aggressor_qubits:
            for multiplexed_qubits in targets.batch():
                duration = (
                    operation_len * u.ns
                    if operation_len is not None
                    else max(q.xy.operations[operation].length for q in multiplexed_qubits.values()) * u.ns
                )
                for qubit in dict.fromkeys(list(multiplexed_qubits.values()) + [aggressor_qubit]):
                    node.machine.initialize_qpu(target=qubit)
                align()

                for qubit in multiplexed_qubits.values():
                    flux_detuning = flux_detunings[qubit.name]
                    set_dc_offset(
                        qubit.z.name,
                        "single",
                        qubit.z.independent_offset + flux_detuning,
                    )
                    qubit.z.settle()
                align()

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(df, dfs_aggressor)):
                        with for_(*from_array(dc, dcs_aggressor)):
                            for i, qubit in multiplexed_qubits.items():
                                expected_frequency_offset = expected_frequency_offsets[qubit.name]
                                qubit.xy.update_frequency(
                                    df + qubit.xy.intermediate_frequency + expected_frequency_offset,
                                    keep_phase=True,
                                )
                                qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()

                            aggressor_qubit.z.play(
                                "const",
                                amplitude_scale=dc / aggressor_qubit.z.operations["const"].amplitude,
                                duration=duration + 2 * (flux_pulse_padding // 4),
                            )
                            for i, qubit in multiplexed_qubits.items():
                                wait(flux_pulse_padding // 4, qubit.xy.name)
                                qubit.xy.play(
                                    operation,
                                    amplitude_scale=operation_amp,
                                    duration=duration,
                                )
                            align()

                            for i, qubit in multiplexed_qubits.items():
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                if not (
                                    node.parameters.measure_self
                                    and qubit.name == aggressor_qubit.name
                                ):
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            for i, target_qubit in enumerate(target_qubits):
                n_pairs_i = n_pairs_by_target[target_qubit.name]
                I_st[i].buffer(len(dcs_aggressor)).buffer(len(dfs_aggressor)).buffer(n_avg).map(
                    FUNCTIONS.average()
                ).buffer(n_pairs_i).save(f"I{i + 1}")
                Q_st[i].buffer(len(dcs_aggressor)).buffer(len(dfs_aggressor)).buffer(n_avg).map(
                    FUNCTIONS.average()
                ).buffer(n_pairs_i).save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    _, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program, and fetch the raw dataset."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    u = unit(coerce_to_integer=True)
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id

    node.parameters.qubits = list(node.parameters.target_qubits or [])
    node.namespace["target_qubits"] = target_qubits = [
        node.machine.qubits[q] for q in node.parameters.qubits
    ]
    node.namespace["pairs_by_target"], _ = build_crosstalk_pairs(node)
    node.namespace["qubits"] = get_qubits(node)

    flux_detunings = {q.name: get_flux_detuning_in_v(node.parameters, q) for q in target_qubits}
    expected_frequency_offsets = {}
    for target_qubit in target_qubits:
        flux_detuning = flux_detunings[target_qubit.name]
        expected_frequency = get_expected_frequency_at_flux_detuning(target_qubit, flux_detuning)
        expected_frequency_offsets[target_qubit.name] = expected_frequency - target_qubit.xy.RF_frequency
    node.namespace["flux_detunings"] = flux_detunings
    node.namespace["expected_frequency_offsets"] = expected_frequency_offsets

    target_qubit_frequency_span = node.parameters.target_qubit_frequency_span * u.MHz
    frequency_num_points = int(node.parameters.frequency_num_points)
    flux_num_points = int(node.parameters.flux_num_points)
    target_flux_span = node.parameters.target_flux_offset_span_in_v * u.V
    node.namespace["self_sweep_grids"] = {
        "detuning": np.linspace(-target_qubit_frequency_span / 2, target_qubit_frequency_span / 2, frequency_num_points),
        "flux_bias": np.linspace(-target_flux_span / 2, target_flux_span / 2, flux_num_points),
    }

    if "ds_raw" not in node.results and "ds" in node.results:
        node.results["ds_raw"] = node.results["ds"]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and extract crosstalk coefficients."""
    node.results["ds_proc"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_proc"], node)
    node.results["fit_results"] = fit_results
    node.results["flux_detunings"] = node.namespace["flux_detunings"]

    log_fitted_results(
        fit_results,
        log_callable=node.log,
        aggressor_qubits=node.parameters.aggressor_qubits,
    )

    node.outcomes = {}
    for target_qubit_name, target_qubit_results in fit_results.items():
        self_ok = True
        if node.parameters.measure_self:
            self_ok = target_qubit_results.get("_self_calibration", {}).get("success", False)

        cross_talk_results = [
            result
            for aggressor_qubit_name, result in target_qubit_results.items()
            if not aggressor_qubit_name.startswith("_") and result.get("pair_type") == "cross"
        ]

        if not cross_talk_results:
            node.outcomes[target_qubit_name] = "failed"
        elif self_ok and all(result.get("success", False) for result in cross_talk_results):
            node.outcomes[target_qubit_name] = "successful"
        else:
            node.outcomes[target_qubit_name] = "failed"


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot spectroscopy maps, peak tracks, and linear fits for each qubit pair."""
    fig = plot_analysis(
        node.results["ds_proc"],
        node.results["peak_results"],
        node.results["fit_results"],
        node.results.get("flux_detunings"),
        node.machine.qubits,
    )
    add_node_info_subtitle(node, fig)
    matrix_fig = plot_crosstalk_matrix(
        node.results["fit_results"],
        aggressor_qubits=node.parameters.aggressor_qubits,
    )
    add_node_info_subtitle(node, matrix_fig)
    plt.show()
    node.results["figures"] = {"main": fig, "crosstalk_matrix": matrix_fig}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update crosstalk compensation on shared FEM outputs when analysis succeeds."""
    with node.record_state_updates():
        for target_qubit_name, target_qubit_results in node.results["fit_results"].items():
            if node.outcomes.get(target_qubit_name) == "failed":
                continue

            for aggressor_qubit_name, fit_result in target_qubit_results.items():
                if (
                    aggressor_qubit_name.startswith("_")
                    or fit_result.get("pair_type") == "self"
                    or not fit_result.get("success", False)
                ):
                    continue

                target_qubit = node.machine.qubits[target_qubit_name]
                aggressor_qubit = node.machine.qubits[aggressor_qubit_name]
                target_output = target_qubit.z.opx_output
                aggressor_output = aggressor_qubit.z.opx_output

                if (
                    target_output.fem_id == aggressor_output.fem_id
                    and target_output.controller_id == aggressor_output.controller_id
                ):
                    if not target_output.crosstalk:
                        target_output.crosstalk = {}
                    if (
                        aggressor_output.port_id not in target_output.crosstalk
                        or np.isnan(target_output.crosstalk[aggressor_output.port_id])
                    ):
                        target_output.crosstalk[aggressor_output.port_id] = 0
                    target_output.crosstalk[aggressor_output.port_id] += fit_result["crosstalk_coefficient"]
                else:
                    node.log(
                        f"Couldn't compensate crosstalk between {target_qubit.name} and {aggressor_qubit.name}, "
                        f"since they are on different fems ({target_output.controller_id, target_output.fem_id} and "
                        f"{aggressor_output.controller_id, aggressor_output.fem_id}) respectively."
                    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results to the node storage."""
    if "peak_results" in node.results:
        node.results["peak_results"] = node.results["peak_results"].reset_index("pair")
    node.save()


# %%
