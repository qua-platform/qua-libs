# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.qubit_spectroscopy_parity_diff import (
    InitDurParameters as Parameters,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        QUBIT INITIALISATION vs DURATION TYPE
This sequence involves parking the qubit at the manipulation bias point, and playing pulses of varying frequency
to drive the qubit. When the pulse frequency is the Larmor frequency, the qubit is driven, and the parity is measured
via PSB.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.


State update:
    - The qubit frequency (and optionally the corresponding LO/IF plan) for the specified qubit operation.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08c_qubit_init_ramp_vs_hold",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["Q1", "Q2", "Q3", "Q4"]
    # node.parameters.simulate = True
    # node.parameters.qubits = ["q1"]
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # operation = node.parameters.operation  # The qubit operation to play

    # # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    # operation_len = node.parameters.operation_len_in_ns

    # # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    # operation_amp_factor = node.parameters.operation_amplitude_factor

    # for qubit in qubits:
    #     qubit.x.update(
    #         amplitude_scale=operation_amp_factor,
    #         duration=operation_len,
    #     )

    hold_durations = np.arange(node.parameters.hold_duration_start, node.parameters.hold_duration_stop, node.parameters.hold_duration_step)
    ramp_durations = np.arange(node.parameters.ramp_duration_start, node.parameters.ramp_duration_stop, node.parameters.ramp_duration_step)

    u = unit(coerce_to_integer=True)
    n_avg = node.parameters.num_shots  # The number of averages

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "hold_durations": xr.DataArray(
            hold_durations, attrs={"long_name": "Init hold duration", "units": "ns"}
        ),
        "ramp_durations": xr.DataArray(
            ramp_durations, attrs={"long_name": "Init ramp duration", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        n = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubits)

        i_st_no_pulse = {qubit.name: declare_output_stream() for qubit in qubits}
        q_st_no_pulse = {qubit.name: declare_output_stream() for qubit in qubits}
        state_st_no_pulse = {qubit.name: declare_output_stream() for qubit in qubits}

        i_st_pulse = {qubit.name: declare_output_stream() for qubit in qubits}
        q_st_pulse = {qubit.name: declare_output_stream() for qubit in qubits}
        state_st_pulse = {qubit.name: declare_output_stream() for qubit in qubits}

        n_st = declare_output_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {qubit.name: declare_output_stream() for qubit in qubits}
            if heralded_and_return_n_loops
            else {}
        )

        hold = declare(int)
        ramp = declare(int)
        state_no_int = declare(int)

        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(hold, hold_durations)):
                #for hold in hold_durations:
                    with for_(*from_array(ramp, ramp_durations)): 
                    
                        # 1: Baseline, no pulse
                        n_init = qubit.initialize(
                            hold_duration=10000,
                            ramp_duration=ramp,
                            conditional_drive=True,
                        )
                        if heralded_and_return_n_loops:
                            save(n_init, n_loops_st[qubit.name])
                        align()
                        (i_n, q_n, a2_n) = qubit.measure(return_iq=True)
                        qubit.voltage_sequence.ramp_to_zero()
                        save(i_n, i_st_no_pulse[qubit.name])
                        save(q_n, q_st_no_pulse[qubit.name])
                        assign(state_no_int, Cast.to_int(a2_n))
                        save(state_no_int, state_st_no_pulse[qubit.name])

                        # 2: With pulse
                        align()
                        qubit.initialize(
                            hold_duration=10000,
                            ramp_duration=ramp,
                            conditional_drive=True,
                        )
                        align()
                        qubit.x180()
                        align()
                        (i_p, q_p, a2_p) = qubit.measure(return_iq=True)
                        qubit.voltage_sequence.ramp_to_zero()
                        align()

                        assign(p2, Cast.to_int(a2_p))
                        save_parity_measurement(node, qubit.name, p1, p2, parity_streams)
                        save(i_p, i_st_pulse[qubit.name])
                        save(q_p, q_st_pulse[qubit.name])
                        save(p2, state_st_pulse[qubit.name])

        with stream_processing():
            n_st.save("n")
            n_hold = len(hold_durations)
            n_ramp = len(ramp_durations)

            for qubit in qubits:
                # buffer_parity_streams(node, qubit.name, parity_streams, [n_hold, n_ramp])
                i_st_no_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"I_no_pulse_{qubit.name}"
                )
                q_st_no_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"Q_no_pulse_{qubit.name}"
                )
                i_st_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"I_pulse_{qubit.name}"
                )
                q_st_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"Q_pulse_{qubit.name}"
                )
                state_st_no_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"state_no_pulse_{qubit.name}"
                )
                state_st_pulse[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                    f"state_pulse_{qubit.name}"
                )
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_ramp).buffer(n_hold).average().save(
                        f"n_loops_{qubit.name}"
                    )

# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect(timeout = node.parameters.timeout)
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Build stream-based maps and their pulse-minus-no-pulse differences."""
    ds_raw = node.results["ds_raw"]

    diff_vars = {}
    for qubit in node.namespace["qubits"]:
        qname = qubit.name
        qbase = qname.rstrip("0123456789")

        i_no = ds_raw[f"I_no_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)
        q_no = ds_raw[f"Q_no_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)
        i_p = ds_raw[f"I_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)
        q_p = ds_raw[f"Q_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)
        state_no = ds_raw[f"state_no_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)
        state_p = ds_raw[f"state_pulse_{qbase}"].sel(qubit = qname).astype(np.float64)

        i_diff = i_p - i_no
        q_diff = q_p - q_no
        iq_abs_diff = np.sqrt(i_diff**2 + q_diff**2)
        state_diff = state_p - state_no

        diff_vars[f"I_no_pulse_{qname}"] = i_no
        diff_vars[f"Q_no_pulse_{qname}"] = q_no
        diff_vars[f"state_no_pulse_{qname}"] = state_no

        diff_vars[f"I_diff_{qname}"] = i_diff
        diff_vars[f"Q_diff_{qname}"] = q_diff
        diff_vars[f"IQ_abs_diff_{qname}"] = iq_abs_diff
        diff_vars[f"state_diff_{qname}"] = state_diff

    node.results["ds_raw"] = ds_raw.assign(diff_vars)


# %% {Analysis_helpers}
def _get_analysis_signal_for_qubit(
    ds: xr.Dataset, qubit_name: str, signal_prefix: str
) -> xr.DataArray:
    """Return the selected analysis signal for one qubit."""
    by_qubit_var = f"{signal_prefix}_{qubit_name}"
    if by_qubit_var in ds:
        return ds[by_qubit_var]

    if signal_prefix in ds:
        signal = ds[signal_prefix]
        if "qubit" in signal.dims:
            return signal.sel(qubit=qubit_name)
        return signal

    raise KeyError(
        f"Could not find analysis signal '{signal_prefix}' for qubit '{qubit_name}'."
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Simple analysis: find the minimum in each qubit map."""
    signal_prefix = "I_diff"
    ds_raw = node.results["ds_raw"]
    fit_results = {}

    for qubit in node.namespace["qubits"]:
        qname = qubit.name
        signal_da = _get_analysis_signal_for_qubit(ds_raw, qname, signal_prefix)
        signal_values = np.asarray(signal_da.values, dtype=float)

        if signal_values.size == 0 or np.all(np.isnan(signal_values)):
            fit_results[qname] = {
                "success": False,
                "maximum": np.nan,
                "hold_duration": None,
                "ramp_duration": None,
            }
            node.log(f"{qname}: maximum not found (empty or NaN-only data).")
            continue

        flat_max_idx = int(np.nanargmax(signal_values))
        max_index = np.unravel_index(flat_max_idx, signal_values.shape)
        dim_to_index = {dim: int(idx) for dim, idx in zip(signal_da.dims, max_index)}

        hold_duration = None
        ramp_duration = None
        if "hold_durations" in signal_da.dims:
            hold_idx = dim_to_index["hold_durations"]
            hold_duration = float(signal_da.coords["hold_durations"].values[hold_idx])
        if "ramp_durations" in signal_da.dims:
            ramp_idx = dim_to_index["ramp_durations"]
            ramp_duration = float(signal_da.coords["ramp_durations"].values[ramp_idx])

        maximum_value = float(signal_values[max_index])
        fit_results[qname] = {
            "success": True,
            "minimum": maximum_value,
            "hold_duration": hold_duration,
            "ramp_duration": ramp_duration,
        }
        node.log(
            f"{qname}: maximum {maximum_value:.6g} at "
            f"hold={hold_duration} ns, ramp={ramp_duration} ns."
        )

    # Keep this key for compatibility with existing tooling.
    node.results["ds_fit"] = ds_raw
    node.results["fit_results"] = fit_results
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot a 2D map per qubit: 6 panels (I/Q/state × raw/diff)."""
    qubits = node.namespace["qubits"]
    ds_raw = node.results["ds_raw"]

    _PANELS = [
        ("I_no_pulse", "I, raw"),
        ("Q_no_pulse", "Q, raw"),
        ("I_diff",     "I, diff"),
        ("Q_diff",     "Q, diff"),
        ("state_no_pulse", "state, raw"),
        ("state_diff", "state, diff"),
    ]
    n_panels = len(_PANELS)
    n = len(qubits)
    fig_map, axes = plt.subplots(n_panels, n, figsize=(7 * n, 5 * n_panels), squeeze=False)

    for idx, qubit in enumerate(qubits):
        qname = qubit.name
        fit = node.results["fit_results"].get(qname, {})

        for row, (signal_prefix, panel_label) in enumerate(_PANELS):
            ax = axes[row, idx]
            signal_da = _get_analysis_signal_for_qubit(ds_raw, qname, signal_prefix)

            if {"ramp_durations", "hold_durations"}.issubset(signal_da.dims):
                map_da = signal_da.transpose("ramp_durations", "hold_durations")
                x = map_da.coords["hold_durations"].values
                y = map_da.coords["ramp_durations"].values
                image = ax.pcolormesh(x, y, map_da.values, shading="auto", cmap="viridis")
                ax.set_xlabel("Hold duration [ns]")
                ax.set_ylabel("Ramp duration [ns]")
            else:
                map_values = np.atleast_2d(np.asarray(signal_da.values, dtype=float))
                image = ax.imshow(map_values, aspect="auto", origin="lower", cmap="viridis")
                ax.set_xlabel("Sweep index")
                ax.set_ylabel("Sweep index")

            if (
                signal_prefix == "I_diff"
                and fit.get("success")
                and fit.get("hold_duration") is not None
                and fit.get("ramp_duration") is not None
            ):
                ax.plot(
                    fit["hold_duration"],
                    fit["ramp_duration"],
                    marker="x",
                    markersize=10,
                    color="red",
                    label="maximum",
                )
                ax.legend(loc="best")

            ax.set_title(f"{qname} — {panel_label}")
            fig_map.colorbar(image, ax=ax, label=panel_label)

    fig_map.suptitle("Initialisation optimisation map")
    fig_map.tight_layout()

    node.results["figures"] = {
        "initialisation_map": fig_map,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No-op by design: this node only reports the maximum map location."""
    for q in node.namespace["qubits"]:
        fit = node.results["fit_results"].get(q.name, {})
        if fit.get("success"):
            node.log(
                f"{q.name}: selected maximum at hold={fit['hold_duration']} ns, "
                f"ramp={fit['ramp_duration']} ns."
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
