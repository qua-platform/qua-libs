# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_conditional_phase_error_amp import (
    FitResults,
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from matplotlib.colors import ListedColormap
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
CALIBRATION OF THE CONTROLLED-PHASE (CPHASE) OF THE CZ GATE

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the
resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The amount of leakage to the |f> state when the control qubit is in the excited state

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Measuring the phase of the target qubit for both control qubit states
3. Calculating the phase difference
4. Measuring the population in the |f> state to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

State update:
- The optimal CZ gate amplitude: qubit_pair.gates["Cz"].flux_pulse_control.amplitude
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="14b_cz_conditional_phase_error_amp",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.qubit_pairs = ["qD1-qD2"]
    node.parameters.num_averages = 100
    node.parameters.operation = "cz_flattop"
    node.parameters.number_of_operations = 10
    node.parameters.amp_range = 0.015
    node.parameters.amp_step = 0.0001
    node.parameters.use_state_discrimination = True
    node.parameters.reset_type = "active"
    node.parameters.num_frames = 15
    node.parameters.load_data_id = 3698
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_averages
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    frames = np.arange(0, 1, 1 / node.parameters.num_frames)

    operation_name = node.parameters.operation
    num_operations = node.parameters.number_of_operations
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "number_of_operations": xr.DataArray(
            np.arange(1, num_operations + 1),
            attrs={"long_name": "number of operations"},
        ),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
        "control_axis": xr.DataArray([0, 1], attrs={"long_name": "control qubit state"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        amp = declare(fixed)
        frame = declare(fixed)
        frame_odd = declare(fixed)
        control_initial = declare(int)
        n = declare(int)
        n_op = declare(int)
        count = declare(int)
        n_st = declare_stream()
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                    with for_(*from_array(amp, amplitudes)):
                        with for_(*from_array(frame, frames)):
                            with for_(*from_array(control_initial, [0, 1])):
                                for ii, qp in multiplexed_qubit_pairs.items():
                                    qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                                    qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                                    qp.align()
                                    reset_frame(qp.qubit_target.xy.name)
                                    reset_frame(qp.qubit_control.xy.name)
                                    # setting both qubits to the initial state
                                    qp.qubit_control.xy.play("x180", condition=control_initial == 1)
                                    qp.qubit_target.xy.play("x90")
                                    qp.align()
                                    with for_(count, 0, count < n_op, count + 1):
                                        # play the CZ gate
                                        qp.macros[operation_name].apply(amplitude_scale_control=amp)
                                        # qp.wait(50)
                                    # rotate the frame
                                    with if_(((n_op & 1) == 0) & (control_initial == 1)):
                                        assign(frame_odd, frame - 0.5)
                                        qp.qubit_target.xy.frame_rotation_2pi(frame_odd)
                                    with else_():
                                        qp.qubit_target.xy.frame_rotation_2pi(frame)
                                    # return the target qubit before measurement
                                    qp.qubit_target.xy.play("x90")
                                    qp.align()

                                    if node.parameters.use_state_discrimination:
                                        # measure both qubits
                                        qp.qubit_control.readout_state(state_c[ii])
                                        qp.qubit_target.readout_state(state_t[ii])
                                        save(state_c[ii], state_c_st[ii])
                                        save(state_t[ii], state_t_st[ii])
                                    else:
                                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                        save(I_c[ii], I_c_st[ii])
                                        save(Q_c[ii], Q_c_st[ii])
                                        save(I_t[ii], I_t_st[ii])
                                        save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"state_control{i + 1}")
                    state_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"state_target{i + 1}")
                else:
                    I_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"Q_target{i + 1}")


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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
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
                node.parameters.num_averages,
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
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"] = fit_raw_data(node.results["ds_raw"], node)

    import matplotlib.colors as mcolors
    # Define colors that wrap around smoothly
    colors = ["white", "grey", "blue", "midnightblue", "blue", "grey", "white"]


    # Build cyclic colormap
    cyclic_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cyclic", colors, N=256)
    fig, ax = plt.subplots()
    # Base plot in greyscale
    # phase_diff_da = np.abs((node.results["ds_fit"].phase_diff.isel(qubit_pair=0) - 0.5))
    phase_diff_da = node.results["ds_fit"].phase_diff.isel(qubit_pair=0)
    phase_diff_da.plot(cmap="twilight_shifted", ax=ax)

    # tol = 0.03
    # # Build a mask for values within ±0.02 of 0.5
    # mask = (np.abs(phase_diff_da) <= tol) | (np.abs(phase_diff_da - 1) <= tol)
    # # Convert mask to 1 (red) / NaN (transparent)
    # mask_da = phase_diff_da.where(mask)
    # # We only need a single solid color (red) regardless of the underlying phase value,
    # # so replace all finite values by 1.
    # mask_plot = xr.full_like(mask_da, 1.0)
    # mask_plot = mask_plot.where(mask)  # keep NaN where mask is False

    # # Create a single-color colormap (red). Using alpha for translucency so the greyscale shows through.
    # red_cmap = ListedColormap(["red"])
    # mask_plot.plot(ax=ax, cmap=red_cmap, add_colorbar=False)
    # ax.set_title("Phase diff with highlighted region |value - 0.5| ≤ 0.02")

    # # Store fit results in the format expected by the rest of the node
    # node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # # Log the relevant information extracted from the data analysis
    # log_fitted_results(fit_results, log_callable=node.log)

    # # Set outcomes based on fit success
    # node.outcomes = {
    #     qubit_pair_name: ("successful" if fit_result.success else "failed")
    #     for qubit_pair_name, fit_result in fit_results.items()
    # }
# %%

X = node.results["ds_fit"].amp_full.values[0]
Y = node.results["ds_fit"].number_of_operations.values
Z = node.results["ds_fit"].phase_diff.isel(qubit_pair=0).values


import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline

def track_ridge_rowwise(X, Y, Z, center_guess=None, search_window=8, smooth=True):
    """
    Track ridge row by row, starting from the first row.

    Parameters
    ----------
    X : 1D array (amps)
    Y : 1D array (number_of_operations)
    Z : 2D array [len(Y), len(X)] with values in [0,1) wrapping cyclically
    center_guess : float or None
        Expected center (X-units). If None, defaults to mean(X).
    search_window : int
        Half-width of the window (in index units) for row-to-row tracking.
    smooth : bool
        If True, return also a spline-smoothed version.

    Returns
    -------
    x_ridge : raw tracked ridge (X-units)
    x_ridge_smooth : smoothed spline fit (X-units) if smooth=True else None
    """
    # optional smoothing
    Zs = gaussian_filter(Z, sigma=(0.6, 0.6))

    # circular distance to 0.5
    D = np.abs(((Zs - 0.5 + 0.5) % 1.0) - 0.5)

    ny, nx = D.shape
    ridge_idx = np.zeros(ny, dtype=float)

    # --- start at first row ---
    if center_guess is None:
        center_guess = np.mean(X)
    start_idx = np.argmin(np.abs(X - center_guess))

    # among minima, choose closest to start_idx
    local_min = np.argmin(D[0, :])
    ridge_idx[0] = local_min if abs(local_min - start_idx) < search_window else start_idx

    # --- trace subsequent rows ---
    for i in range(1, ny):
        prev_idx = int(round(ridge_idx[i-1]))
        lo = max(0, prev_idx - search_window)
        hi = min(nx, prev_idx + search_window + 1)

        local_idx = np.argmin(D[i, lo:hi]) + lo

        # quadratic sub-pixel refinement
        if 0 < local_idx < nx-1:
            y1, y2, y3 = D[i, local_idx-1:local_idx+2]
            denom = (y1 - 2*y2 + y3)
            if denom != 0:
                delta = 0.5 * (y1 - y3) / denom
                local_idx += np.clip(delta, -0.5, 0.5)

        ridge_idx[i] = local_idx

    # convert to physical X-units
    x_ridge = np.interp(ridge_idx, np.arange(nx), X)

    if smooth:
        spline = UnivariateSpline(Y, x_ridge, s=len(Y)*1e-6)
        return x_ridge, spline(Y)
    else:
        return x_ridge, None


x_ridge_raw, x_ridge_smooth = track_ridge_rowwise(X, Y, Z, center_guess=0.0642)

plt.pcolor(X,Y,Z,cmap='twilight_shifted')
plt.plot(x_ridge_raw, Y, 'k.', label='tracked raw points')
# if x_ridge_smooth is not None:
#     plt.plot(x_ridge_smooth, Y, 'g-', lw=2, label='smoothed ridge')
plt.legend()
# %%

# %%
import numpy as np
from scipy.ndimage import gaussian_filter1d

def _circ_dist_to_half(Z):
    # circular distance to 0.5 in [0, 0.5]
    return np.abs(((Z - 0.5 + 0.5) % 1.0) - 0.5)

def fit_full_amp(X, Z, row_mask=None, trim=0.2, smooth_rows_sigma=0.6, smooth_cols_sigma=1.0):
    """
    Find the single amplitude that minimizes distance to 0.5 across all repetitions.

    Parameters
    ----------
    X : (nx,) array of amplitudes (in physical units).
    Z : (ny, nx) array, values in [0,1) with wrap.
    row_mask : (ny,) bool array to include only certain repetitions (optional).
    trim : fraction (0..0.45) for a trimmed-mean aggregator across rows (robust).
    smooth_rows_sigma : gaussian sigma to lightly smooth along rows (ny axis).
    smooth_cols_sigma : gaussian sigma to lightly smooth the final 1-D cost over columns.

    Returns
    -------
    x_star : best full_amp (float)
    C      : column cost array (nx,), smaller = better
    j_star : subpixel column index of the optimum
    """

    Zw = Z.copy()
    # Light smoothing only across repetitions; preserves column structure
    if smooth_rows_sigma and smooth_rows_sigma > 0:
        Zw = gaussian_filter1d(Zw, sigma=smooth_rows_sigma, axis=0, mode='nearest')

    # choose rows to use
    if row_mask is None:
        row_mask = np.ones(Zw.shape[0], dtype=bool)

    D = _circ_dist_to_half(Zw[row_mask, :])  # (n_sel, nx)

    # Robust column cost: trimmed mean across selected rows
    n = D.shape[0]
    k = int(np.floor(trim * n))
    D_sorted = np.sort(D, axis=0)
    if 2 * k < n:
        C = D_sorted[k:n - k, :].mean(axis=0)
    else:
        C = D_sorted.mean(axis=0)

    # Optional gentle smoothing of the 1-D cost over columns (denoise)
    if smooth_cols_sigma and smooth_cols_sigma > 0:
        C = gaussian_filter1d(C, sigma=smooth_cols_sigma, axis=0, mode='nearest')

    # Discrete minimum
    j0 = int(np.argmin(C))

    # Sub-pixel quadratic refinement on C[j0-1:j0+2]
    j_star = float(j0)
    if 0 < j0 < len(X) - 1:
        y1, y2, y3 = C[j0 - 1], C[j0], C[j0 + 1]
        denom = (y1 - 2 * y2 + y3)
        if denom != 0:
            delta = 0.5 * (y1 - y3) / denom   # vertex offset in [-0.5, 0.5]
            j_star = j0 + np.clip(delta, -0.5, 0.5)

    # Map refined index to physical amplitude
    x_star = np.interp(j_star, np.arange(len(X)), X)

    return x_star, C, j_star

x_star, C, j_star = fit_full_amp(X, Z, trim=0.2)

# Overlay on your heatmap
plt.pcolor(X,Y,Z,cmap='twilight_shifted')
plt.axvline(x_star, color='green', lw=3, label='full_amp fit')
plt.legend()

# (Optional) show the 1-D cost vs amplitude
plt.figure()
plt.plot(X, C)
plt.axvline(x_star, lw=2)
plt.xlabel('amp'); plt.ylabel('robust cost to 0.5'); plt.title('Column-wise cost')



# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    qubit_pairs = node.namespace["qubit_pairs"]

    # Plot phase calibration data
    fig_phase = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubit_pairs,
    )
    plt.show()

    node.results["phase_figure"] = fig_phase


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit pair data analysis was successful."""

    operation_name = node.parameters.operation
    with node.record_state_updates():
        fit_results = node.results["fit_results"]
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            qp.macros[operation_name].flux_pulse_control.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
