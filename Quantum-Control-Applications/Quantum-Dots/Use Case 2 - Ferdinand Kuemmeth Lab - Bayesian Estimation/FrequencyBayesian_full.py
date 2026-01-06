# %% Imports
from qualibrate import QualibrationNode, NodeParameters
from iqcc_calibration_tools.quam_config.components import Quam
from iqcc_calibration_tools.analysis.plot_utils import QubitGrid, grid_iter
from iqcc_calibration_tools.quam_config.macros import qua_declaration, readout_state
from qualang_tools.units import unit
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qm.qua import *
from typing import Optional, List
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

u = unit(coerce_to_integer=True)

# %% Helper Functions

def extract_string(input_string):
    """Extract the prefix string before the first digit."""
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)
    return input_string[:index] if index is not None else None


def fetch_results_as_xarray_arb_var(handles, qubits, measurement_axis, var_name=None):
    """
    Fetch measurement results as an xarray dataset.
    
    Parameters
    ----------
    handles : dict
        Dictionary containing stream handles from job.result_handles
    qubits : list
        List of qubit objects to fetch data for
    measurement_axis : dict
        Dictionary mapping axis names to coordinate values, e.g. {"frequency": freqs, "flux": flux_vals}
    var_name : str, optional
        Specific variable name to fetch. If None, automatically detects from handles.
    
    Returns
    -------
    xarray.Dataset
        Dataset containing the fetched measurement results with proper dimensions and coordinates
    """
    if var_name is None:
        stream_handles = handles.keys()
        meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    else:
        meas_vars = [var_name]
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, qubit in enumerate(qubits)] for meas_var in meas_vars
    ]
    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}
    
    
    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )

    return ds


# %% Node Parameters
class Parameters(NodeParameters):
    """Configuration parameters for Bayesian frequency estimation experiment."""
    
    # Qubit selection
    qubits: Optional[List[str]] = ["qC1", "qC2"]

    # Experiment parameters
    num_repetitions: int = 100
    detuning: int = 2 * u.MHz
    min_wait_time_in_ns: int = 36
    max_wait_time_in_ns: int = 6000
    wait_time_step_in_ns: int = 40
    physical_detuning: int = 0 * u.MHz

    # Bayesian estimation parameters (MHz)
    # Note: Frequency range should be between 0 and 8 MHz due to QUA fixed variable limitations
    f_min: float = 1  # MHz
    f_max: float = 3 # MHz
    df: float = 0.01  # MHz
    
    # Data collection
    keep_shot_data: bool = True

    # Execution parameters
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False


# Create experiment node
node = QualibrationNode(name="FrequencyBayes", parameters=Parameters())

# %% Initialize QuAM and QOP

# Load QuAM configuration
machine = Quam.load()

# Generate hardware configurations
config = machine.generate_config()

# Connect to quantum control hardware
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get qubit objects
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

num_qubits = len(qubits)


# %% QUA Program
# Set up experiment parameters
v_f = np.arange(node.parameters.f_min, node.parameters.f_max + 0.5 * node.parameters.df, node.parameters.df)

flux_shifts = {}
for qubit in qubits:
    flux_shift = np.sqrt(-node.parameters.physical_detuning/qubit.freq_vs_flux_01_quad_term)
    flux_shifts[qubit.name] = flux_shift

n_reps = node.parameters.num_repetitions
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
detuning = node.parameters.detuning - node.parameters.physical_detuning

# Define QUA program
with program() as BayesFreq:
    # Declare variables
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    # Bayes variables
    frequencies = declare(fixed, value=v_f.tolist())
    Pf_st = [declare_stream() for _ in range(num_qubits)]
    estimated_frequency_st = [declare_stream() for _ in range(num_qubits)]

    # Main experiment loop
    for i, qubit in enumerate(qubits):
        t = declare(int)
        phase = declare(fixed)
        estimated_frequency = declare(fixed)  # in MHz
        Pf = declare(fixed, value=(np.ones(len(v_f)) / len(v_f)).tolist())
        norm = declare(fixed)
        s = declare(int)

        t_sample = declare(fixed)  # normalization for time in us
        f = declare(fixed)
        C = declare(fixed)
        rk = declare(fixed)

        # SPAM parameters
        alpha = declare(fixed)
        beta = declare(fixed)

        # SPAM parameters from confusion matrix
        assign(alpha, qubit.resonator.confusion_matrix[0][1] - qubit.resonator.confusion_matrix[1][0])
        assign(beta, 1 - qubit.resonator.confusion_matrix[0][1] - qubit.resonator.confusion_matrix[1][0])

        # Set flux bias
        machine.set_all_fluxes(flux_point="joint", target=qubit)
        # Averaging loop
        with for_(n, 0, n < n_reps, n + 1):
            save(n, n_st)

            # Time sweep loop
            with for_(*from_array(t, idle_times)):
                assign(phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))

                qubit.xy.play("x90")
                qubit.xy.frame_rotation_2pi(phase)
                qubit.z.wait(duration=qubit.xy.operations["x180"].length // 4)
                qubit.xy.wait(t)
                qubit.z.play(
                    "const",
                    amplitude_scale=flux_shifts[qubit.name] / qubit.z.operations["const"].amplitude,
                    duration=t
                )
                qubit.xy.play("x90")


                # Measurement
                readout_state(qubit, state[i])
                if node.parameters.keep_shot_data:
                    save(state[i], state_st[i])
                qubit.align()
                qubit.xy.play("x180", condition=Cast.to_bool(state[i]))

                assign(rk, Cast.to_fixed(state[i]) - 0.5)
                assign(t_sample, Cast.mul_fixed_by_int(1e-3, t * 4))
                f_idx = declare(int)

                # Update P(f)
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(C, Math.cos2pi(frequencies[f_idx] * t_sample))
                    assign(
                        Pf[f_idx],
                        (0.5 + rk * (alpha + beta * C) * 0.99) * Pf[f_idx],
                    )

                # Normalize P(f)
                assign(norm, Cast.to_fixed(0.01 / Math.sum(Pf)))
                assign(norm, Math.abs(norm))
                with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                    assign(Pf[f_idx], Cast.mul_fixed_by_int(norm * Pf[f_idx], 100))

                qubit.align()
                reset_frame(qubit.xy.name)

            # Estimated frequency (argmax of posterior)
            # assign(f_idx, Math.argmax(Pf))
            # assign(estimated_frequency, frequencies[f_idx])
            assign(estimated_frequency, Math.dot(frequencies, Pf))

            qubit.xy.play("x90", amplitude_scale=0, duration=4, timestamp_stream=f'time_stamp{i+1}')

            # Save and reset P(f)
            with for_(f_idx, 0, f_idx < len(v_f), f_idx + 1):
                save(Pf[f_idx], Pf_st[i])
                assign(Pf[f_idx], 1 / len(v_f))

            save(estimated_frequency, estimated_frequency_st[i])

    # Stream processing
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            Pf_st[i].buffer(n_reps, len(v_f)).save(f"Pf{i + 1}")
            if node.parameters.keep_shot_data:
                state_st[i].buffer(n_reps, len(idle_times)).save(f"state{i + 1}")
            estimated_frequency_st[i].buffer(n_reps).save(f"estimated_frequency{i + 1}")
# %% Simulate or Execute
if node.parameters.simulate:
    pass
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(BayesFreq)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, node.parameters.num_repetitions, start_time=results.start_time)

# %% Data Fetching and Dataset Creation
if not node.parameters.simulate:
    if node.parameters.keep_shot_data:
        ds_single = fetch_results_as_xarray_arb_var(
            job.result_handles, qubits,
            {"t": idle_times * 4, "repetition": np.arange(1, n_reps + 1)},
            "state"
        )
    else:
        ds_single = None

    vf_array = np.arange(
        node.parameters.f_min,
        node.parameters.f_max + 0.5 * node.parameters.df,
        node.parameters.df
    )
    ds_Pf = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"vf": vf_array, "repetition": np.arange(1, n_reps + 1)},
        "Pf"
    )
    ds_estimated_frequency = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"repetition": np.arange(1, n_reps + 1)},
        "estimated_frequency"
    )
    ds_time_stamp = fetch_results_as_xarray_arb_var(
        job.result_handles, qubits,
        {"repetition": np.arange(1, n_reps + 1)},
        "time_stamp"
    )

    timestamp_values = ds_time_stamp.time_stamp.values

    ds_time_stamp = ds_time_stamp.assign(time_stamp=(ds_time_stamp.time_stamp.dims, timestamp_values))
    time_stamp = ((ds_time_stamp - ds_time_stamp.min(dim="repetition")) * 4e-9).time_stamp

    if node.parameters.keep_shot_data:
        ds = xr.merge([ds_single, ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, time_stamp])
    else:
        ds = xr.merge([ds_Pf / ds_Pf.Pf.sum(dim='vf'), ds_estimated_frequency, ds_time_stamp])

    node.results = {"ds": ds}

    # %% Plotting

    # Create qubit grid for single-shot data
    if node.parameters.keep_shot_data:
        grid_shot = QubitGrid(ds, [q.grid_location for q in qubits])
        y_data_key = "state"
        # Loop over grid axes and qubits
        for ax, qubit in grid_iter(grid_shot):
            qubit_name = qubit["qubit"]
            t_vals = ds_single.t.values
            y_vals = ds_single[y_data_key].sel(qubit=qubit_name).values

            # Plot data with pcolormesh
            X, Y = np.meshgrid(t_vals*1e-3, time_stamp.sel(qubit=qubit_name).values)
            pcm = ax.pcolormesh(X, Y, y_vals)
            ax.set_xlabel("time (µs)")
            ax.set_ylabel("time (s)")
            ax.set_title(qubit_name)
            grid_shot.fig.colorbar(pcm, ax=ax, label=f"{y_data_key}")
            ax.grid(False)

        grid_shot.fig.suptitle("Single-shot data")
        plt.tight_layout()
        node.results["state_figure"] = grid_shot.fig

    # Create qubit grid for Bayesian probability distribution
    grid_bayes = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "Pf"

    for ax, qubit in grid_iter(grid_bayes):
        qubit_name = qubit["qubit"]
        da = ds_Pf[y_data_key].sel(qubit=qubit_name)
        X, Y = np.meshgrid(da.vf.values, time_stamp.sel(qubit=qubit_name).values)
        data = da.values
        vmin = np.nanpercentile(data, 1)
        vmax = np.nanpercentile(data, 99)
        pcm = ax.pcolormesh(X, Y, data, vmin=vmin, vmax=vmax)
        ax.set_xlabel("frequency (MHz)")
        ax.set_ylabel("time (s)")
        ax.set_title(qubit_name)
        grid_bayes.fig.colorbar(pcm, ax=ax, label=y_data_key)
        ax.grid(False)

    grid_bayes.fig.suptitle("Frequency Bayes distribution")
    plt.tight_layout()
    node.results["PF_figure"] = grid_bayes.fig

    # Create qubit grid for estimated frequency
    grid_freq = QubitGrid(ds, [q.grid_location for q in qubits])
    y_data_key = "estimated_frequency"
    # Loop over grid axes and qubits
    for ax, qubit in grid_iter(grid_freq):
        qubit_name = qubit["qubit"]
        y_vals = ds_estimated_frequency[y_data_key].sel(qubit=qubit_name).values
        ax.plot(time_stamp.sel(qubit=qubit_name).values, y_vals, marker='.', linestyle='-', alpha=0.5)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("Estimated Frequency (MHz)")
        ax.set_title(qubit_name)
        ax.grid(False)

    grid_freq.fig.suptitle("Estimated Frequency")
    plt.tight_layout()
    node.results["estimated_frequency_figure"] = grid_freq.fig

    # %% Save Results
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
