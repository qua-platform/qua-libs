# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.parametric_iswap import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
        ...
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="99_parametric_iswap",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q1-2"]
    node.parameters.num_shots = 200
    node.parameters.qubit_pairs = ["qB1-B2"]
    node.parameters.modulation_range_mhz = 10
    node.parameters.modulation_step_mhz = 0.1
    node.parameters.min_time = 16
    node.parameters.max_time = 300
    node.parameters.time_step = 4
    node.parameters.use_state_discrimination = True
    node.parameters.reset_type = "active"
    # node.parameters.modulation_amplitude = 0.07
    node.parameters.modulation_amplitude = 0.37
    node.parameters.cz_or_iswap = "iswap"
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
    n_avg = node.parameters.num_shots  # The number of averages

    # Loop parameters
    if node.parameters.cz_or_iswap == "iswap":
        node.namespace["central_frequencies"] =central_frequencies = [
            int(
                node.machine.qubits[qp.qubit_control.name].xy.RF_frequency
                - node.machine.qubits[qp.qubit_target.name].xy.RF_frequency
            )
            for qp in qubit_pairs
        ]
    else:
        node.namespace["central_frequencies"] = central_frequencies = [
            int(
                node.machine.qubits[qp.qubit_control.name].xy.RF_frequency
                - node.machine.qubits[qp.qubit_target.name].xy.RF_frequency
                + qp.qubit_target.anharmonicity
            )
            for qp in qubit_pairs
        ]


    # node.namespace["central_frequencies"] = central_frequencies = [440e6] #TODO: remove
    node.namespace["central_frequencies"] = central_frequencies = [312e6] #TODO: remove
    print("Central frequencies (MHz): ", central_frequencies)

     # Define the frequency sweep around the central frequency
    span = node.parameters.modulation_range_mhz * u.MHz
    step = node.parameters.modulation_step_mhz * u.MHz
    frequency_sweep = np.arange(-span / 2, span / 2, step)

    durations = np.arange(
        node.parameters.min_time,
        node.parameters.max_time,
        node.parameters.time_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "frequencies": xr.DataArray(frequency_sweep, attrs={"long_name": "coupler frequency", "units": "MHz"}),
        "durations": xr.DataArray(durations, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        coupler_freq = declare(int)
        duration = declare(int)
        n_st = declare_stream()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for ii, qp in multiplexed_qubit_pairs.items():
                print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(coupler_freq, frequency_sweep)):
                        qp.coupler.update_frequency(coupler_freq + central_frequencies[ii])
                        with for_(*from_array(duration, durations)):
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()

                            qp.qubit_control.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                qp.qubit_target.xy.play("x180")
                            qp.align()
                            # qp.coupler.reset_if_phase()
                            qp.coupler.play(
                                "const",
                                amplitude_scale=node.parameters.modulation_amplitude
                                / qp.coupler.operations["const"].amplitude,
                                duration=duration >> 2,
                            )
                            qp.align()

                            # readout
                            if node.parameters.use_state_discrimination:
                                # if node.parameters.cz_or_iswap == "cz":
                                #     qp.qubit_control.readout_state_gef(state_c[ii])
                                # else:
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
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(
                        f"state_control{i}"
                    )
                    state_t_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(f"state_target{i}")
                else:
                    I_c_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(durations)).buffer(len(frequency_sweep)).average().save(f"Q_target{i}")


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
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %%

ds = node.results["ds_raw"]

offset = node.namespace["central_frequencies"][0]   # example offset

ds = ds.assign_coords(
    frequencies_shifted = ds.frequencies + offset
)


fig, axs = plt.subplots(1, 2, figsize=(10, 7))
ds.state_control.plot(ax=axs[0], y="frequencies_shifted")
ds.state_target.plot(ax=axs[1], y="frequencies_shifted")
fig.tight_layout()

node.results["figures"] = {"raw_data_example": fig}
data = ds.state_control.values[0]

fft_data = np.fft.fft(data - np.mean(data))
freqs = np.fft.fftfreq(len(data[0]), d=node.parameters.time_step * 1e-9)

fig_fft, ax_fft = plt.subplots(figsize=(6, 4))
ax_fft.pcolormesh(freqs[fft_data.shape[1] // 2:], ds.frequencies_shifted.values, np.abs(fft_data)[:, fft_data.shape[1] // 2:], shading="auto")
fig_fft.tight_layout()

node.results["figures"]["fft_data_example"] = fig_fft




node.save()

# %%
# 1) Add the new coord; this keeps its dims from frequencies_shifted
ds2 = ds.assign_coords(if_freq=ds.frequencies_shifted)

# 2) Make if_freq the dimension instead of 'frequencies'
ds2 = ds2.swap_dims({"frequencies": "if_freq"})

fig , axs = plt.subplots(1, 1, figsize=(10, 7))
ds2.state_target.sel(if_freq=311.4e6).plot(ax=axs)
ds2.state_control.sel(if_freq=311.4e6).plot(ax=axs)
plt.show()




# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result["success"] else "failed")
        for qubit_pair_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"]
    )
    plt.show()
    node.results["figures"] = {"raw_fit": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            qp.coupler.decouple_offset = node.results["fit_results"][qp.name]["optimal_coupler_flux"]
            qp.detuning = node.results["fit_results"][qp.name]["optimal_qubit_flux"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
