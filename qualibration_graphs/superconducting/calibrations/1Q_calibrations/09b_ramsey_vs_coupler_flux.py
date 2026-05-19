# %% {Imports}
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_vs_coupler_flux import (
    Parameters,
    fit_raw_data,
    plot_raw_data,
    plot_fit_data,
    plot_frequency_vs_coupler_flux,
    process_raw_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        RAMSEY VS COUPLER FLUX
This program performs a Ramsey sequence (x90 - idle_time - x90 - measurement) on a
selected qubit (control or target) of a qubit pair while sweeping the flux applied to
the tunable coupler during the idle time.  A virtual Z-rotation is used to introduce an
artificial detuning so that the Ramsey oscillation frequency can be resolved.

This allows characterisation of how the coupler flux modifies the effective qubit
frequency (through dispersive shift / hybridisation).

Prerequisites:
    - Calibrated single-qubit gates (x90) and readout on the measured qubit.
    - A tunable-coupler element with a "const" operation defined in QUAM.
    - (optional) Calibrated readout for improved SNR.
"""


node = QualibrationNode[Parameters, Quam](
    name="09b_ramsey_vs_coupler_flux",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits

    n_avg = node.parameters.num_shots

    # Dephasing time sweep (in clock cycles = 4 ns)
    idle_times = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.wait_time_step_in_ns // 4,
    )

    detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
    fluxes = np.linspace(
        node.parameters.coupler_flux_min,
        node.parameters.coupler_flux_max,
        node.parameters.coupler_flux_num,
    )

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "coupler_flux": xr.DataArray(fluxes, attrs={"long_name": "coupler flux", "units": "V"}),
        "idle_times": xr.DataArray(4 * idle_times, attrs={"long_name": "idle times", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        init_state = [declare(int) for _ in range(num_qubit_pairs)]
        current_state = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
        t = declare(int)
        phi = declare(fixed)
        flux = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize QPU
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            measured_qubits_map = {
                ii: qp.qubit_control if node.parameters.measure_qubit == "control" else qp.qubit_target
                for ii, qp in multiplexed_qubit_pairs.items()
            }

            # Initial readout for XOR-based state tracking
            for ii, qp in multiplexed_qubit_pairs.items():
                measured_qubits_map[ii].readout_state(init_state[ii])

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(flux, fluxes)):
                    with for_(*from_array(t, idle_times)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qubit = measured_qubits_map[ii]
                            assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                            with strict_timing_():
                                qubit.xy.play("x90")
                                qubit.xy.frame_rotation_2pi(phi)
                                qubit.xy.wait(t + 1)
                                qp.coupler.wait(qubit.xy.operations["x90"].length * u.ns // 4)
                                qp.coupler.play(
                                    "const",
                                    amplitude_scale=flux / qp.coupler.operations["const"].amplitude,
                                    duration=t,
                                )
                                qubit.xy.play("x90")
                        align()

                        for ii, qp in multiplexed_qubit_pairs.items():
                            qubit = measured_qubits_map[ii]
                            qubit.readout_state(current_state[ii])
                            assign(state[ii], init_state[ii] ^ current_state[ii])
                            assign(init_state[ii], current_state[ii])
                            save(state[ii], state_st[ii])
                            reset_frame(qubit.xy.name)
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect, execute QUA program, fetch raw data and store as dataset 'ds_raw'."""
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
    if "qubit_pair" in dataset.dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
        dataset = dataset.rename({"qubit_pair": "qubit"})
        dataset = dataset.assign_coords(qubit=qubit_pair_names)
        dataset = dataset.assign_coords(measured_qubit_name=("qubit", measured_qubit_names))
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    measured_qubits = []
    for qp in node.namespace["qubit_pairs"]:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    node.namespace["qubits"] = measured_qubits
    # Rename qubit_pair dimension to qubit for compatibility with analysis functions
    # Use unique pair names as coordinate values (see execute_qua_program for rationale)
    if "qubit_pair" in node.results["ds_raw"].dims:
        qubit_pair_names = [qp.name for qp in node.namespace["qubit_pairs"]]
        measured_qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].rename({"qubit_pair": "qubit"})
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(qubit=qubit_pair_names)
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(
            measured_qubit_name=("qubit", measured_qubit_names)
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Extract oscillation frequencies from raw Ramsey data (no model fit)."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"] = fit_raw_data(node.results["ds_raw"], node)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw data heatmap and extracted frequency vs coupler flux."""
    fig_raw = plot_raw_data(node.results["ds_raw"], node.namespace["qubit_pairs"])
    fig_fitted = plot_fit_data(node.results["ds_fit"], node.namespace["qubit_pairs"])
    fig_freq = plot_frequency_vs_coupler_flux(node.results["ds_fit"], node.namespace["qubit_pairs"])
    node.results["figures"] = {
        "raw_data": fig_raw,
        **({"fitted_state": fig_fitted} if fig_fitted is not None else {}),
        **({"frequency_vs_coupler_flux": fig_freq} if fig_freq is not None else {}),
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            measured_qubit_name = (
                qp.qubit_control.name if node.parameters.measure_qubit == "control" else qp.qubit_target.name
            )
            node.machine.qubits[measured_qubit_name].extras[f"{qp.coupler.name}_dispersion_load_id"] = node.snapshot_idx


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results."""
    node.save()


# %%
