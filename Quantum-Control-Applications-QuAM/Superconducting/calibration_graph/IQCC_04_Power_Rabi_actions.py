"""
POWER RABI WITH ERROR AMPLIFICATION

This sequence repeatedly executes a qubit pulse (e.g. x180) 'N' times and measures the resonator
state across a sweep of pulse amplitudes and numbers of pulses. By amplifying the effect of amplitude
inaccuracies, the precise pi pulse amplitude is determined.

Prerequisites:
    - Resonator frequency determined via resonator spectroscopy.
    - IQ mixer connected to the qubit drive line is calibrated.
    - Rough qubit frequency and desired pi pulse duration are known (from qubit spectroscopy).
    - Desired flux bias is set.

Post-run steps:
    - Update the qubit pulse amplitude in the state.
    - Save the current state.
"""

# %% Imports
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% Node Parameters
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 50
    operation_x180_or_any_90: Literal["x180", "x90", "-x90", "y90", "-y90"] = "x180"
    min_amp_factor: float = 0.0
    max_amp_factor: float = 1.5
    amp_factor_step: float = 0.05
    max_number_rabi_pulses_per_sweep: int = 1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    state_discrimination: bool = False
    update_x90: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


# Create the calibration node instance.
node = QualibrationNode[Parameters, QuAM](name="IQCC_04_Power_Rabi_actions")


# %% Helper Functions
def select_qubits(node: QualibrationNode[Parameters, QuAM]):
    """
    Returns the list of qubits to use.
    If no specific qubits are given in the parameters, returns all active qubits.
    """
    if not node.parameters.qubits:
        return node.machine.active_qubits
    else:
        return [node.machine.qubits[q] for q in node.parameters.qubits]


def generate_pulse_count_vector(max_pulses: int, pulse_op: Literal["x180", "x90", "-x90", "y90", "-y90"]):
    """
    Generates an array of pulse counts for the sweep.
    For a single pulse sweep, returns a single value.
    For multiple pulses, returns values according to the pulse type.
    """
    if max_pulses > 1:
        if pulse_op == "x180":
            return np.arange(1, max_pulses, 2).astype(int)
        elif pulse_op in ["x90", "-x90", "y90", "-y90"]:
            return np.arange(2, max_pulses, 4).astype(int)
        else:
            raise ValueError(f"Unknown pulse operation: {pulse_op}")
    else:
        return np.linspace(1, max_pulses, max_pulses).astype(int)[::2]


def generate_amplitude_factors(node: QualibrationNode[Parameters, QuAM]):
    """
    Generates the amplitude scaling factors for the qubit pulse.
    """
    return np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )


# %% Action: Load Machine Configuration
@node.run_action
def initialize_quam(node: QualibrationNode[Parameters, QuAM]):
    """
    Loads the machine configuration from QuAM.
    """
    machine = QuAM.load()
    return {"machine": machine}


# %% Action: Connect to QMM (if not loading previous data)
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def initialize_qmm(node: QualibrationNode[Parameters, QuAM]):
    """
    Connects to the QMM hardware if no previous data is being loaded.
    """
    qmm = node.machine.connect()
    return {"qmm": qmm}


# %% Action: Build the QUA Program
@node.run_action
def qua_program(node: QualibrationNode[Parameters, QuAM]):
    """
    Constructs the QUA program for the power Rabi experiment.
    The program sweeps over amplitude scaling factors and number of pulses.
    """
    # Shorthand for units.
    u = unit(coerce_to_integer=True)

    # Read parameters.
    num_averages = node.parameters.num_averages
    max_pulses = node.parameters.max_number_rabi_pulses_per_sweep
    flux_target = node.parameters.flux_point_joint_or_independent
    reset_mode = node.parameters.reset_type_thermal_or_active
    state_disc = node.parameters.state_discrimination
    pulse_operation = node.parameters.operation_x180_or_any_90

    # Create sweep arrays.
    amplitude_factors = generate_amplitude_factors(node)
    pulse_count_vector = generate_pulse_count_vector(max_pulses, pulse_operation)
    qubits = select_qubits(node)

    with program() as qua_prog:
        # Declare variables and streams for each qubit.
        I, I_stream, Q, Q_stream, avg_idx, avg_idx_stream = qua_declaration(num_qubits=len(qubits))
        if state_disc:
            state = [declare(bool) for _ in range(len(qubits))]
            state_stream = [declare_stream() for _ in range(len(qubits))]
        amp_scale = declare(fixed)  # Amplitude scaling factor variable.
        pulse_count = declare(int)  # Number of pulses variable.
        pulse_counter = declare(int)  # Loop counter for pulses.

        # Loop over each qubit.
        for idx, qb in enumerate(qubits):
            # Set the flux bias.
            node.machine.set_all_fluxes(flux_point=flux_target, target=qb)

            with for_(avg_idx, 0, avg_idx < num_averages, avg_idx + 1):
                save(avg_idx, avg_idx_stream)
                with for_(*from_array(pulse_count, pulse_count_vector)):
                    with for_(*from_array(amp_scale, amplitude_factors)):
                        # Reset the qubit state.
                        if reset_mode == "active":
                            active_reset(qb, "readout")
                        else:
                            qb.wait(qb.thermalization_time * u.ns)

                        # Apply the qubit pulse repeatedly.
                        with for_(pulse_counter, 0, pulse_counter < pulse_count, pulse_counter + 1):
                            qb.xy.play(pulse_operation, amplitude_scale=amp_scale)
                        qb.align()
                        qb.resonator.measure("readout", qua_vars=(I[idx], Q[idx]))
                        if state_disc:
                            assign(state[idx], I[idx] > qb.resonator.operations["readout"].threshold)
                            save(state[idx], state_stream[idx])
                        else:
                            save(I[idx], I_stream[idx])
                            save(Q[idx], Q_stream[idx])
            if not node.parameters.multiplexed:
                align()

        # Process and save streams.
        with stream_processing():
            avg_idx_stream.save("n")
            for idx, qb in enumerate(qubits):
                if pulse_operation == "x180":
                    if state_disc:
                        state_stream[idx].boolean_to_int().buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 2))
                        ).average().save(f"state{idx + 1}")
                    else:
                        I_stream[idx].buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 2))
                        ).average().save(f"I{idx + 1}")
                        Q_stream[idx].buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 2))
                        ).average().save(f"Q{idx + 1}")
                elif pulse_operation in ["x90", "-x90", "y90", "-y90"]:
                    if state_disc:
                        state_stream[idx].boolean_to_int().buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 4))
                        ).average().save(f"state{idx + 1}")
                    else:
                        I_stream[idx].buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 4))
                        ).average().save(f"I{idx + 1}")
                        Q_stream[idx].buffer(len(amplitude_factors)).buffer(
                            int(np.ceil(max_pulses / 4))
                        ).average().save(f"Q{idx + 1}")
                else:
                    raise ValueError(f"Unknown pulse operation: {pulse_operation}")

    # Store sweep arrays for later actions.
    return {
        "qua_program": qua_prog,
        "amplitude_factors": amplitude_factors,
        "pulse_count_vector": pulse_count_vector,
    }


# %% Action: Simulation (if simulation flag is True)
@node.run_action(skip_if=not node.parameters.simulate)
def simulate(node: QualibrationNode[Parameters, QuAM]):
    """
    Simulates the QUA program for a specified duration.
    The simulated data for each controller is plotted.
    """
    sim_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
    job = node.machine.qmm.simulate(
        config=node.machine.generate_config(),
        qua_program=node.namespace["qua_program"],
        simulation_config=sim_config,
    )
    samples = job.get_simulated_samples()
    fig, axes = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, controller in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[controller].plot()
        plt.title(controller)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}


# %% Action: Execute on Hardware (if not simulating or loading previous data)
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute(node: QualibrationNode[Parameters, QuAM]):
    """
    Executes the QUA program on the hardware using qm_session.
    A progress counter is shown as the averages are acquired.
    """
    config = node.machine.generate_config()
    with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(node.namespace["qua_program"])
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n_val = results.fetch_all()[0]
            progress_counter(n_val, node.parameters.num_averages, start_time=results.start_time)
    return {"job": job}


# %% Action: Fetch Data (if not simulating or loading previous data)
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def fetch_data(node: QualibrationNode[Parameters, QuAM]):
    """
    Retrieves calibration data, converts it to an xarray dataset with appropriate coordinates,
    and, if needed, converts I/Q data to voltage.
    """
    qubits = select_qubits(node)
    job = node.namespace["job"]
    amplitude_factors = node.namespace["amplitude_factors"]
    pulse_count_vector = node.namespace["pulse_count_vector"]
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amplitude_factors, "N": pulse_count_vector})
    if not node.parameters.state_discrimination:
        ds = convert_IQ_to_V(ds, qubits)

    # Add the absolute qubit pulse amplitude as a coordinate.
    pulse_operation = node.parameters.operation_x180_or_any_90
    pulse_amplitudes = {qb: qb.xy.operations[pulse_operation].amplitude for qb in qubits}
    amplitude_scales = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    ds = ds.assign_coords(
        {
            "abs_amp": (["qubit", "amp"], np.array([pulse_amplitudes[qb] * amplitude_scales for qb in qubits])),
        }
    )
    node.results["ds"] = ds


# %% Action: Load Previously Saved Data
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, QuAM]):
    """
    Loads data from a previous calibration run.
    """
    node.load_from_id(node.parameters.load_data_id)


# %% Action: Data Analysis
@node.run_action(skip_if=node.parameters.simulate)
def data_analysis(node: QualibrationNode[Parameters, QuAM]):
    """
    Analyzes the calibration dataset to determine the optimal pi pulse amplitude.
    For a single pulse sweep, the oscillation is fit.
    For multiple pulses, the data are averaged to find the best amplitude.
    """
    ds = node.results["ds"]
    qubits = select_qubits(node)
    pulse_operation = node.parameters.operation_x180_or_any_90
    fit_results = {}
    fit_evals = None  # For single pulse analysis.
    data_max_idx = None  # For multi-pulse analysis.

    if node.parameters.max_number_rabi_pulses_per_sweep == 1:
        # Fit oscillations for a single pulse sweep.
        if node.parameters.state_discrimination:
            fit = fit_oscillation(ds.state, "amp")
        else:
            fit = fit_oscillation(ds.I, "amp")
        fit_evals = oscillation(
            ds.amp,
            fit.sel(fit_vals="a"),
            fit.sel(fit_vals="f"),
            fit.sel(fit_vals="phi"),
            fit.sel(fit_vals="offset"),
        )
        # Calculate new pulse amplitudes based on fit.
        for qb in qubits:
            fit_results[qb.name] = {}
            f_fit = fit.loc[qb.name].sel(fit_vals="f")
            phi_fit = fit.loc[qb.name].sel(fit_vals="phi")
            # Adjust phase if needed.
            phi_fit = phi_fit - np.pi * (phi_fit > np.pi / 2)
            scale_factor = float((np.pi - phi_fit) / (2 * np.pi * f_fit))
            current_pulse = qb.xy.operations[pulse_operation]
            new_pi_amp = current_pulse.amplitude * scale_factor
            limits = instrument_limits(qb.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                print(f"Qubit {qb.name}: Pi pulse amplitude scaled by factor {scale_factor:.2f}")
                print(f"New amplitude: {1e3 * new_pi_amp:.2f} {limits.units}\n")
                fit_results[qb.name]["Pi_amplitude"] = new_pi_amp
            else:
                print(f"Qubit {qb.name}: Fitted amplitude too high; using {limits.max_x180_wf_amplitude}\n")
                fit_results[qb.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude
        node.results["fit_results"] = fit_results

    elif node.parameters.max_number_rabi_pulses_per_sweep > 1:
        # Average data along pulse count to choose the best amplitude.
        if node.parameters.state_discrimination:
            signal_avg = ds.state.mean(dim="N")
        else:
            signal_avg = ds.I.mean(dim="N")
        pulse_count_vector = generate_pulse_count_vector(
            node.parameters.max_number_rabi_pulses_per_sweep, pulse_operation
        )
        # Choose optimum based on whether the pulse count is even or odd.
        if (pulse_count_vector[0] % 2 == 0 and pulse_operation == "x180") or (
            pulse_count_vector[0] % 2 != 0 and pulse_operation != "x180"
        ):
            data_max_idx = signal_avg.argmin(dim="amp")
        else:
            data_max_idx = signal_avg.argmax(dim="amp")
        for qb in qubits:
            fit_results[qb.name] = {}
            new_pi_amp = float(ds.abs_amp.sel(qubit=qb.name)[data_max_idx.sel(qubit=qb.name)].data)
            limits = instrument_limits(qb.xy)
            if new_pi_amp < limits.max_x180_wf_amplitude:
                fit_results[qb.name]["Pi_amplitude"] = new_pi_amp
                print(f"Qubit {qb.name}: New Pi amplitude is {1e3 * new_pi_amp:.2f} {limits.units}\n")
            else:
                print(f"Qubit {qb.name}: Fitted amplitude too high; using {limits.max_x180_wf_amplitude}\n")
                fit_results[qb.name]["Pi_amplitude"] = limits.max_x180_wf_amplitude
        node.results["fit_results"] = fit_results

    return {"fit_evals": fit_evals, "data_max_idx": data_max_idx}


# %% Action: Plot Data
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, QuAM]):
    """
    Generates plots of the calibration data.
    For single pulse sweeps, overlays the fitted oscillation.
    For multiple pulses, highlights the optimal amplitude.
    """
    ds = node.results["ds"]
    qubits = select_qubits(node)
    fit_evals = node.results.get("fit_evals")
    data_max_idx = node.results.get("data_max_idx")
    grid = QubitGrid(ds, [qb.grid_location for qb in qubits])
    for ax, qb in grid_iter(grid):
        if node.parameters.max_number_rabi_pulses_per_sweep == 1:
            if node.parameters.state_discrimination:
                ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qb].state.plot(ax=ax, x="amp_mV")
                ax.plot(ds.abs_amp.loc[qb] * 1e3, fit_evals.loc[qb][0])
                ax.set_ylabel("Qubit state")
            else:
                (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qb].I * 1e3).plot(ax=ax, x="amp_mV")
                ax.plot(ds.abs_amp.loc[qb] * 1e3, 1e3 * fit_evals.loc[qb][0])
                ax.set_ylabel("Trans. amp. I [mV]")
        elif node.parameters.max_number_rabi_pulses_per_sweep > 1:
            if node.parameters.state_discrimination:
                ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qb].state.plot(ax=ax, x="amp_mV", y="N")
            else:
                (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qb].I * 1e3).plot(ax=ax, x="amp_mV", y="N")
            ax.set_ylabel("Number of pulses")
            ax.axvline(1e3 * ds.abs_amp.loc[qb][data_max_idx.loc[qb]], color="r")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qb["qubit"])
    grid.fig.suptitle("Rabi: I vs. Amplitude")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


# %% Action: Update Calibration State
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, QuAM]):
    """
    Updates the qubit state with the newly determined pi pulse amplitude.
    If the primary operation is 'x180' and update_x90 is enabled, also updates the x90 amplitude.
    """
    qubits = select_qubits(node)
    pulse_operation = node.parameters.operation_x180_or_any_90
    fit_results = node.results["fit_results"]
    with node.record_state_updates():
        for qb in qubits:
            qb.xy.operations[pulse_operation].amplitude = fit_results[qb.name]["Pi_amplitude"]
            if pulse_operation == "x180" and node.parameters.update_x90:
                qb.xy.operations["x90"].amplitude = fit_results[qb.name]["Pi_amplitude"] / 2


# %% Action: Save Results
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def save_results(node: QualibrationNode[Parameters, QuAM]):
    """
    Records the outcomes of the calibration and saves the initial parameters.
    """
    qubits = select_qubits(node)
    node.outcomes = {qb.name: "successful" for qb in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.save()


# %% End of Script
