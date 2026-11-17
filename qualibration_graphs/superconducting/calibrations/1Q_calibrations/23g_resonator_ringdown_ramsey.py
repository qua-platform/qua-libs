# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from calibration_utils.resonator_ringdown_ramsey import (
    Parameters,
    get_wait_times_in_ns,
    get_phases,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    plot_normalized_phase,
    plot_contrast,
)

# %% {Node initialisation}
description = """
        RESONATOR RING-DOWN WITH A RAMSEY PROBE
The qubit is prepared in |g> or |e>, the readout pulse rings up the resonator, and after a varying delay a fixed
Ramsey sequence (x90 - idle - phase - x90) probes the photons left in the resonator: they shift the qubit frequency
through the AC-Stark effect, so the Ramsey phase decays with the delay as the resonator rings down. The phase of the
second x90 pulse is swept over one full turn to extract the Ramsey phase at every delay, which is fitted to

    phi(t) = A * exp(-t / tau) + phi_inf.

The resonator is rung up with the qubit in |g> and in |e>, as the two ring-down times generally differ. The depletion
time is the delay after which the residual phase has decayed to residual_fraction, taking the longer of the two:
    depletion_time = max(tau_g, tau_e) * ln(1 / residual_fraction).

Prerequisites:
    - Having calibrated the qubit x90 and x180 pulses (nodes 04b and 10b).
    - Having calibrated the readout parameters (nodes 02a, 02b, 07 and/or 08a, 08b).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The resonator depletion time: qubit.resonator.depletion_time
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="23g_resonator_ringdown_ramsey",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    wait_times = get_wait_times_in_ns(node.parameters)
    phases = get_phases(node.parameters)
    init_states = ["g", "e"]
    ramsey_idle_cycles = node.parameters.ramsey_idle_time_in_ns // 4
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "init_state": xr.DataArray(init_states, attrs={"long_name": "qubit state during the readout pulse"}),
        "wait_time": xr.DataArray(wait_times, attrs={"long_name": "delay after the readout pulse", "units": "ns"}),
        "phase": xr.DataArray(phases, attrs={"long_name": "phase of the second x90 pulse", "units": "2pi"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        t = declare(int)
        phi = declare(fixed)
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                for init_state in init_states:
                    with for_each_(t, (wait_times // 4).tolist()):
                        with for_each_(phi, phases.tolist()):
                            # Reset the qubits to the ground state
                            for i, qubit in multiplexed_qubits.items():
                                reset_frame(qubit.xy.name)
                                qubit.reset(
                                    node.parameters.reset_type,
                                    node.parameters.simulate,
                                    log_callable=node.log,
                                )
                            align()

                            for i, qubit in multiplexed_qubits.items():
                                # Ring up the resonator with the qubit in |g> or |e>
                                if init_state == "e":
                                    qubit.xy.play("x180")
                                    qubit.align()
                                qubit.resonator.measure(operation, qua_vars=(I[i], Q[i]))
                                qubit.align()
                                # Ramsey probe of the residual photons, a varying delay after the readout pulse
                                qubit.xy.wait(t)
                                qubit.xy.play("x90")
                                qubit.xy.wait(ramsey_idle_cycles)
                                qubit.xy.frame_rotation_2pi(phi)
                                qubit.xy.play("x90")
                            align()

                            # Measure the state of the resonators
                            for i, qubit in multiplexed_qubits.items():
                                if node.parameters.use_state_discrimination:
                                    qubit.readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(phases)).buffer(len(wait_times)).buffer(len(init_states)).average().save(
                        f"state{i + 1}"
                    )
                else:
                    I_st[i].buffer(len(phases)).buffer(len(wait_times)).buffer(len(init_states)).average().save(
                        f"I{i + 1}"
                    )
                    Q_st[i].buffer(len(phases)).buffer(len(wait_times)).buffer(len(init_states)).average().save(
                        f"Q{i + 1}"
                    )


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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in ds_fit and the fitted results in fit_results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
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
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit.grid_location."""
    fig_phase = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    fig_normalized = plot_normalized_phase(node.namespace["qubits"], node.results["ds_fit"])
    fig_contrast = plot_contrast(node.namespace["qubits"], node.results["ds_fit"])
    # Store the generated figures
    node.results["figures"] = {
        "ramsey_phase": fig_phase,
        "ramsey_phase_normalized": fig_normalized,
        "ramsey_contrast": fig_contrast,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            depletion_time = node.results["fit_results"][q.name]["depletion_time_ns"]
            # Rounded up to the 4 ns clock cycle
            q.resonator.depletion_time = max(
                int(4 * np.ceil(depletion_time / 4)), node.parameters.min_depletion_time_in_ns
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
