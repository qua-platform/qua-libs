# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.xyx_delay import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    baked_flux_xy_segments,
)
from qm.qua import *
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
        XY-Z DELAY CALIBRATION
This calibration determines the relative delay between the microwave XY control line (e.g. x180) and the Z line (flux)
for each qubit. The goal is to ensure that the flux pulse reaches the qubit at the same time as the XY drive and correct
for any latency differences between the two control lines. By applying the XY and Z pulses simultaneously, the qubit
frequency is shifted during the XY rotation, which can lead to incomplete rotations if the pulses are not properly aligned
in time. By inserting variable leading / trailing zeros around the fixed XY and Z pulse shapes, the sequence scans the
relative timing at 1ns resolution and measures the qubit state for two initial preparations
(|e> created by an initial x180 and |g> with identity). The resulting population (or I/Q) versus relative timing is
fitted to extract the optimal flux delay that best aligns the Z pulse with the XY drive (see Chen PhD thesis, p.108).

Prerequisites:
    - Having calibrated a pi-pulse (x180) for the given qubit.
    - (Optional) State discrimination calibrated if use_state_discrimination = True.

State update:
    - Adds extracted flux delay (fit_results[qubit]["flux_delay"]) to q.z.opx_output.delay per successful qubit.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="16_XY_Z_delay",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Create the sweep axes and generate the QUA program from the pulse sequence and the
    node parameters.
    """
    # Class containing tools to help handling units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the config from the machine
    node.namespace["qubits"] = qubits = get_qubits(
        node
    )  # Get active qubit objects and store them in the node namespace
    num_qubits = len(qubits)  # Count how many qubits are involved in this calibration

    config = node.machine.generate_config()  # Generate the QM configuration from the current machine state

    n_avg = node.parameters.num_shots  # Number of averages (used in the QUA averaging loop)
    total_zeros = 2 * node.parameters.zeros_before_after_pulse  # Total number of delay positions scanned (± range)

    flux_waveform_list = {}  # Will store per-qubit flux pulse sample lists prior to baking

    for qubit in qubits:
        flux_waveform_list[qubit.xy.name] = [node.parameters.z_pulse_amplitude] * qubit.xy.operations["x180"].length

    delay_segments = {}
    # Baked flux pulse segments with 1ns resolution

    for i, qubit in enumerate(qubits):
        delay_segments[qubit.xy.name] = baked_flux_xy_segments(
            config,
            flux_waveform_list[qubit.xy.name],
            qubit,
            node.parameters.zeros_before_after_pulse,
        )
        print(f"Baked waveform for {qubit.xy.name}")

    node.namespace["config"] = config
    relative_time = np.arange(
        -node.parameters.zeros_before_after_pulse, node.parameters.zeros_before_after_pulse, 1
    )  # x-axis for plotting - Must be in ns.
    number_of_segments = 2 * node.parameters.zeros_before_after_pulse

    n_avg = node.parameters.num_shots  # The number of averages

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "init_state": xr.DataArray(["e", "g"], attrs={"long_name": "initial qubit state", "units": "a.u."}),
        "relative_time": xr.DataArray(
            relative_time, attrs={"long_name": "relative delay between pulses", "units": "ns"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        segment = declare(int)  # QUA variable for the flux pulse segment index
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        npi = declare(int)  # QUA variable for the number of qubit pulses
        count = declare(int)  # QUA variable for counting the qubit pulses

        # --- Batch over qubits (allows time-multiplexed execution respecting hardware constraints)
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()  # Ensure all elements start aligned before the averaging loop

            # --- Averaging loop
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # Save current average index for live progress

                # --- Initial state preparation loop: prepare |e> (via x180) and |g> (idle) for contrast
                for init_state in ["e", "g"]:
                    # --- Relative delay scan loop (index over baked XY+Z aligned segments)
                    with for_(segment, 0, segment < number_of_segments, segment + 1):

                        # 1. Reset qubits to ground state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            qubit.align()

                            # 2. State preparation: excited (x180) or ground (wait same duration)
                            if init_state == "e":
                                qubit.xy.play("x180")
                            elif init_state == "g":
                                qubit.xy.wait(qubit.xy.operations["x180"].length)

                            qubit.align()
                            # 3. Optional coarse pre-wait (accounts for leading padding before fine scan)
                            qubit.wait(node.parameters.zeros_before_after_pulse // 4)
                            # 4. Apply baked XY+Z segment with specific relative shift
                            with switch_(segment):
                                for j in range(0, number_of_segments):
                                    with case_(j):
                                        delay_segments[qubit.xy.name][j].run()

                            qubit.align()
                            # 5. Measurement (state discrimination or I/Q acquisition)
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

            align()  # Final alignment after batch completes

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(number_of_segments).buffer(2).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(number_of_segments).buffer(2).average().save(f"I{i + 1}")
                    Q_st[i].buffer(number_of_segments).buffer(2).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.namespace["config"]
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, node.namespace["config"], timeout=node.parameters.timeout) as qm:
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
    node.namespace["qubits"] = get_qubits(node)


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
        "delay_scan": fig_raw_fit,
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
                # Update the qubit flux delay
                q.z.opx_output.delay += int(node.results["fit_results"][q.name]["flux_delay"])


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
