# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.psb_search_ramp_timing import Parameters
from calibration_utils.common_utils.experiment import get_sensors
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Fixed Detuning vs Ramp Duration
The goal of this sequence is to characterise the ramp duration of the PSB readout macro.
To do so, the following triangle in voltage space (empty - random initialization - ramp to measurement point - measurement) is applied using OPX
channels on the fast lines of the bias-tees. The ramp duration is then varied.

The OPX measures the response via RF reflectometry or DC current sensing during the readout window
(last segment of the triangle). The sequence is repeated several time in order to find the optimal average ramp_duration.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage on average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time, unless a compensation pulse is played.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.
    - Having identified the suitable PSB measurement point through nodes 06a, 06b, and 06c.

State update:
    - The optimal ramp duration to perform PSB readout.
"""


node = QualibrationNode[Parameters, Quam](
    name="06o_DEPRECATED_PSB_search_opx_ramp_timing",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.quantum_dot_pair_names = ["virtual_dot_1_virtual_dot_2_pair"]
    # node.parameters.num_shots = 10
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    dot_pair_objects = [
        node.machine.quantum_dot_pairs[name]
        for name in node.parameters.quantum_dot_pair_names
    ]

    node.namespace["dot_pairs"] = dot_pair_objects

    ramp_min = int(node.parameters.ramp_duration_min)
    ramp_max = int(node.parameters.ramp_duration_max)
    ramp_step = int(node.parameters.ramp_duration_step)
    if ramp_min % 4 != 0 or ramp_max % 4 != 0 or ramp_step % 4 != 0:
        raise ValueError(
            f"Ramp settings must be divisible by 4. Received ramp_duration_min: {ramp_min}, ramp_duration_max: {ramp_max}, ramp_duration_step: {ramp_step}"
        )

    ramp_duration_array = np.arange(ramp_min, ramp_max, ramp_step, dtype=int)

    # We can change the detuning value as a tracked change, which the user can accept to change at the end.
    # normal tracked_updates don't seem to be able to track through it, so create a light manual tracker
    node.namespace["tracked_original_detunings"] = {}
    for dot_pair in dot_pair_objects:
        if node.parameters.detuning is not None:

            # Identify gateset. Should be the same for all dot_pairs
            dot_pair_gate_set = dot_pair.voltage_sequence.gate_set

            # Get point
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair_gate_set.get_macros()[point_name]

            # Save the original detuning
            node.namespace["tracked_original_detunings"][dot_pair.name] = (
                point.voltages.get(dot_pair.detuning_axis_name)
            )

            # Apply the change for the node
            point.voltages[dot_pair.detuning_axis_name] = node.parameters.detuning

    # The swept 'axes'. Do we do a full 2D scan here?
    node.namespace["sweep_axes"] = {
        "ramp_durations": xr.DataArray(
            ramp_duration_array, attrs={"long_name": "ramp duration", "units": "ns"}
        ),
        "quantum_dot_pair": xr.DataArray([pair.name for pair in dot_pair_objects]),
    }
    with program() as prog:
        n = declare(int)
        n_st = declare_stream()
        ramp_duration = declare(int)

        I = {pair.name: declare(fixed) for pair in dot_pair_objects}
        Q = {pair.name: declare(fixed) for pair in dot_pair_objects}
        I_st = {pair.name: declare_stream() for pair in dot_pair_objects}
        Q_st = {pair.name: declare_stream() for pair in dot_pair_objects}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            # Perform them all sequentially for now. Can add footprint batching later
            for dot_pair in dot_pair_objects:
                with for_(*from_array(ramp_duration, ramp_duration_array)):

                    # Potentially a gap here in the programme? Add artificial wait?
                    # wait(100)
                    # ---------------------------------------------------------
                    # Step 1a: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    dot_pair.macros["empty"].apply()

                    # ---------------------------------------------------------
                    # Step 2: Initialize - load electron into dots (fixed duration)
                    # ---------------------------------------------------------
                    dot_pair.macros[node.parameters.initialization_macro].apply()

                    # ---------------------------------------------------------
                    # Step 3: Measure
                    # ---------------------------------------------------------

                    align()

                    # Measure point is a tracked change, so this should respect the user's detuning parameter input
                    # Ramp duration is overridden temporarily for each measure macro.
                    I[dot_pair.name], Q[dot_pair.name] = dot_pair.measure(
                        ramp_duration=ramp_duration
                    )
                    save(I[dot_pair.name], I_st[dot_pair.name])
                    save(Q[dot_pair.name], Q_st[dot_pair.name])

                    align()
                    # Apply the compensation pulse via the voltage sequence
                    dot_pair.voltage_sequence.ramp_to_zero()

        with stream_processing():
            n_st.save("n")

            for pair in dot_pair_objects:
                I_st[pair.name].buffer(len(ramp_duration_array)).average().save(
                    f"I_{pair.name}"
                )
                Q_st[pair.name].buffer(len(ramp_duration_array)).average().save(
                    f"Q_{pair.name}"
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
    # Get the active sensors from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    for dot_pair in node.namespace["dot_pairs"]:
        if dot_pair.name in node.namespace.get("tracked_original_detunings", {}):
            # Gate set
            dot_pair_gate_set = dot_pair.voltage_sequence.gate_set

            # Get point
            point_name = dot_pair._create_point_name("measure")
            point = dot_pair_gate_set.get_macros()[point_name]

            # Revert change
            point.voltages[dot_pair.detuning_axis_name] = node.namespace[
                "tracked_original_detunings"
            ][dot_pair.name]

    with node.record_state_updates():
        # This is a characterization measurement and typically does not update state parameters.
        # If needed in the future, the identified PSB region coordinates could be stored.
        # Example of potential state update (commented out):
        # for qubit_pair in node.namespace["qubit_pairs"]:
        #     if not node.results["fit_results"][qubit_pair.name]["success"]:
        #         continue
        #     # Update PSB region coordinates if needed
        # TODO: update threshold and rotation angle here?
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


    
