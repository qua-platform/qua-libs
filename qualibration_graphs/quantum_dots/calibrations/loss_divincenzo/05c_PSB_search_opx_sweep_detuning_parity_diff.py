# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.psb_search_sweep_detuning_parity_diff import Parameters
from qualang_tools.loops import from_array
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Sweep Detuning with Parity Difference
The goal of this sequence is to find the Pauli Spin Blockade (PSB) region.
To do so, the following triangle in voltage space (empty - random initialization - measurement) is applied using OPX
channels on the fast lines of the bias-tees while sweeping the "measure" voltage point along the detuning axis.

The OPX measures the response via RF reflectometry or DC current sensing during the readout window
(last segment of the triangle). A single-point averaging is performed and the data is extracted while
the program is running to display the results.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage on average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time, unless a compensation pulse is played.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.

State update:
    TODO: how to update the PSB region coordinates and optimal detuning for a given qubit/dot?
"""


node = QualibrationNode[Parameters, Quam](
    name="05c_PSB_search_opx_sweep_detuning_parity_diff", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.quantum_dot_pair_names = ["virtual_dot_1_virtual_dot_2_pair"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    dot_pair_objects = [node.machine.quantum_dot_pairs[name] for name in node.parameters.quantum_dot_pair_names]

    node.namespace["dot_pairs"] = dot_pair_objects

    detuning_min = node.parameters.detuning_min
    detuning_max = node.parameters.detuning_max
    detuning_points = node.parameters.detuning_points
    detuning_step = (detuning_max - detuning_min) / detuning_points

    detuning_array = np.arange(detuning_min, detuning_max, detuning_step)

    node.namespace["sweep_axes"] = {
        "quantum_dot_pair": xr.DataArray([pair.name for pair in dot_pair_objects]),
        "detuning": xr.DataArray(detuning_array, attrs={"long_name": "voltage", "units": "V"}),
    }
    with program() as prog:
        n = declare(int)
        n_st = declare_stream()

        detuning = declare(fixed)

        # Parity variables and streams keyed by dot pair name.
        # dot_pair.measure() returns a bool; Cast.to_int converts it so we
        # can average the result in stream processing.
        p1 = {pair.name: declare(int) for pair in dot_pair_objects}
        p2 = {pair.name: declare(int) for pair in dot_pair_objects}
        p1_st = {pair.name: declare_stream() for pair in dot_pair_objects}
        p2_st = {pair.name: declare_stream() for pair in dot_pair_objects}
        pdiff_st = {pair.name: declare_stream() for pair in dot_pair_objects}

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            for dot_pair in dot_pair_objects:
                with from_array(detuning, detuning_array):
                    # ---------------------------------------------------------
                    # Step 1: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    dot_pair.empty()

                    align()

                    # ---------------------------------------------------------
                    # Step 2: Measure initial parity (p1) — reference state
                    # ---------------------------------------------------------
                    assign(p1[dot_pair.name], Cast.to_int(dot_pair.measure()))

                    # ---------------------------------------------------------
                    # Step 3: Initialize - load electron into dots
                    # ---------------------------------------------------------
                    dot_pair.initialize()

                    # dot_pair.ramp_to_detuning(detuning, ramp_duration=node.parameters.ramp_duration)
                    # align()

                    # ---------------------------------------------------------
                    # Step 4: Measure final parity (p2)
                    # ---------------------------------------------------------

                    # After node 05b, the measure macro will contain the step to the PSB point and the measurement.
                    # Therefore, no need to ramp to detuning or align.
                    assign(p2[dot_pair.name], Cast.to_int(dot_pair.measure()))

                    align()

                    # ---------------------------------------------------------
                    # Step 5: Apply compensation pulse to reset DC bias
                    # ---------------------------------------------------------
                    dot_pair.voltage_sequence.apply_compensation_pulse()

                    # ---------------------------------------------------------
                    # Save results
                    # ---------------------------------------------------------
                    save(p1[dot_pair.name], p1_st[dot_pair.name])
                    save(p2[dot_pair.name], p2_st[dot_pair.name])

                    with if_(p1[dot_pair.name] == p2[dot_pair.name]):
                        save(0, pdiff_st[dot_pair.name])
                    with else_():
                        save(1, pdiff_st[dot_pair.name])

        with stream_processing():
            n_st.save("n")
            for pair in dot_pair_objects:
                p1_st[pair.name].buffer(len(detuning_array)).average().save(f"p1_{pair.name}")
                p2_st[pair.name].buffer(len(detuning_array)).average().save(f"p2_{pair.name}")
                pdiff_st[pair.name].buffer(len(detuning_array)).average().save(f"pdiff_{pair.name}")


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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
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
    # Restore dot pair objects from the loaded parameters
    node.namespace["dot_pairs"] = [
        node.machine.quantum_dot_pairs[name] for name in node.parameters.quantum_dot_pair_names
    ]


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

    with node.record_state_updates():
        # This is a characterization measurement and typically does not update state parameters.
        # If needed in the future, the identified PSB region coordinates and optimal detuning could be stored.
        # Example of potential state update (commented out):
        # for qubit_pair in node.namespace["qubit_pairs"]:
        #     if not node.results["fit_results"][qubit_pair.name]["success"]:
        #         continue
        #     # Update PSB region coordinates and optimal detuning if needed
        # TODO: how to update the PSB region coordinates and optimal detuning for a given qd pair?
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
