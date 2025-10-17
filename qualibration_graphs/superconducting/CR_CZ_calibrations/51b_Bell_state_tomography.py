# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.bell_state_tomography import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Description}
description = """
    Bell State Tomography
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="51b_Bell_state_tomography",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1-2", "q3-4"]
    # # node.parameters.simulate = True
    # # node.parameters.simulation_duration_ns = 6000
    # node.parameters.num_shots = 25
    # node.parameters.wf_type = "flattop"
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Update the readout power to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_qubit_pairs"] = []
    for qp in qubit_pairs:
        with tracked_updates(qp, auto_revert=False) as qp:
            pass
        node.namespace["tracked_qubit_pairs"].append(qp)

    n_shots = node.parameters.num_shots  # The number of averages
    state_discrimination = node.parameters.use_state_discrimination
    wf_type = node.parameters.wf_type

    assert node.parameters.use_state_discrimination, "this protocol requires to use state discriminations"

    # Control states
    tomo_axes = [0, 1, 2]

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "n_shots": xr.DataArray(np.arange(n_shots), attrs={"long_name": "number of shots"}),
        "tomo_axis_control": xr.DataArray(tomo_axes, attrs={"long_name": "tomography axis for control"}),
        "tomo_axis_target": xr.DataArray(tomo_axes, attrs={"long_name": "tomography axis for target"}),
    }

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        tomo_axis_control = declare(int)
        tomo_axis_target = declare(int)

        # Reset explicitly
        reset_global_phase()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_shots, n + 1):
                save(n, n_st)

                with for_(tomo_axis_control, 0, tomo_axis_control < 3, tomo_axis_control + 1):
                    with for_(tomo_axis_target, 0, tomo_axis_target < 3, tomo_axis_target + 1):
                        # Reset the qubits to the ground state
                        for i, qp in multiplexed_qubit_pairs.items():
                            qc = qp.qubit_control
                            qt = qp.qubit_target
                            zz = qp.zz_drive
                            zz_elems = [zz.name, qc.xy.name, qt.xy.name, qt.xy_detuned.name]

                            # Reset the qubits to the ground state
                            qc.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
                            qt.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)

                            align(*zz_elems)

                            # Play Hadamard
                            qc.xy.play("x180")
                            qc.xy.play("y90")

                            align(*zz_elems)

                            # Play CNOT
                            qt.xy.play("x180")
                            qt.xy.play("y90")
                            align(*zz_elems)
                            qp.apply("stark_cz")
                            align(*zz_elems)
                            qt.xy.play("x180")
                            qt.xy.play("y90")

                            align(*zz_elems)

                            # Tomography gates
                            with if_(tomo_axis_control == 0):  # X axis
                                qc.xy.play("-y90")
                            with elif_(tomo_axis_control == 1):  # Y axis
                                qc.xy.play("x90")
                            with if_(tomo_axis_target == 0):  # X axis
                                qt.xy.play("-y90")
                            with elif_(tomo_axis_target == 1):  # Y axis
                                qt.xy.play("x90")

                            align(*zz_elems)

                            # Measure the state of the resonators
                            qc.readout_state(state_c[i])
                            qt.readout_state(state_t[i])
                            save(state_c[i], state_c_st[i])
                            save(state_t[i], state_t_st[i])

                            # Reset the frame of the qubits in order not to accumulate rotations
                            reset_frame(zz.name)
                            reset_frame(qt.xy_detuned.name)
                            reset_frame(qt.xy.name)
                            reset_frame(qt.xy.name)

                            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                            qc.resonator.wait(qc.resonator.depletion_time // 4)
                            qt.resonator.wait(qt.resonator.depletion_time // 4)

        with stream_processing():
            n_st.save("n")
            for i, qp in enumerate(qubit_pairs):
                state_c_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_c{i + 1}")
                state_t_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_t{i + 1}")


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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    figs_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        f"raw_fit_{qp.name}": fig for fig, qp in zip(figs_raw_fit, node.namespace["qubit_pairs"])
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_qubit_pair in node.namespace.get("tracked_qubit_pairs", []):
        tracked_qubit_pair.zz_drive.revert_changes()
        tracked_qubit_pair.qubit_control.revert_changes()
        tracked_qubit_pair.qubit_target.revert_changes()

    with node.record_state_updates():
        for i, qp in enumerate(node.namespace["qubit_pairs"]):
            if node.outcomes[qp.name] == "failed":
                continue
            print("here")
            pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
