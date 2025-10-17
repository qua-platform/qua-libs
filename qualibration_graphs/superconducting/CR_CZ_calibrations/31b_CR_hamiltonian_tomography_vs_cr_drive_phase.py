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
from calibration_utils.cr_ham_tomo_cr_drive_phase import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.cr_utils import *
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Description}
description = """
        Cross-Resonance Time Rabi with Quantum State Tomography (QST) + drive phase optimization
This experiment measures the target qubit response under a variable-length cross-resonance (CR) drive, 
with quantum state tomography for both control states. The sequence has two parts, separated by qubit relaxation:
1. Control qubit prepared in |g>, apply a CR pulse of variable duration to the target.  
2. Control qubit prepared in |e>, apply the same CR pulse to the target, then a corrective x180 on the control.  
   (Ensures the target effectively starts in |g> at zero CR length in both cases.)
QST is performed by projecting the target onto X, Y, and Z bases before measurement. We can then calculate the
interaction strengths of ["IX", "IY", "IZ", "ZX", "ZY", "ZZ"] from the evolution.

The optimal drive phase is chosen such that the ZX component is maximized and ZY is zero

Prerequisites:
    - Resonator spectroscopy (to locate resonator frequency).
    - Qubit spectroscopy, Rabi chevron, and power Rabi (to calibrate the qubit π pulse and update the config).
    - State discrimination
    - (Optional) Readout calibration (frequency, amplitude, duration optimization, IQ blobs) for improved SNR.

Reference: A. D. Córcoles et al., Phys. Rev. A 87, 030301(R) (2013).

"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="31b_CR_hamiltonian_tomography_vs_cr_drive_phase",  # Name should be unique
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
    # node.parameters.use_state_discrimination = True
    # node.parameters.wf_type = "square"
    # node.parameters.cr_type = "direct+cancel+echo"
    # node.parameters.cr_drive_amp_scaling = [0.89, 0.89]  # None : setting None to use the amp from the config
    # node.parameters.cr_drive_phase = [0.12, 0.12]  # None : setting None to use the amp from the config
    # node.parameters.cr_cancel_amp_scaling = [0.34, 0.34]  # None : setting None to use the amp from the config
    # node.parameters.cr_cancel_phase = [0.23, 0.23]  # None : setting None to use the amp from the config
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

    n_avg = node.parameters.num_shots  # The number of averages
    state_discrimination = node.parameters.use_state_discrimination
    wf_type = node.parameters.wf_type
    cr_type = node.parameters.cr_type
    cr_drive_amp_scaling = broadcast_param_to_list(node.parameters.cr_drive_amp_scaling, num_qubit_pairs)
    cr_cancel_amp_scaling = broadcast_param_to_list(node.parameters.cr_cancel_amp_scaling, num_qubit_pairs)
    cr_cancel_phase = broadcast_param_to_list(node.parameters.cr_cancel_phase, num_qubit_pairs)

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.time_step_in_ns // 4,
    )
    phases = np.arange(
        node.parameters.min_cr_drive_phase,
        node.parameters.max_cr_drive_phase,
        node.parameters.step_cr_drive_phase,
    )
    qst_basis = np.array([0, 1, 2])
    control_state = np.array([0, 1])

    if not node.parameters.use_state_discrimination:
        raise ValueError("use_state_discrimination must be True for Hamiltonian Tomography!")

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "phase": xr.DataArray(phases, attrs={"long_name": "cr drive phase"}),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
        "qst_basis": xr.DataArray(qst_basis, attrs={"long_name": "qst basis"}),
        "control_state": xr.DataArray(control_state, attrs={"long_name": "control state"}),
    }

    with program() as node.namespace["qua_program"]:
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        t = declare(int)
        s = declare(int)  # QUA variable for the control state
        c = declare(int)  # QUA variable for the projection index in QST
        phase_qua = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(phase_qua, phases)):
                    with for_(*from_array(t, pulse_durations)):
                        with for_(c, 0, c < 3, c + 1):  # bases
                            with for_(s, 0, s < 2, s + 1):  # states
                                for i, qp in multiplexed_qubit_pairs.items():
                                    qc, qt, cr, cr_elems = get_cr_elements(qp)

                                    # Reset the qubits to the ground state
                                    qc.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                        log_callable=node.log,
                                    )
                                    qt.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                        log_callable=node.log,
                                    )
                                    align(*cr_elems)

                                    # Prepare Qc at 0/1
                                    with if_(s == 1):
                                        qc.xy.play("x180")
                                        align(*cr_elems)

                                    # Play CR
                                    qp.apply(
                                        "cr",
                                        cr_type=cr_type,
                                        wf_type=wf_type,
                                        cr_drive_amp_scaling=cr_drive_amp_scaling[i],
                                        cr_drive_phase=phase_qua,
                                        cr_cancel_amp_scaling=cr_cancel_amp_scaling[i],
                                        cr_cancel_phase=cr_cancel_phase[i],
                                        cr_duration_clock_cycles=t,
                                    )

                                    # QST on Qt
                                    align(*cr_elems)
                                    with switch_(c):
                                        with case_(0):  # projection along X
                                            qc.xy.play("-y90")
                                            qt.xy.play("-y90")
                                        with case_(1):  # projection along Y
                                            qc.xy.play("x90")
                                            qt.xy.play("x90")
                                        with case_(2):  # projection along Z
                                            qc.xy.wait(qc.xy.operations["x180"].length * u.ns)
                                            qt.xy.wait(qt.xy.operations["x180"].length * u.ns)
                                    align(*cr_elems, qc.resonator.name, qt.resonator.name)

                                    # Measure the state of the resonators
                                    qc.readout_state(state_c[i])
                                    qt.readout_state(state_t[i])
                                    save(state_c[i], state_c_st[i])
                                    save(state_t[i], state_t_st[i])

                                    # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                                    qc.resonator.wait(qc.resonator.depletion_time * u.ns)
                                    qt.resonator.wait(qt.resonator.depletion_time * u.ns)

        with stream_processing():
            n_st.save("n")
            for i, qp in enumerate(qubit_pairs):
                state_c_st[i].buffer(len(control_state)).buffer(len(qst_basis)).buffer(len(pulse_durations)).buffer(
                    len(phases)
                ).average().save(f"state_c{i + 1}")
                state_t_st[i].buffer(len(control_state)).buffer(len(qst_basis)).buffer(len(pulse_durations)).buffer(
                    len(phases)
                ).average().save(f"state_t{i + 1}")


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
    node.results["figures"] = {f"IQ_{qp.name}": fig for fig, qp in zip(figs_raw_fit, node.namespace["qubit_pairs"])}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_qubit_pair in node.namespace.get("tracked_qubit_pairs", []):
        tracked_qubit_pair.cross_resonance.revert_changes()
        tracked_qubit_pair.qubit_control.revert_changes()
        tracked_qubit_pair.qubit_target.revert_changes()

    with node.record_state_updates():
        for multiplexed_qubit_pairs in node.namespace["qubit_pairs"].batch():
            for i, qp in multiplexed_qubit_pairs.items():
                if node.outcomes[qp.name] == "failed":
                    continue

                # # cr drive
                # operation_c = qp.cross_resonance.operations[node.parameters.wf_type]
                # operation_c.axis_angle = node.parameters.cr_drive_phase[i] * 2 * np.pi
                pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
