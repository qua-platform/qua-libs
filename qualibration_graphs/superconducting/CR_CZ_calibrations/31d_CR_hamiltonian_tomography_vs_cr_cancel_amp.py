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
from calibration_utils.cr_ham_tomo_cr_cancel_amp_scaling import (
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
description =  """
        Cross-Resonance Time Rabi
The sequence consists two consecutive pulse sequences with the qubit's thermal decay in between.
In the first sequence, we set the control qubit in |g> and play a rectangular cross-resonance pulse to
the target qubit; the cross-resonance pulse has a variable duration. In the second sequence, we initialize the control
qubit in |e> and play the variable duration cross-resonance pulse to the target qubit. Note that in
the second sequence after the cross-resonance pulse we send a x180_c pulse. With it, the target qubit starts
in |g> in both sequences when CR lenght -> zero.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Reference: A. D. Corcoles et al., Phys. Rev. A 87, 030301 (2013)

"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="31d_CR_hamiltonian_tomography_vs_cr_cancel_amp_scaling",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["qA1-A2", "qA3-A4"]
    node.parameters.use_state_discrimination = True

    node.parameters.wf_type = "square"
    node.parameters.cr_type = "direct+cancel+echo"
    node.parameters.cr_drive_amp_scaling = [0.89, 0.89] # None : setting None to use the amp from the config
    node.parameters.cr_drive_phase = [0.12, 0.12] # None : setting None to use the amp from the config
    node.parameters.cr_cancel_amp_scaling = [0.34, 0.34] # None : setting None to use the amp from the config
    node.parameters.cr_cancel_phase = [0.23, 0.23] # None : setting None to use the amp from the config


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
    cr_drive_amp_scaling = node.parameters.cr_drive_amp_scaling
    cr_drive_phase = node.parameters.cr_drive_phase
    cr_cancel_amp_scaling = node.parameters.cr_cancel_amp_scaling
    cr_cancel_phase = node.parameters.cr_cancel_phase

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns // 4,
        node.parameters.max_wait_time_in_ns // 4,
        node.parameters.time_step_in_ns // 4,
    )
    amp_scalings = np.arange(
        node.parameters.min_cr_drive_amp_scaling,
        node.parameters.max_cr_drive_amp_scaling,
        node.parameters.step_cr_drive_amp_scaling,
    )
    qst_basis = np.array([0, 1, 2])
    control_state = np.array([0, 1])

    if not node.parameters.use_state_discrimination:
        raise ValueError("use_state_discrimination must be True for Hamiltonian Tomography!")

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amp_scaling": xr.DataArray(amp_scalings, attrs={"long_name": "cr drive amplitude scaling"}),
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
        amp_scaling_qua = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(amp_scaling_qua, amp_scalings)):
                    with for_(*from_array(t, pulse_durations)):
                        with for_(c, 0, c < 3, c + 1):  # bases
                            with for_(s, 0, s < 2, s + 1):  # states
                            
                                for i, qp in multiplexed_qubit_pairs.items():
                                    qc, qt, cr, cr_elems = get_cr_elements(qp)

                                    # Reset the qubits to the ground state
                                    qp.qubit_control.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                        log_callable=node.log,
                                    )
                                    qp.qubit_target.reset(
                                        node.parameters.reset_type,
                                        node.parameters.simulate,
                                        log_callable=node.log,
                                    )

                                    # Prepare Qc at 0/1
                                    with if_(s == 1):
                                        qc.xy.play("x180")
                                        align(*cr_elems)

                                    # Play CR
                                    qp.apply("cr",
                                        cr_type=cr_type,
                                        wf_type=wf_type,
                                        cr_drive_amp_scaling=cr_drive_amp_scaling[i],
                                        cr_drive_phase=cr_drive_phase[i],
                                        cr_cancel_amp_scaling=amp_scaling_qua,
                                        cr_cancel_phase=cr_cancel_phase[i],
                                        cr_duration_clock_cycles=t,
                                    )
                                    align(*cr_elems)

                                    # QST on qt
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
                state_c_st[i].buffer(2).buffer(3).buffer(len(pulse_durations)).buffer(len(amp_scalings)).average().save(f"state_c{i}")
                state_t_st[i].buffer(2).buffer(3).buffer(len(pulse_durations)).buffer(len(amp_scalings)).average().save(f"state_t{i}")


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
        f"IQ_{qp.name}": fig
        for fig, qp in zip(figs_raw_fit, node.namespace["qubit_pairs"])
    }


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
        for i, qp in enumerate(node.namespace["qubit_pairs"]):
            if node.outcomes[qp.name] == "failed":
                continue

            # cr drive
            operation_c = qp.cross_resonance.operations[node.parameters.wf_type]
            operation_c.amplitude = node.parameters.cr_drive_amp_scaling[i] * operation_c.amplitude
            operation_c.axis_angle = node.parameters.cr_drive_phase[i] * 2 * np.pi
            # cr cancel 
            operation_t = qp.qubit_target.xy.operations[f"cr_{node.parameters.wf_type}_{qp.name}"]
            operation_t.amplitude = node.parameters.cr_cancel_amp_scaling[i] * operation_t.amplitude
            operation_t.axis_angle = node.parameters.cr_cancel_phase[i] * 2 * np.pi


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
