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
from calibration_utils.stark_cz_vs_correction_phase import (
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
    CZ CALIB CZ PULSE VS CORRECTION PHASE
Fine-tune virtual-Z correction phases φ_ZI and φ_IZ that compensate local phase shifts after a Stark-CZ gate.  
For each correction term (ZI or IZ), prepare one qubit in |0⟩/|1⟩ and place the other on the equator (Y/2),  
apply the Stark-CZ, undo with −Y/2, and measure population oscillations as a function of correction phase.

Prerequisites:
    - Fixed Stark-CZ duration, drive frequency, amplitudes, and relative phase.
    - Calibrated single-qubit gates and state discrimination.

State Update:
    - Fit the <Z> vs. correction phase for both ZI and IZ experiments and
      pick phaes to realize CZ such that |+,0⟩→|0,0⟩ and |+,1⟩→|1,1⟩ (control case), 
    - Update
          qp.macros.stark_cz.qc_correction_phase
          qp.macros.stark_cz.qt_correction_phase
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="42a_CZ_calib_cz_pulse_vs_correction_phase",  # Name should be unique
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
    # # node.parameters.simulate = True
    # # node.parameters.simulation_duration_ns = 6000
    # node.parameters.num_shots = 3
    # node.parameters.wf_type = "flattop"
    # node.parameters.qc_correction_phase_2pi = [None, None]
    # node.parameters.qt_correction_phase_2pi = [None, None]
    # node.parameters.zz_drive_relative_phase_2pi = [None, None]
    # node.parameters.zz_drive_control_amp_scaling = [None, None]
    # node.parameters.zz_drive_target_amp_scaling = [None, None]
    # node.parameters.min_cz_pulse_correction_phase_2pi = 0.0
    # node.parameters.max_cz_pulse_correction_phase_2pi = +1.1
    # node.parameters.step_cz_pulse_correction_phase_2pi = 0.1
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

    qc_correction_phases = broadcast_param_to_list(node.parameters.qc_correction_phase_2pi, num_qubit_pairs)
    qt_correction_phases = broadcast_param_to_list(node.parameters.qt_correction_phase_2pi, num_qubit_pairs)
    zz_relative_phases = broadcast_param_to_list(node.parameters.zz_drive_relative_phase_2pi, num_qubit_pairs)
    zz_control_amp_scalings = broadcast_param_to_list(node.parameters.zz_drive_control_amp_scaling, num_qubit_pairs)
    zz_target_amp_scalings = broadcast_param_to_list(node.parameters.zz_drive_target_amp_scaling, num_qubit_pairs)

    corr_phases = np.arange(
        node.parameters.min_cz_pulse_correction_phase_2pi,
        node.parameters.max_cz_pulse_correction_phase_2pi,
        node.parameters.step_cz_pulse_correction_phase_2pi,
    )

    # Calibrate ZI or IZ
    correction_target_terms = ["ZI", "IZ"]  # (c, t)

    # Control states
    control_states = np.array([0, 1])

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "correction_target_term": xr.DataArray(
            correction_target_terms, attrs={"long_name": "Pauli Z term (control vs target) for Stark-ZZ correction"}
        ),
        "correction_phase": xr.DataArray(corr_phases, attrs={"long_name": "stark zz drive relative phase"}),
        "control_state": xr.DataArray(control_states, attrs={"long_name": "control state"}),
    }

    with program() as node.namespace["qua_program"]:
        _, _, _, _, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        ph = declare(float)
        s = declare(int)

        # Reset explicitly
        reset_global_phase()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                for corr_term in correction_target_terms:
                    assert corr_term in ["ZI", "IZ"]

                    with for_(*from_array(ph, corr_phases)):
                        with for_(s, 0, s < 2, s + 1):  # states 0:g or 1:e
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

                                if corr_term == "ZI":
                                    # Ramsey the CONTROL; prepare TARGET in |0/1>
                                    q1 = qt  # prepared 0/1
                                    q2 = qc  # Ramsey (y90 ... -y90)
                                elif corr_term == "IZ":
                                    # Ramsey the TARGET; prepare CONTROL in |0/1>
                                    q1 = qc  # prepared 0/1
                                    q2 = qt  # Ramsey

                                # Prepare Q1 at 0/1 and Play pi/2 for Q2
                                with if_(s == 1):
                                    q1.xy.play("x180")
                                    q2.xy.play("y90")
                                with if_(s == 0):
                                    q1.xy.wait(q1.xy.operations["x180"].length * u.ns)
                                    q2.xy.play("y90")
                                align(*zz_elems)

                                # Play CZ
                                qp.apply(
                                    "stark_cz",
                                    wf_type=wf_type,
                                    zz_control_amp_scaling=zz_control_amp_scalings[i],
                                    zz_target_amp_scaling=zz_target_amp_scalings[i],
                                    zz_relative_phase=zz_relative_phases[i],
                                    qc_correction_phase=ph if corr_term == "ZI" else qc_correction_phases[i],
                                    qt_correction_phase=ph if corr_term == "IZ" else qt_correction_phases[i],
                                )
                                align(*zz_elems)

                                # Play pi/2 for Q2
                                q2.xy.play("-y90")
                                align(qc.resonator.name, qt.resonator.name, *zz_elems)

                                # Measure the state of the resonators
                                qc.readout_state(state_c[i])
                                qt.readout_state(state_t[i])
                                save(state_c[i], state_c_st[i])
                                save(state_t[i], state_t_st[i])
                                align(qc.resonator.name, qt.resonator.name, *zz_elems)

                                # Reset the frame of the qubits in order not to accumulate rotations
                                reset_frame(zz.name)
                                reset_frame(qt.xy_detuned.name)
                                reset_frame(qc.xy.name)
                                reset_frame(qt.xy.name)

                                # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                                qc.resonator.wait(qc.resonator.depletion_time // 4)
                                qt.resonator.wait(qt.resonator.depletion_time // 4)

        with stream_processing():
            n_st.save("n")
            for i, qp in enumerate(qubit_pairs):
                state_c_st[i].buffer(2).buffer(len(corr_phases)).buffer(len(correction_target_terms)).average().save(
                    f"state_c{i + 1}"
                )
                state_t_st[i].buffer(2).buffer(len(corr_phases)).buffer(len(correction_target_terms)).average().save(
                    f"state_t{i + 1}"
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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}

    from pathlib import Path

    # Visualize and save the waveform report
    wf_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))


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

            fit_result = node.results["fit_results"][qp.name]

            qp.macros.stark_cz.qc_correction_phase = fit_result["best_correction_phase_zi_c"]
            qp.macros.stark_cz.qt_correction_phase = fit_result["best_correction_phase_iz_t"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
