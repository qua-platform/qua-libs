# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_unitary_meadd import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_floquet,
    plot_meadd_phi,
    plot_meadd_theta,
    plot_process_matrix,
    process_raw_dataset,
    require_prerequisites,
)
from calibration_utils.cz_unitary_meadd.circuits import (
    EXP_FLOQUET,
    EXP_PHI,
    EXP_THETA_XX,
    EXP_THETA_YX,
    PREP_0P,
    PREP_10,
    PREP_P0,
    RO_XODD,
    RO_XX,
    RO_YODD,
    RO_YY,
    build_circuit_table,
)
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        CZ UNITARY RECONSTRUCTION (MEADD + FLOQUET)
Measures the five angles of the CZ gate model (conditional phase ϕ, swap angle θ, swap phase χ, and the Z phases
γ and ζ) in a single run. All circuits of the three sub-experiments are interleaved inside the same shot loop,
so slow drifts affect them equally:

    - MEADD-ϕ: repeat (CZ, then X⊗X) from |0+⟩ and |+0⟩; the determinant of the odd-parity matrix grows as 2nϕ.
    - MEADD-θ: from |10⟩, repeat (CZ without virtual-Z corrections, then X⊗X or Y⊗X); the odd-parity Bloch vector,
      read with Z and Bell-basis readouts, rotates by 4θcosχ and 4θsinχ per CZ pair.
    - Floquet: repeat the CZ alone from |0+⟩ and |+0⟩; the determinant gives γ and the |10⟩ eigenphase gives ζ.

Control = L ("top wire") and target = R in the gate model; the signs of ζ and χ depend on this mapping.
Methods: J. A. Gross et al., arXiv:2404.12550 (MEADD); F. Arute et al., arXiv:2010.07965 (Floquet).

Prerequisites:
    - Calibrated single-qubit gates (x180, y180, x90, y90, -y90) and readout thresholds on both qubits.
    - A calibrated CZ macro (conditional phase within a few hundred mrad of π).
    - The pair's readout confusion matrix (node 35) if use_readout_mitigation is True.

State update (if the fit succeeded):
    - phase_shift_control and phase_shift_target of the CZ macro, set to virtual-Z corrections that zero ζ and set
      γ to the target chosen by correction_target. ϕ and θ cannot be changed by single-qubit Z rotations, so the
      corrected gate reaches the fidelity reported as "after suggested corrections".
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="40_cz_unitary_MEADD",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1-q2"]
    pass


# %% {Circuit_helpers}
# Single-qubit pulses (control, target) of each state preparation
PREPARATION_PULSES = {
    PREP_P0: ("y90", None),
    PREP_0P: (None, "y90"),
    PREP_10: ("x180", None),
}


def play_on_pair(qp, control_pulse, target_pulse):
    """Play one single-qubit pulse on each qubit of the pair (None = no pulse)."""
    if control_pulse is not None:
        qp.qubit_control.xy.play(control_pulse)
    if target_pulse is not None:
        qp.qubit_target.xy.play(target_pulse)


def play_cz(qp, operation, with_corrections):
    """Play the CZ macro. Without corrections, the pulses are identical but the virtual-Z phases are zero."""
    if with_corrections:
        qp.macros[operation].apply()
    else:
        qp.macros[operation].apply(phase_shift_control=0.0, phase_shift_target=0.0)


def play_cycle(qp, operation, exp):
    """One iteration of the repeated block of sub-experiment exp (a Python int)."""
    play_cz(qp, operation, with_corrections=exp in (EXP_PHI, EXP_FLOQUET))
    if exp in (EXP_PHI, EXP_THETA_XX):
        play_on_pair(qp, "x180", "x180")
    elif exp == EXP_THETA_YX:
        play_on_pair(qp, "y180", "x180")


def prepare(qp, prep):
    """Prepare the pair in the state encoded by the QUA variable prep."""
    with switch_(prep):
        for code, pulses in PREPARATION_PULSES.items():
            with case_(code):
                play_on_pair(qp, *pulses)


def readout_rotation(qp, operation, ro):
    """Rotate the pair so that a Z measurement gives the basis encoded by the QUA variable ro (ZZ: nothing)."""
    with switch_(ro):
        with case_(RO_XX):
            play_on_pair(qp, "-y90", "-y90")
        with case_(RO_YY):
            play_on_pair(qp, "x90", "x90")
        # Bell-basis readout: a CNOT (control -> target) built from the CZ maps the odd-parity Bloch vector onto
        # the control qubit, and the target qubit then reads the parity (1 = odd).
        for ro_code, control_pulse in ((RO_XODD, "-y90"), (RO_YODD, "x90")):
            with case_(ro_code):
                qp.qubit_target.xy.play("-y90")
                play_cz(qp, operation, with_corrections=True)
                play_on_pair(qp, control_pulse, "y90")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    operation = node.parameters.operation
    require_prerequisites(qubit_pairs, operation, node.parameters.use_readout_mitigation)

    # The circuit table: one row (sub-experiment, preparation, readout, number of CZ gates) per circuit
    circuits = build_circuit_table(
        node.parameters.max_cz_meadd,
        node.parameters.step_cz_meadd,
        node.parameters.max_cz_floquet,
    )
    num_circuits = len(circuits)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "circuit": xr.DataArray(np.arange(num_circuits), attrs={"long_name": "circuit index"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        k = declare(int)
        exp, prep, ro, ncz = declare(int), declare(int), declare(int), declare(int)
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state_both = [declare(int) for _ in range(num_qubit_pairs)]
        state_control_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_target_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_both_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Run every circuit of every sub-experiment, one after the other
                with for_each_(
                    (exp, prep, ro, ncz),
                    (
                        [c["exp"] for c in circuits],
                        [c["prep"] for c in circuits],
                        [c["ro"] for c in circuits],
                        [c["ncz"] for c in circuits],
                    ),
                ):
                    # Reset the qubits and the frames (CZ corrections accumulate in the frame otherwise)
                    for qp in multiplexed_qubit_pairs.values():
                        qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                        qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                        reset_frame(qp.qubit_control.xy.name, qp.qubit_target.xy.name)
                    align()

                    for qp in multiplexed_qubit_pairs.values():
                        prepare(qp, prep)

                    # Branch outside the depth loop so that every iteration of the repeated block is identical
                    with switch_(exp):
                        for exp_code in (EXP_PHI, EXP_THETA_XX, EXP_THETA_YX, EXP_FLOQUET):
                            with case_(exp_code):
                                with for_(k, 0, k < ncz, k + 1):
                                    for qp in multiplexed_qubit_pairs.values():
                                        play_cycle(qp, operation, exp_code)

                    for qp in multiplexed_qubit_pairs.values():
                        readout_rotation(qp, operation, ro)
                    align()

                    # Measure both qubits and save the joint statistics (1 = excited)
                    for ii, qp in multiplexed_qubit_pairs.items():
                        qp.qubit_control.readout_state(state_control[ii])
                        qp.qubit_target.readout_state(state_target[ii])
                        assign(state_both[ii], state_control[ii] * state_target[ii])
                        save(state_control[ii], state_control_st[ii])
                        save(state_target[ii], state_target_st[ii])
                        save(state_both[ii], state_both_st[ii])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_control_st[i].buffer(num_circuits).average().save(f"state_control{i + 1}")
                state_target_st[i].buffer(num_circuits).average().save(f"state_target{i + 1}")
                state_both_st[i].buffer(num_circuits).average().save(f"state_both{i + 1}")


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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Compute the joint probabilities, extract the five CZ angles and store the intermediate curves in "ds_fit"
    and the fitted values in the "fit_results" dictionary.
    """
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fit_result["success"] else "failed")
        for qp_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot one figure per sub-experiment and one for the process matrix, with one row per qubit pair."""
    ds_fit, fit_results = node.results["ds_fit"], node.results["fit_results"]
    node.results["figures"] = {
        "meadd_phi": plot_meadd_phi(ds_fit, fit_results),
        "meadd_theta": plot_meadd_theta(ds_fit, fit_results),
        "floquet": plot_floquet(ds_fit, fit_results),
        "process_matrix": plot_process_matrix(fit_results),
    }
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the suggested virtual-Z corrections to the CZ macro of every pair whose fit succeeded."""
    operation = node.parameters.operation
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            fit_result = node.results["fit_results"][qp.name]
            qp.macros[operation].phase_shift_control = fit_result["suggested_phase_shift_control"]
            qp.macros[operation].phase_shift_target = fit_result["suggested_phase_shift_target"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
